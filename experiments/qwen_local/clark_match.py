"""Small harmonization test: for each source column of a GDC study, ask a local Qwen (Ollama)
to pick the matching target column from a shortlist. Scored with hbench against GDC-SM (default) or the
older GDC_new groundtruth.csv.

Usage: python -I clark_match.py --model qwen2.5:14b --run 1 [--mode shortlist|full] [--study clark] [--k 20]
       [--think] [--temperature T] [--seed S] [--transport ollama|openai_v1] [--num-ctx 32768] [--top-logprobs 5]
       [--engine-protocol natural|cold_start|cold] [--gt gdc-sm|old] [--data-root <datasets>] [--out-dir <results>]
       [--raw-dir <run logs>]
       [--rescore <dir with an earlier run's CSV>]   # reuse its predictions, no model calls

Outputs in --out-dir (never overwritten): <tag>.csv and a line in summary.jsonl, as before. Live runs also
write <raw-dir>/<run_id>/ (default $HBENCH_RAW or ~/hbench_raw, outside the repo): the run log (every call with
raw request/response and token logprobs), predictions.jsonl (with the shown candidates and each decision's
evidence), the hbench scores, and engine_audit.json (with each call's lineage).
"""
import argparse, json, re, sys, time
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
# Always use the hbench of this checkout, never another copy installed in the environment: the library version
# must match the commit recorded in the run header.
sys.path.insert(0, str(HERE.parents[1]))
from hbench.data import Data, default_root
from hbench.decisions import evidence
from hbench.design import make_condition
from hbench.llm import LLMClient
from hbench.predictions import sm_record, write_predictions
from hbench.runlog.session import raw_root, start_run
from hbench.score_sm import score_sm

GT_VARIANT = {"gdc-sm": "gdc-sm", "old": "gdc-new-raw"}
CALLER = "method:clark_match"
DECISION = {"type": "json_field", "field": "target"}  # stored with every call: the log says where the decision is
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data-root", default=None, help="benchmark data root (default: $HBENCH_DATA or <repo>/datasets)")
ap.add_argument("--gt", choices=["gdc-sm", "old"], default="gdc-sm",
                help="gdc-sm = GDC-SM v1.0 (259 pairs, n:m); old = GDC_new/<study>/groundtruth.csv (197 rows)")
ap.add_argument("--out-dir", default=HERE / "results")
ap.add_argument("--raw-dir", default=None, help="run logs (default: $HBENCH_RAW or ~/hbench_raw, outside the repo)")
ap.add_argument("--rescore", default=None, help="folder with an earlier run's CSV (same tag): rescore its predictions")
ap.add_argument("--run", type=int, default=1)
ap.add_argument("--study", default="clark")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--think", action="store_true")
ap.add_argument("--mode", choices=["shortlist", "full"], default="shortlist", help="full = all target names in prompt")
ap.add_argument("--temperature", type=float, default=None, help="unset = model default")
ap.add_argument("--seed", type=int, default=None, help="Ollama sampling seed (unset = random)")
ap.add_argument("--num-ctx", type=int, default=32768, help="runs before E0 v2 used 16384")
ap.add_argument("--transport", choices=["ollama", "openai_v1"], default="ollama")
ap.add_argument("--top-logprobs", type=int, default=5, help="0 = do not request logprobs")
ap.add_argument("--engine-protocol", choices=["natural", "cold_start", "cold"], default="natural",
                help="cold_start = unload the model once before the run (every repeat starts from the same state); "
                     "cold = before every call (full recompute each time, slow)")
args = ap.parse_args()

data = Data(args.data_root) if args.data_root else Data(default_root())
d = data.source_path(args.study).parent  # accepts "gillette" or GDC_new's "gilette"
src = pd.read_csv(d / "source.csv", low_memory=False)  # default parsing, as in the original prompts
tgt = pd.read_csv(d / "target.csv", low_memory=False)

def norm(s):  # "Tumor_Size_cm" -> "tumor size cm"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def samples(series, n=5):
    vals = series.dropna().astype(str).unique()
    return [v[:60] for v in vals[:n]]

tgt_vals = {c: set(v.lower() for v in tgt[c].dropna().astype(str).unique()) for c in tgt.columns}
tgt_norm = {c: norm(c) for c in tgt.columns}

def shortlist(col, k):
    """Half by name similarity, half by value overlap (cheap stand-in for embeddings)."""
    n = norm(col); toks = set(n.split())
    name = {c: SequenceMatcher(None, n, tn).ratio() + 0.5 * len(toks & set(tn.split())) / max(len(toks), 1)
            for c, tn in tgt_norm.items()}
    svals = set(v.lower() for v in src[col].dropna().astype(str).unique())
    def hit(v, tv):  # exact, or containment either way (e.g. "pt3" vs "t3", "clear cell ..." vs "clear cell ..., nos")
        return v in tv or (len(v) >= 2 and any(v in x or (len(x) >= 2 and x in v) for x in tv))
    overlap = {c: sum(hit(v, tv) for v in svals) / max(len(svals), 1) for c, tv in tgt_vals.items() if tv}
    by_name = sorted(name, key=name.get, reverse=True)
    by_val = [c for c in sorted(overlap, key=overlap.get, reverse=True) if overlap[c] > 0]
    out = []
    for c in by_val[: k // 2] + by_name:
        if c not in out: out.append(c)
        if len(out) == k: break
    return out

FULL_PROMPT = """You are matching columns from a source clinical dataset to the GDC (Genomic Data Commons) data model.

All target columns:
{cands}

Source column: "{col}"
Sample values: {vals}

Pick the ONE target column from the list above that represents the same attribute as the source column. If none fits, answer "NONE".
Respond with JSON only: {{"target": "<exact target column name or NONE>", "reason": "<one short sentence>"}}"""

PROMPT = """You are matching columns from a source clinical dataset to the GDC (Genomic Data Commons) data model.

Source column: "{col}"
Sample values: {vals}

Candidate target columns (name: sample values):
{cands}

Pick the ONE candidate that represents the same attribute as the source column. If none fits, answer "NONE".
Respond with JSON only: {{"target": "<exact candidate name or NONE>", "reason": "<one short sentence>"}}"""

tag = f"{args.study}_{args.mode}{args.k if args.mode == 'shortlist' else ''}_{args.model.replace(':', '-').replace('/', '-')}{'_think' if args.think else ''}_run{args.run}"
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
if (out / f"{tag}.csv").exists():
    sys.exit(f"{out / f'{tag}.csv'} exists; use another --run number or --out-dir")
prior, log, client = None, None, None
if args.rescore:
    prior = pd.read_csv(Path(args.rescore) / f"{tag}.csv", keep_default_na=False).set_index("source")
    run_id = tag
else:
    condition = make_condition(
        method={"name": "clark_match", "mode": args.mode, "k": args.k if args.mode == "shortlist" else None},
        model={"name": args.model},
        decoding={"temperature": args.temperature, "seed": args.seed, "think": args.think, "format": "json"},
        engine={"transport": args.transport, "num_ctx": args.num_ctx, "protocol": args.engine_protocol,
                "top_logprobs": args.top_logprobs or None},
        data={"study": args.study}, replicate=args.run)
    log = start_run(Path(args.raw_dir) if args.raw_dir else raw_root(), "qwen_local", condition,
                    task=dict(study=args.study, kind="sm"),
                    models=[args.model], prompts={"full": FULL_PROMPT, "shortlist": PROMPT}, data=data, tag=tag)
    run_id = log.run_id
    client = LLMClient(log, args.model, transport=args.transport, num_ctx=args.num_ctx,
                       top_logprobs=args.top_logprobs or None,
                       decoding={"temperature": args.temperature, "seed": args.seed, "think": args.think})

records, legacy, t0 = [], [], time.time()
for i, col in enumerate(src.columns):
    cands = list(tgt.columns) if args.mode == "full" else shortlist(col, args.k)
    if prior is not None:  # same candidates (shortlist is deterministic), earlier prediction
        p = prior.loc[col]
        raw, dt = json.dumps({"target": p["pred"], "reason": p["reason"]}), float(p["secs"])
    else:
        if args.mode == "full":  # same prefix every call -> Ollama reuses the KV cache
            prompt = FULL_PROMPT.format(col=col, vals=samples(src[col]), cands=", ".join(cands))
        else:
            cand_txt = "\n".join(f"- {c}: {samples(tgt[c], 4)}" for c in cands)
            prompt = PROMPT.format(col=col, vals=samples(src[col]), cands=cand_txt)
        res = client.chat([{"role": "user", "content": prompt}], caller=CALLER, step=i, format="json",
                          decision=DECISION)
        raw, dt = res.text or "", res.latency_s
    emitted = None
    try:
        obj = json.loads(raw); emitted = obj.get("target"); pred = str(obj.get("target", "")).strip(); reason = obj.get("reason", "")
        if pred.upper() == "NONE": pred = "NONE"
        status = "ok" if pred in cands or pred == "NONE" else ("hallucinated_in_target" if pred in tgt.columns else "hallucinated")
    except Exception:
        pred, reason, status = "", raw[:200], "invalid_json"
    ev = None if prior is not None else evidence(res.span_id, res.token_logprobs, raw,
                                                 emitted if isinstance(emitted, str) else None, json_key="target",
                                                 temperature=args.temperature, sampler_params=log.effective_decoding,
                                                 grammar=True, prompt_sha256=res.prompt_sha256)  # format=json
    records.append(sm_record(run_id, args.study, col, [] if pred in ("", "NONE") else [pred],
                             shown="*" if args.mode == "full" else cands,
                             status="invalid_output" if status == "invalid_json" else "ok",
                             latency_s=round(dt, 1), reason=reason, legacy_status=status, evidence=ev))
    legacy.append(dict(source=col, pred=pred, status=status, secs=round(dt, 1), reason=reason))

items, s = score_sm(records, data, args.study, GT_VARIANT[args.gt])
items = items.set_index("source_column")
rows = []
for r in legacy:
    it = items.loc[r["source"]]
    rows.append(dict(source=r["source"], gold=it["gold"] or "NONE",
                     in_shortlist=bool(it["candidate_recall"]) if it["annotated"] else True,
                     pred=r["pred"], correct=bool(it["closed_world_correct"]), status=r["status"], secs=r["secs"],
                     reason=r["reason"]))
    print(f"{r['source']:32s} -> {r['pred']:35s} {'✓' if it['closed_world_correct'] else '✗'} ({r['status']}, {r['secs']:.1f}s)", flush=True)

df = pd.DataFrame(rows)
total_secs = round(time.time() - t0, 1)
if prior is not None:  # keep the original run's wall time
    old = [json.loads(l) for l in open(Path(args.rescore) / "summary.jsonl")]
    total_secs = next((x["total_secs"] for x in old if x["tag"] == tag), round(df.secs.sum(), 1))
df.to_csv(out / f"{tag}.csv", index=False)
summary = dict(tag=tag, gt=args.gt, total_secs=total_secs, n_cols=len(df),
               acc_gold_cols=f"{s['top1_correct']}/{s['n_annotated']}",
               shortlist_recall=f"{int(items[items.annotated].candidate_recall.sum())}/{s['n_annotated']}",
               acc_all_incl_NONE=f"{s['closed_world_correct']}/{s['n_columns']}",
               statuses=df.status.value_counts().to_dict(), run_id=run_id, scoring_version=s["scoring_version"])
if log is not None:
    write_predictions(log.run_dir / "predictions.jsonl", records)
    log.emit("prediction_batch", path="predictions.jsonl", n=len(records))
    log.emit("score", gt=s["gt"], summary=s)
    log.write_file(f"items.{GT_VARIANT[args.gt]}.csv", items.reset_index().to_csv(index=False))
    log.close(status="ok", total_secs=total_secs)
    m = items[items.margin.notna()].margin if "margin" in items else pd.Series(dtype=float)
    if len(m):
        print(f"decision margins: min {m.min():.4f}, median {m.median():.4f}, below 0.05: {int((m < 0.05).sum())}")
print(json.dumps(summary, ensure_ascii=False))
with open(out / "summary.jsonl", "a") as f: f.write(json.dumps(summary, ensure_ascii=False) + "\n")
