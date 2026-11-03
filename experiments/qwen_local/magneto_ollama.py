"""Run Magneto (VIDA-NYU) on one GDC study with its LLM reranker pointed at local Ollama via LiteLLM.
Scores the same matches with hbench against GDC-SM (259 pairs, n:m) and the older GDC_new groundtruth.csv.

Usage (from repo root, shared env from env/): ~/venvs/harmonize/bin/python -I experiments/qwen_local/magneto_ollama.py
       [--llm openai/qwen2.5:14b] [--study clark] [--no-llm] [--temperature 0] [--seed S] [--run N]
       [--top-logprobs 5] [--engine-protocol natural|cold_start] [--data-root <datasets>] [--out-dir <results>]
       [--raw-dir <run logs>]

The reranker goes through LiteLLM's openai/ route to Ollama's /v1 by default: that route keeps logprobs and the
cached-token count (ollama_chat/ rejects logprobs). Runs before E0 v2 used ollama_chat/.
Each run writes <raw-dir>/<run_id>/ (default $HBENCH_RAW or ~/hbench_raw, outside the repo): the run log (every
reranker call with raw request/response), predictions.jsonl with the full ranking for every column, each column
linked to its reranker call with the LLM's score gap, the hbench scores, and engine_audit.json. Nothing is
overwritten.
"""
import argparse, json, re, sys, time, warnings
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
# Always use the hbench of this checkout, never another copy installed in the environment: the library version
# must match the commit recorded in the run header.
sys.path.insert(0, str(HERE.parents[1]))
from hbench.adapters.magneto import from_magneto_matches
from hbench.data import Data, default_root
from hbench.design import make_condition
from hbench.predictions import write_predictions
from hbench.runlog.session import raw_root, start_run
from hbench.runlog.writer import BlobStore, read_events
from hbench.score_sm import score_sm
from magneto import Magneto

CALLER = "method:magneto_reranker"
ap = argparse.ArgumentParser()
ap.add_argument("--llm", default="openai/qwen2.5:14b")
ap.add_argument("--study", default="clark")
ap.add_argument("--no-llm", action="store_true")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--top-logprobs", type=int, default=5, help="openai/ route only; 0 = do not request")
ap.add_argument("--run", type=int, default=1, help="repeat index, recorded in the condition")
ap.add_argument("--engine-protocol", choices=["natural", "cold_start"], default="natural",
                help="cold_start = unload the model once before the run (LiteLLM routes allow no per-call action)")
ap.add_argument("--data-root", default=None, help="benchmark data root (default: $HBENCH_DATA or <repo>/datasets)")
ap.add_argument("--out-dir", default=HERE / "results")
ap.add_argument("--raw-dir", default=None, help="run logs (default: $HBENCH_RAW or ~/hbench_raw, outside the repo)")
args = ap.parse_args()

data = Data(args.data_root) if args.data_root else Data(default_root())
d = data.source_path(args.study).parent
src = pd.read_csv(d / "source.csv", low_memory=False)
tgt = pd.read_csv(d / "target.csv", low_memory=False)

route, model = (args.llm.split("/", 1) + [None])[:2] if "/" in args.llm else ("ollama_chat", args.llm)
kw = dict(embedding_model="mpnet", topk=20)
llm_kwargs = {"temperature": args.temperature}
if route == "openai":
    llm_kwargs.update(api_base="http://localhost:11434/v1", api_key="ollama")
    if args.top_logprobs:
        llm_kwargs.update(logprobs=True, top_logprobs=args.top_logprobs)
else:
    llm_kwargs["api_base"] = "http://localhost:11434"
if args.seed is not None:
    llm_kwargs["seed"] = args.seed
if not args.no_llm:
    kw.update(use_gpt_reranker=True, llm_model=args.llm, llm_model_kwargs=llm_kwargs)
label = "no-llm" if args.no_llm else args.llm

condition = make_condition(
    method={"name": "magneto", "embedding_model": "mpnet", "topk": 20, "reranker": not args.no_llm},
    model={"name": "none" if args.no_llm else model},
    decoding={"temperature": None if args.no_llm else args.temperature, "seed": args.seed},
    engine={"transport": "none" if args.no_llm else f"litellm/{route}", "protocol": args.engine_protocol,
            "top_logprobs": args.top_logprobs if route == "openai" and not args.no_llm else None},
    data={"study": args.study}, replicate=args.run)
log = start_run(Path(args.raw_dir) if args.raw_dir else raw_root(), "qwen_local", condition,
                task=dict(study=args.study, kind="sm"),
                models=[] if args.no_llm else [model], data=data,
                llm=None if args.no_llm else dict(model=args.llm, **{k: v for k, v in llm_kwargs.items() if k != "api_key"}))
cb = None
if not args.no_llm:
    from hbench.runlog import litellm_cb
    cb = litellm_cb.install(log, caller=CALLER)

t = time.time()
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        matches = Magneto(**kw).get_matches(src, tgt)
        llm_warns = [str(x.message)[:150] for x in w if "parse" in str(x.message).lower()]
finally:
    if cb is not None and not litellm_cb.uninstall(cb):
        print("warning: some reranker calls were not logged", file=sys.stderr)
secs = time.time() - t


def reranker_evidence() -> dict:
    """Source column -> evidence from its (last successful) reranker call: the call's span and the gap between
    the LLM's top two scores. The column is read from the prompt's last "Candidate Column: Column: <name>,"."""
    blobs, out = BlobStore(log.run_dir.parent / "_blobs"), {}
    for e in read_events(log.run_dir):
        if e["type"] != "llm_call" or e.get("caller") != CALLER or e.get("error") or not e.get("messages"):
            continue
        prompt = blobs.get(e["messages"][-1]).get("content") or ""
        cols = re.findall(r"Candidate Column: Column: (.*?), Sample values:", prompt)
        if not cols:
            continue
        gap = None
        try:
            scores = sorted((float(x.get("score", 0)) for x in json.loads((e.get("output") or {}).get("text") or "")
                             if isinstance(x, dict)), reverse=True)
            gap = scores[0] - scores[1] if len(scores) > 1 else None
        except (ValueError, TypeError):
            pass
        out[cols[-1]] = {"span_id": e["span_id"], "llm_score_gap": gap, "margin": None,
                         "margin_note": "reranker outputs a score list; logprob margin not defined"}
    return out


records = from_magneto_matches(matches, log.run_id, args.study, source_columns=list(src.columns))
if not args.no_llm:
    ev = reranker_evidence()
    for r in records:
        if r["source_column"] in ev:
            r["evidence"] = ev[r["source_column"]]
write_predictions(log.run_dir / "predictions.jsonl", records)
log.emit("prediction_batch", path="predictions.jsonl", n=len(records), parse_failures=len(llm_warns))
for name, variant in [("GDC-SM", "gdc-sm"), ("old GT", "gdc-new-raw")]:
    items, s = score_sm(records, data, args.study, variant)
    log.emit("score", gt=s["gt"], summary=s)
    log.write_file(f"items.{variant}.csv", items.to_csv(index=False))
    print(f"--- scored against {name}")
    for it in items[items.annotated].sort_values("source_column").itertuples(index=False):
        rank = None if pd.isna(it.rank) else int(it.rank)
        print(f"{it.source_column:32s} -> {it.pred if it.pred != 'NONE' else '-':35s} {'✓' if it.correct else '✗'} gold_rank={rank}")
    print(f"[{label} | {name}] top1={s['top1_correct']}/{s['n_annotated']} MRR={s['mrr']:.3f} "
          f"hit@5={s['hit@5']:.3f} vida_pair_MRR={s['vida_pair_mrr']:.3f}")
log.close(status="ok", secs=round(secs, 1), llm_parse_failures=len(llm_warns))
print(f"[{label}] secs={secs:.1f} llm_parse_failures={len(llm_warns)} run={log.run_dir}")
for x in llm_warns[:3]: print("  warn:", x)
