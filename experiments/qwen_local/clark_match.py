"""Small harmonization test: for each source column of a GDC study, ask a local Qwen (Ollama)
to pick the matching target column from a shortlist. Scores against GDC-SM (default) or the
older GDC_new groundtruth.csv.

Usage: python -I clark_match.py --model qwen2.5:14b --run 1 [--mode shortlist|full] [--study clark] [--k 20]
       [--think] [--gt gdc-sm|old] [--data-dir <GDC_new>] [--gt-dir <GDC-SM ground-truth>] [--out-dir <results>]
       [--rescore <dir with an earlier run's CSV>]   # reuse its predictions, no model calls
"""
import argparse, json, os, re, time, urllib.request
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data-dir", default=os.environ.get("GDC_DATA_DIR", HERE.parents[1] / "datasets" / "GDC_new"),
                help="folder holding the GDC_new studies (default: $GDC_DATA_DIR or <repo>/datasets/GDC_new)")
ap.add_argument("--gt", choices=["gdc-sm", "old"], default="gdc-sm",
                help="gdc-sm = GDC-SM v1.0 (259 pairs, n:m); old = GDC_new/<study>/groundtruth.csv (197 rows)")
ap.add_argument("--gt-dir", default=os.environ.get("GDC_SM_GT_DIR", HERE.parents[1] / "datasets" / "gdc_sm" / "extracted" / "data" / "ground-truth"),
                help="GDC-SM ground-truth folder (default: $GDC_SM_GT_DIR or <repo>/datasets/gdc_sm/extracted/data/ground-truth)")
ap.add_argument("--out-dir", default=HERE / "results")
ap.add_argument("--rescore", default=None, help="folder with an earlier run's CSV (same tag): rescore its predictions")
ap.add_argument("--run", type=int, default=1)
ap.add_argument("--study", default="clark")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--think", action="store_true")
ap.add_argument("--mode", choices=["shortlist", "full"], default="shortlist", help="full = all target names in prompt")
ap.add_argument("--temperature", type=float, default=None, help="unset = model default")
args = ap.parse_args()

d = Path(args.data_dir) / args.study
src = pd.read_csv(d / "source.csv", low_memory=False)
tgt = pd.read_csv(d / "target.csv", low_memory=False)

def load_gold():
    """{source column: set of correct target columns}. Columns not listed should be answered NONE."""
    if args.gt == "old":
        gt = pd.read_csv(d / "groundtruth.csv", dtype=str).drop_duplicates()
    else:  # GDC-SM file names are capitalized; GDC_new spells Gillette as "gilette"
        key = {"gilette": "gillette"}.get(args.study.lower(), args.study.lower())
        path = next(f for f in Path(args.gt_dir).glob("*.csv") if f.stem.lower() == key)
        gt = pd.read_csv(path, dtype=str)
    gt = gt.iloc[:, :2].apply(lambda c: c.str.strip())
    gt.columns = ["source", "target"]
    return gt.groupby("source")["target"].apply(set).to_dict()

gold = load_gold()

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

def ask(prompt):
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "format": "json", "think": args.think, "options": {"num_ctx": 16384}}
    if args.temperature is not None: body["options"]["temperature"] = args.temperature
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["message"].get("content", ""), time.time() - t

tag = f"{args.study}_{args.mode}{args.k if args.mode == 'shortlist' else ''}_{args.model.replace(':', '-').replace('/', '-')}{'_think' if args.think else ''}_run{args.run}"
prior = None
if args.rescore:
    prior = pd.read_csv(Path(args.rescore) / f"{tag}.csv", keep_default_na=False).set_index("source")

rows, t0 = [], time.time()
for col in src.columns:
    cands = list(tgt.columns) if args.mode == "full" else shortlist(col, args.k)
    if prior is not None:  # same candidates (shortlist is deterministic), earlier prediction
        p = prior.loc[col]
        raw, dt = json.dumps({"target": p["pred"], "reason": p["reason"]}), float(p["secs"])
    elif args.mode == "full":  # same prefix every call -> Ollama reuses the KV cache
        raw, dt = ask(FULL_PROMPT.format(col=col, vals=samples(src[col]), cands=", ".join(cands)))
    else:
        cand_txt = "\n".join(f"- {c}: {samples(tgt[c], 4)}" for c in cands)
        raw, dt = ask(PROMPT.format(col=col, vals=samples(src[col]), cands=cand_txt))
    try:
        obj = json.loads(raw); pred = str(obj.get("target", "")).strip(); reason = obj.get("reason", "")
        if pred.upper() == "NONE": pred = "NONE"
        status = "ok" if pred in cands or pred == "NONE" else ("hallucinated_in_target" if pred in tgt.columns else "hallucinated")
    except Exception:
        pred, reason, status = "", raw[:200], "invalid_json"
    g = gold.get(col, set())
    correct = (pred in g) if g else (pred == "NONE")
    rows.append(dict(source=col, gold="|".join(sorted(g)) or "NONE", in_shortlist=bool(g & set(cands)) if g else True,
                     pred=pred, correct=correct, status=status, secs=round(dt, 1), reason=reason))
    print(f"{col:32s} -> {pred:35s} {'✓' if correct else '✗'} ({status}, {dt:.1f}s)", flush=True)

df = pd.DataFrame(rows)
total_secs = round(time.time() - t0, 1)
if prior is not None:  # keep the original run's wall time
    old = [json.loads(l) for l in open(Path(args.rescore) / "summary.jsonl")]
    total_secs = next((x["total_secs"] for x in old if x["tag"] == tag), round(df.secs.sum(), 1))
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / f"{tag}.csv", index=False)
has_gold = df[df.gold != "NONE"]
summary = dict(tag=tag, gt=args.gt, total_secs=total_secs, n_cols=len(df),
               acc_gold_cols=f"{has_gold.correct.sum()}/{len(has_gold)}",
               shortlist_recall=f"{has_gold.in_shortlist.sum()}/{len(has_gold)}",
               acc_all_incl_NONE=f"{df.correct.sum()}/{len(df)}", statuses=df.status.value_counts().to_dict())
print(json.dumps(summary, ensure_ascii=False))
with open(out / "summary.jsonl", "a") as f: f.write(json.dumps(summary, ensure_ascii=False) + "\n")
