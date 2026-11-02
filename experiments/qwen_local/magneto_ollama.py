"""Run Magneto (VIDA-NYU) on one GDC study with its LLM reranker pointed at local Ollama via LiteLLM.
Scores the same matches against GDC-SM (259 pairs, n:m) and the older GDC_new groundtruth.csv (197 rows).

Usage (from repo root): .venv_magneto/bin/python -I experiments/qwen_local/magneto_ollama.py --llm ollama_chat/qwen2.5:14b
       [--study clark] [--no-llm] [--data-dir <GDC_new>] [--gt-dir <GDC-SM ground-truth>]
"""
import argparse, os, time, warnings
from pathlib import Path
import pandas as pd
from magneto import Magneto

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--llm", default="ollama_chat/qwen2.5:14b")
ap.add_argument("--study", default="clark")
ap.add_argument("--no-llm", action="store_true")
ap.add_argument("--data-dir", default=os.environ.get("GDC_DATA_DIR", HERE.parents[1] / "datasets" / "GDC_new"),
                help="folder holding the GDC_new studies (default: $GDC_DATA_DIR or <repo>/datasets/GDC_new)")
ap.add_argument("--gt-dir", default=os.environ.get("GDC_SM_GT_DIR", HERE.parents[1] / "datasets" / "gdc_sm" / "extracted" / "data" / "ground-truth"),
                help="GDC-SM ground-truth folder (default: $GDC_SM_GT_DIR or <repo>/datasets/gdc_sm/extracted/data/ground-truth)")
args = ap.parse_args()

d = Path(args.data_dir) / args.study
src = pd.read_csv(d / "source.csv", low_memory=False)
tgt = pd.read_csv(d / "target.csv", low_memory=False)

def to_gold(gt):
    gt = gt.iloc[:, :2].apply(lambda c: c.str.strip())
    gt.columns = ["source", "target"]
    return gt.groupby("source")["target"].apply(set).to_dict()

key = {"gilette": "gillette"}.get(args.study.lower(), args.study.lower())  # GDC-SM spells it Gillette
golds = {"GDC-SM": to_gold(pd.read_csv(next(f for f in Path(args.gt_dir).glob("*.csv") if f.stem.lower() == key), dtype=str)),
         "old GT": to_gold(pd.read_csv(d / "groundtruth.csv", dtype=str).drop_duplicates())}

kw = dict(embedding_model="mpnet", topk=20)
if not args.no_llm:
    kw.update(use_gpt_reranker=True, llm_model=args.llm,
              llm_model_kwargs={"api_base": "http://localhost:11434", "temperature": 0})
t = time.time()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    matches = Magneto(**kw).get_matches(src, tgt)
    llm_warns = [str(x.message)[:150] for x in w if "parse" in str(x.message).lower()]
secs = time.time() - t

best = {}
for ((_, s), (_, tcol)), score in matches.items():
    if s not in best or score > best[s][1]: best[s] = (tcol, score)
def rank(s, g):
    ranked = sorted([(tc, sc) for ((_, ss), (_, tc)), sc in matches.items() if ss == s], key=lambda x: -x[1])
    return next((i + 1 for i, (tc, _) in enumerate(ranked) if tc in g), None)
label = "no-llm" if args.no_llm else args.llm
for name, gold in golds.items():
    print(f"--- scored against {name}")
    for s, g in gold.items(): print(f"{s:32s} -> {best.get(s, ('-',0))[0]:35s} {'✓' if best.get(s, ('',))[0] in g else '✗'} gold_rank={rank(s, g)}")
    top1 = sum(best.get(s, ("",))[0] in g for s, g in gold.items())
    mrr = sum(1 / r for s, g in gold.items() if (r := rank(s, g))) / len(gold)
    print(f"[{label} | {name}] top1={top1}/{len(gold)} MRR={mrr:.3f}")
print(f"[{label}] secs={secs:.1f} llm_parse_failures={len(llm_warns)}")
for x in llm_warns[:3]: print("  warn:", x)
