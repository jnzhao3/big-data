"""Run Magneto (VIDA-NYU) on one GDC study with its LLM reranker pointed at local Ollama via LiteLLM.

Usage (from repo root): .venv_magneto/bin/python -I experiments/qwen_local/magneto_ollama.py --llm ollama_chat/qwen2.5:14b [--study clark] [--no-llm] [--data-dir <GDC_new>]
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
args = ap.parse_args()

d = Path(args.data_dir) / args.study
src = pd.read_csv(d / "source.csv", low_memory=False)
tgt = pd.read_csv(d / "target.csv", low_memory=False)
gold = pd.read_csv(d / "groundtruth.csv").drop_duplicates().groupby("source")["target"].apply(set).to_dict()

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
top1 = sum(best.get(s, ("",))[0] in g for s, g in gold.items())
def rank(s, g):
    ranked = sorted([(tc, sc) for ((_, ss), (_, tc)), sc in matches.items() if ss == s], key=lambda x: -x[1])
    return next((i + 1 for i, (tc, _) in enumerate(ranked) if tc in g), None)
mrr = sum(1 / r for s, g in gold.items() if (r := rank(s, g))) / len(gold)
for s, g in gold.items(): print(f"{s:32s} -> {best.get(s, ('-',0))[0]:35s} {'✓' if best.get(s, ('',))[0] in g else '✗'} gold_rank={rank(s, g)}")
print(f"[{'no-llm' if args.no_llm else args.llm}] top1={top1}/{len(gold)} MRR={mrr:.3f} secs={secs:.1f} llm_parse_failures={len(llm_warns)}")
for x in llm_warns[:3]: print("  warn:", x)
