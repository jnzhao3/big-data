"""Run Magneto (VIDA-NYU) on one GDC study with its LLM reranker pointed at local Ollama via LiteLLM.
Scores the same matches with hbench against GDC-SM (259 pairs, n:m) and the older GDC_new groundtruth.csv.

Usage (from repo root, shared env from env/): ~/venvs/harmonize/bin/python -I experiments/qwen_local/magneto_ollama.py --llm ollama_chat/qwen2.5:14b
       [--study clark] [--no-llm] [--temperature 0] [--seed S] [--run N] [--data-root <datasets>] [--out-dir <results>]

Each run writes <out-dir>/raw/<run_id>/: the run log (every reranker LLM call), predictions.jsonl with the
full ranking and scores for every column, and the hbench scores. Nothing is overwritten.
"""
import argparse, sys, time, warnings
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
try:
    import hbench  # noqa: F401
except ImportError:  # hbench not installed in this env: use the repo copy
    sys.path.insert(0, str(HERE.parents[1]))
from hbench.adapters.magneto import from_magneto_matches
from hbench.data import Data, default_root
from hbench.predictions import write_predictions
from hbench.runlog import RunLog
from hbench.runlog.header import collect_run_header
from hbench.score_sm import score_sm
from magneto import Magneto

ap = argparse.ArgumentParser()
ap.add_argument("--llm", default="ollama_chat/qwen2.5:14b")
ap.add_argument("--study", default="clark")
ap.add_argument("--no-llm", action="store_true")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--run", type=int, default=1, help="repeat index, recorded in the run header")
ap.add_argument("--data-root", default=None, help="benchmark data root (default: $HBENCH_DATA or <repo>/datasets)")
ap.add_argument("--out-dir", default=HERE / "results")
args = ap.parse_args()

data = Data(args.data_root) if args.data_root else Data(default_root())
d = data.source_path(args.study).parent
src = pd.read_csv(d / "source.csv", low_memory=False)
tgt = pd.read_csv(d / "target.csv", low_memory=False)

kw = dict(embedding_model="mpnet", topk=20)
llm_kwargs = {"api_base": "http://localhost:11434", "temperature": args.temperature}
if args.seed is not None:
    llm_kwargs["seed"] = args.seed
if not args.no_llm:
    kw.update(use_gpt_reranker=True, llm_model=args.llm, llm_model_kwargs=llm_kwargs)
label = "no-llm" if args.no_llm else args.llm

log = RunLog.create(Path(args.out_dir) / "raw", "qwen_local",
                    f"magneto_{'nollm' if args.no_llm else args.llm.split('/')[-1]}_run{args.run}", args.study)
log.emit("run_header", **collect_run_header(
    "qwen_local/magneto", condition=dict(embedding_model="mpnet", topk=20, reranker=None if args.no_llm else args.llm,
                                         temperature=None if args.no_llm else args.temperature, seed=args.seed,
                                         run=args.run),
    task=dict(study=args.study, kind="sm"), llm=None if args.no_llm else dict(model=args.llm, **llm_kwargs),
    data=data, ollama_models=[] if args.no_llm else [args.llm]))
cb = None
if not args.no_llm:
    from hbench.runlog import litellm_cb
    cb = litellm_cb.install(log, caller="method:magneto_reranker")

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

records = from_magneto_matches(matches, log.run_id, args.study, source_columns=list(src.columns))
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
