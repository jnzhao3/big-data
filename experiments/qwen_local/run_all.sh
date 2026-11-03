#!/bin/zsh
# Runs every clark config twice. Extra args go to clark_match.py, e.g. ./run_all.sh --out-dir results/repeat2 (existing CSVs are never overwritten)
# Set PYTHON to pick the interpreter (needs pandas). Log: results/run_all.log
PY=${PYTHON:-python}
cd "${0:A:h}"
mkdir -p results
for m in qwen2.5:14b qwen3.8:27b-q4_K_M; do
  for mode in shortlist full; do
    for r in 1 2; do
      echo "=== $m $mode run$r ==="; $PY -I clark_match.py --model $m --mode $mode --run $r "$@"
    done
  done
done 2>&1 | tee -a results/run_all.log
