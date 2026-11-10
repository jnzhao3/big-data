#!/usr/bin/env bash
# Magneto schema matching with a local model as its LLM reranker on every GDC study, at T=0, every run starting
# from an unloaded model (cold_start). Slower than clark_match: several minutes per study.
#
# Usage (from anywhere):  bash experiments/llama_local/run_magneto.sh
# Settings, all optional, as environment variables:
#   MODEL    default llama3.1:8b   (called through LiteLLM's openai/ route, which keeps logprobs)
#   OUT      default ~/hbench_raw/llama_eval
#   STUDIES  default all 10
#   RUNS     default "1"
#   PYTHON   default ~/venvs/harmonize/bin/python
#
# Each run's printed scores go to OUT/magneto/<study>_run<N>.txt; runs whose file exists are skipped.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$HOME/venvs/harmonize/bin/python}"
MODEL="${MODEL:-llama3.1:8b}"
OUT="${OUT:-$HOME/hbench_raw/llama_eval}"
STUDIES="${STUDIES:-cao clark dou gilette huang krug mcdermott satpathy vasaikar wang}"
RUNS="${RUNS:-1}"
TAG_MODEL="${MODEL//:/-}"; TAG_MODEL="${TAG_MODEL//\//-}"

cd "$REPO"
mkdir -p "$OUT/raw" "$OUT/magneto"
LOG="$OUT/run_magneto.log"

{
  echo "=== magneto sweep started $(date -u +%FT%TZ)"
  echo "repo commit: $(git rev-parse HEAD)$(git diff --quiet HEAD -- hbench experiments || echo ' (uncommitted changes)')"
  echo "model: $MODEL | studies: $STUDIES | runs: $RUNS"
  echo "ollama: $(curl -s -m 5 localhost:11434/api/version || echo 'NOT REACHABLE')"
  "$PY" -m hbench engine check --models "$MODEL" || echo "(engine check reported drift; runs continue and are flagged)"

  failed=()
  for study in $STUDIES; do
    for run in $RUNS; do
      res="$OUT/magneto/${study}_${TAG_MODEL}_run${run}.txt"
      if [ -e "$res" ]; then
        echo "--- skip magneto $study run$run (already done)"
        continue
      fi
      echo "--- magneto $study run$run  $(date -u +%T)"
      if "$PY" -I experiments/qwen_local/magneto_ollama.py --llm "openai/$MODEL" --study "$study" --run "$run" \
           --temperature 0 --engine-protocol cold_start --raw-dir "$OUT/raw" > "$res.partial" 2>&1; then
        mv "$res.partial" "$res"
        grep -E '^\[.*GDC-SM\]|secs=' "$res"
      else
        echo "failed; output kept in $res.partial"
        failed+=("$study:run$run")
      fi
    done
  done

  echo "=== finished $(date -u +%FT%TZ)"
  if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED (${#failed[@]}): ${failed[*]}"
    exit 1
  fi
  echo "all runs done; per-study scores in $OUT/magneto/"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
