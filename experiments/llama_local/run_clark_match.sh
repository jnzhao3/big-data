#!/usr/bin/env bash
# clark_match schema matching with a local model on every GDC study: both candidate modes, RUNS repeats each,
# at T=0 with seed 1, every run starting from an unloaded model (cold_start).
#
# Usage (from anywhere):  bash experiments/llama_local/run_clark_match.sh
# Settings, all optional, as environment variables:
#   MODEL    default llama3.1:8b
#   OUT      default ~/hbench_raw/llama_eval   (outside the repo: summary.jsonl, per-run CSVs, raw/ run folders)
#   STUDIES  default all 10
#   MODES    default "shortlist full"
#   RUNS     default "1 2"
#   PYTHON   default ~/venvs/harmonize/bin/python
#
# Safe to re-run: runs whose CSV already exists in OUT are skipped, so an interrupted sweep continues where it
# stopped. Nothing is overwritten. Keep Ollama free for other clients while it runs.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$HOME/venvs/harmonize/bin/python}"
MODEL="${MODEL:-llama3.1:8b}"
OUT="${OUT:-$HOME/hbench_raw/llama_eval}"
STUDIES="${STUDIES:-cao clark dou gilette huang krug mcdermott satpathy vasaikar wang}"
MODES="${MODES:-shortlist full}"
RUNS="${RUNS:-1 2}"
TAG_MODEL="${MODEL//:/-}"; TAG_MODEL="${TAG_MODEL//\//-}"

cd "$REPO"
mkdir -p "$OUT/raw"
LOG="$OUT/run_clark_match.log"

{
  echo "=== clark_match sweep started $(date -u +%FT%TZ)"
  echo "repo commit: $(git rev-parse HEAD)$(git diff --quiet HEAD -- hbench experiments || echo ' (uncommitted changes)')"
  echo "model: $MODEL | studies: $STUDIES | modes: $MODES | runs: $RUNS"
  echo "ollama: $(curl -s -m 5 localhost:11434/api/version || echo 'NOT REACHABLE')"
  "$PY" -m hbench engine check --models "$MODEL" || echo "(engine check reported drift; runs continue and are flagged)"

  failed=()
  for study in $STUDIES; do
    for mode in $MODES; do
      k=""; [ "$mode" = shortlist ] && k=20
      for run in $RUNS; do
        tag="${study}_${mode}${k}_${TAG_MODEL}_run${run}"
        if [ -e "$OUT/$tag.csv" ]; then
          echo "--- skip $tag (already done)"
          continue
        fi
        echo "--- $tag  $(date -u +%T)"
        if ! "$PY" -I experiments/qwen_local/clark_match.py --model "$MODEL" --study "$study" --mode "$mode" \
             --run "$run" --temperature 0 --seed 1 --engine-protocol cold_start \
             --out-dir "$OUT" --raw-dir "$OUT/raw"; then
          failed+=("$tag")
        fi
      done
    done
  done

  echo "=== finished $(date -u +%FT%TZ)"
  if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED (${#failed[@]}): ${failed[*]}"
    exit 1
  fi
  echo "all runs done; results: $OUT/summary.jsonl"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
