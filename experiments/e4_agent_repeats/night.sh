#!/bin/zsh
# One E4 night: run a plan with E1's run_plan.py (resumable), then write the morning summary.
#
#   experiments/e4_agent_repeats/night.sh <plan.json> <root>
#
# Run it from the pinned checkout (~/bigdata-wt/run-e4-<commit>), plugged in, lid open, with the app's keep-awake
# on. <root> holds plan.json, execution.jsonl, night.log and summary.md; it must be outside the repo and must not
# be the raw dir (runs go to the plan's raw_dir, ~/hbench_raw). Rerun the same command to resume after a stop.
# HBENCH_DATA is the benchmark data root. A pinned checkout (a git worktree) has no datasets/, so it defaults to
# the datasets/ of the main checkout of the same repository; the night refuses to start if a study's files are
# missing there.
set -u
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <plan.json> <root>" >&2
  exit 2
fi
if [[ -z "${E4_NIGHT_CAFFEINATED:-}" ]]; then
  export E4_NIGHT_CAFFEINATED=1
  exec caffeinate -i -s "$0" "$@"  # no idle or system sleep while the night runs
fi

HERE=${0:A:h}
REPO=${HERE:h:h}
PY=${E4_PYTHON:-$HOME/venvs/harmonize/bin/python}
if [[ -z "${HBENCH_DATA:-}" ]]; then
  if [[ -d "$REPO/datasets/GDC_new" ]]; then
    export HBENCH_DATA="$REPO/datasets"
  else  # the main checkout: the parent of the repository's common .git directory
    common=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
    export HBENCH_DATA="${common:h}/datasets"
  fi
fi
PLAN=${1:A}
ROOT=${2:A}
mkdir -p "$ROOT"
LOG="$ROOT/night.log"
say() { print -r -- "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "night start: plan $PLAN, root $ROOT, repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' DIRTY'))"
if ! curl -sf --max-time 10 http://127.0.0.1:11434/api/version >/dev/null; then
  say "Ollama is not answering on 127.0.0.1:11434; start it (brew services start ollama) and rerun"
  exit 1
fi
if pgrep -f "i4Tools" >/dev/null 2>&1; then
  say "WARNING: i4Tools is running. It starts osascript every 0.5 s against Finder; WindowServer died at 02:45 on" \
      "11/12 with it running and took the login session (and Ollama) down. Quit it before a night."
fi
say "HBENCH_DATA=$HBENCH_DATA"
"$PY" -s "$HERE/make_plan.py" prepare "$PLAN" 2>&1 | tee -a "$LOG"
if [[ ${pipestatus[1]} -ne 0 ]]; then
  say "prepare failed (data root or T=0 cache); not starting the night"
  exit 1
fi

"$PY" -s "$REPO/experiments/e1_noise_floor/run_plan.py" run "$PLAN" --root "$ROOT" --resume 2>&1 | tee -a "$LOG"
rc=${pipestatus[1]}
say "run_plan exit $rc"

"$PY" -s "$HERE/make_plan.py" summary --root "$ROOT" 2>&1 | tee -a "$LOG"
say "night end; summary in $ROOT/summary.md"
exit $rc
