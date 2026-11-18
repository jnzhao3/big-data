#!/usr/bin/env bash
# Launch one locked-down Claude Code value-matching session on one GDC study.
#
# Usage: experiments/claude_vm/run_eval.sh --study <study> --mode oracle|end_to_end
#                                          [--allow-sql] [--search-only [--search-top-k K]] [--allow-code]
#                                          [--model <model>] [--row-cap N] [--headless]
#
#   --study      one of: cao clark dou gilette huang krug mcdermott satpathy vasaikar wang
#   --mode       oracle (attribute pairs given) or end_to_end (the model picks the GDC attributes)
#   --allow-sql  expose the read-only `query` tool (ALLOW_SQL=true)
#   --search-only  no list_permissible: the model finds permissible values only with search_permissible
#                (at most --search-top-k results per call, default 5), so every candidate it considers is
#                logged. Cannot be combined with --allow-sql
#   --allow-code  let the model write, save and run its own Python (save_code / run_code / list_code), in a
#                restricted interpreter with no file or network access (not an OS sandbox; see README)
#   --model      Claude Code --model (default: your Claude Code default)
#   --headless   run with `claude -p` and save the stream-json output, instead of an interactive session
#
# Environment: CLAUDE_BIN (the claude binary; default: `claude` on PATH, else the VS Code extension's binary),
# PYTHON (default ~/venvs/harmonize/bin/python), MATCH_EVAL_LOG_ROOT (default ~/hbench_raw/claude_vm).
#
# After Claude Code exits, the session is archived, converted to predictions.jsonl, scored with hbench, and
# written to session.json for the viewer (experiments/claude_vm/viewer/serve.py).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$HOME/venvs/harmonize/bin/python}"
STUDY="" MODE="" ALLOW_SQL="" MODEL="" ROW_CAP=50 HEADLESS="" SEARCH_ONLY="" SEARCH_TOP_K=5 ALLOW_CODE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --study) STUDY="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --allow-sql) ALLOW_SQL=1; shift ;;
    --search-only) SEARCH_ONLY=1; shift ;;
    --search-top-k) SEARCH_TOP_K="$2"; shift 2 ;;
    --allow-code) ALLOW_CODE=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --row-cap) ROW_CAP="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$STUDY" ] && [ -n "$MODE" ] || { echo "--study and --mode are required (see --help)" >&2; exit 2; }

# The Claude Code binary
CLAUDE="${CLAUDE_BIN:-$(command -v claude || true)}"
if [ -z "$CLAUDE" ]; then
  CLAUDE="$(ls -d "$HOME"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude \
                  "$HOME"/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null \
            | sort -V | tail -1)"
fi
[ -x "$CLAUDE" ] || { echo "Claude Code binary not found; set CLAUDE_BIN" >&2; exit 2; }
CLAUDE_VERSION="$("$CLAUDE" --version 2>/dev/null | awk '{print $1}')"

# Create the session: id, workspace, MCP config, index entry. Prints `export` lines.
START_ARGS=(--study "$STUDY" --mode "$MODE" --row-cap "$ROW_CAP" --claude-version "$CLAUDE_VERSION")
[ -n "$ALLOW_SQL" ] && START_ARGS+=(--allow-sql)
[ -n "$SEARCH_ONLY" ] && START_ARGS+=(--search-only --search-top-k "$SEARCH_TOP_K")
[ -n "$ALLOW_CODE" ] && START_ARGS+=(--allow-code)
[ -n "$MODEL" ] && START_ARGS+=(--model "$MODEL")
[ -n "$HEADLESS" ] && START_ARGS+=(--headless)
EXPORTS="$("$PY" -I "$HERE/session.py" start "${START_ARGS[@]}")" || { echo "session start failed" >&2; exit 1; }
eval "$EXPORTS"

DENY="$("$PY" -I -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["permissions"]["deny"]))' \
        "$WORKSPACE/.claude/settings.json")"

echo "session $SESSION_ID  study=$MATCH_EVAL_STUDY  mode=$MATCH_EVAL_MODE  sql=$ALLOW_SQL  search_only=$MATCH_EVAL_SEARCH_ONLY  code=$MATCH_EVAL_ALLOW_CODE  claude=$CLAUDE_VERSION"
echo "workspace $WORKSPACE"
echo "logs      $LOG_ROOT"

# Lockdown flags (verified against Claude Code $CLAUDE_VERSION's --help and the docs):
#   --tools ToolSearch,WaitForMcpServers   the only built-in tools: they load or wait for the matcher's own
#                           tools, which Claude Code defers while the server is still connecting
#   --strict-mcp-config     only the MCP servers in --mcp-config (the matcher)
#   --setting-sources project   ignore user/local settings; the workspace's .claude/settings.json applies
#   --allowed-tools / --disallowed-tools / --permission-mode dontAsk   allow only the matcher, deny the rest
# The prompt goes first: several of these flags take a variable number of values.
FLAGS=(--session-id "$SESSION_ID"
       --mcp-config "$MCP_CONFIG" --strict-mcp-config
       --setting-sources project
       --tools "ToolSearch,WaitForMcpServers"
       --allowed-tools "mcp__matcher__*,ToolSearch,WaitForMcpServers"
       --disallowed-tools "$DENY"
       --permission-mode dontAsk)

# Launched from a VS Code terminal, Claude Code would also connect to the IDE's own MCP server (mcp__ide__*,
# which can run code in a Jupyter kernel); --strict-mcp-config does not cover it.
export CLAUDE_CODE_AUTO_CONNECT_IDE=false
unset CLAUDE_CODE_SSE_PORT
[ -n "$MODEL" ] && FLAGS+=(--model "$MODEL")

cd "$WORKSPACE" || exit 1
if [ -n "$HEADLESS" ]; then
  "$CLAUDE" -p "$START_PROMPT" "${FLAGS[@]}" --output-format stream-json --verbose \
      > "$RUN_DIR/stream.jsonl" 2> "$RUN_DIR/claude_stderr.log"
  EC=$?
else
  "$CLAUDE" "$START_PROMPT" "${FLAGS[@]}"
  EC=$?
fi
cd - > /dev/null || true

echo "Claude Code exited ($EC); archiving and scoring session $SESSION_ID"
"$PY" -I "$HERE/session.py" finish "$SESSION_ID" --exit-code "$EC"
echo "view it: $PY -I $HERE/viewer/serve.py   (then open http://127.0.0.1:8765/#$SESSION_ID)"
