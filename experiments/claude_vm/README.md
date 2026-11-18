# Claude Code value matching (GDC-VM)

A harness that measures how well a Claude Code session maps the values of a CPTAC clinical table to the GDC
data model, using only the tools of a dedicated MCP server (`matcher`). The session is locked down: no shell,
no files, no web, no subagents. Every tool call and the full Claude Code transcript are logged, the answers are
written in the repo's shared prediction format and scored with hbench, and a local viewer shows each session
as a timeline.

- **Task:** GDC-VM, one study per session. An item is (source column, GDC attribute, source value); the answer
  is one GDC value or an explicit NONE. All categories are in scope: categorical, numeric identity, age
  conversion (to days), and no legal target.
- **Modes** (`--mode`):
  - `oracle`: the session is given GDC-VM's attribute pairs and only maps values;
  - `end_to_end`: the session picks the GDC attribute(s) for each column itself (`map_attribute`), then maps
    values. Items of pairs it never maps count as not attempted.
- **Scoring:** hbench's VM scorer, per category (strict and lenient, coverage), plus calibration (Brier, ECE,
  confidence buckets) from each answer's stated confidence. Several correct answers for one item are fine:
  hbench accepts any member of the gold set; an item with an empty gold set is correct only as an explicit NONE.

## Setup

Nothing beyond the repo setup: the repo environment `~/venvs/harmonize` (it already has `mcp`, `RapidFuzz` and
`jellyfish`), the benchmark data in `datasets/` (`python -m hbench manifest verify`), and Claude Code, logged in.

`run_eval.sh` finds the `claude` binary on your PATH, or else the one bundled with the VS Code extension
(`~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude`). Set `CLAUDE_BIN` to use
another.

Check the harness:

```bash
~/venvs/harmonize/bin/python -m pytest -q experiments/claude_vm/tests
```

## Run a session

From the repo root:

```bash
experiments/claude_vm/run_eval.sh --study clark --mode oracle
experiments/claude_vm/run_eval.sh --study clark --mode end_to_end
experiments/claude_vm/run_eval.sh --study dou --mode oracle --allow-sql --model opus
experiments/claude_vm/run_eval.sh --study clark --mode oracle --headless     # claude -p, no terminal UI
experiments/claude_vm/run_eval.sh --study clark --mode oracle --search-only  # matching only through searches
experiments/claude_vm/run_eval.sh --study clark --mode oracle --allow-code   # the model can write and run code
```

Studies: `cao clark dou gilette huang krug mcdermott satpathy vasaikar wang`. All 10 studies = 10 sessions.

What happens:

1. `session.py start` creates a session id (a UUID, used for Claude Code's session too), renders the eval
   workspace at `~/hbench_raw/claude_vm/workspace/` (`.claude/settings.json`, `CLAUDE.md` for the mode,
   `.mcp.json`), and records the session in `session_index.json`.
2. Claude Code starts in that workspace with the task text as its opening message, locked down (below). In an interactive session,
   watch it work; when it has finished its summary, leave with `/exit` (or Ctrl+D).
3. After Claude Code exits, `session.py finish` archives the transcript, converts the submissions to
   `predictions.jsonl`, scores them, writes `session.json`, and prints the per-category scores.

**The first interactive session asks you to trust the workspace folder.** Accept it: the folder is the
harness's own workspace (it holds only the generated settings, `CLAUDE.md` and `.mcp.json`), and trust lets
the workspace's permission rules apply. It is asked once, since every session reuses the same folder.

Run one session at a time: sessions share the workspace folder.

## Search-only mode (`--search-only`)

In a normal session the model reads whole vocabularies (`list_permissible`) and does the matching in its head,
so the logs show what it read but not what it compared. With `--search-only`:

- `list_permissible` is not offered, and `describe_study` says so;
- `search_permissible` returns at most `--search-top-k` results per call (default 5);
- `submit_value_matches` accepts an enumerated value only if a `search_permissible` call in this session
  returned it, and gives no "did you mean" hints;
- the task text tells the model to search for each source value.

Every candidate the model saw for each answer is then in the server log. It cannot be combined with
`--allow-sql`. It changes the task (the model never sees a whole vocabulary), so compare its scores with other
search-only sessions; `compare.py` labels the mode and shows tool calls and searches per session.

## Model-written code (`--allow-code`)

With `--allow-code` the model gets three more tools:

| Tool | What it does |
|---|---|
| `save_code(name, code, description)` | Stores a function `run(inputs)`; saving under the same name adds a version (`runs/<id>/code/<name>_v<k>.py`) |
| `run_code(name, inputs, columns, attributes, version, offset, limit)` | Runs it on `inputs`; `columns=[...]` makes `values(column)` available, `attributes=[...]` makes `permissible(attribute)` available (not in search-only mode). A list or dict result is paged under the row cap |
| `list_code()` | The saved functions and their versions |

The model still submits answers itself with `submit_value_matches`. Every saved version (the full code) and every
run (inputs, output, errors) is in the server log and the viewer, so the rules it wrote down can be read exactly.

How the code is contained (`cvm/codebox.py`): a static check allows imports of `math`, `re`, `statistics` and
`unicodedata` only, and rejects names and attributes starting with `_`, string constants containing `__`,
`eval`/`exec`/`open`/`getattr`/`type` and similar, `.format`, and frame/code attributes. The code then runs in a
child process (`python -I -S`, empty environment, empty temporary directory, 10 s wall / 5 s CPU / 768 MB, no file
writes) with whitelisted builtins and module proxies that hide submodules (`statistics.sys` would otherwise reach
`os`). The code never reads files: its data comes only through the server.

This is a restricted interpreter, not an OS sandbox: restricted Python has a history of escapes, and the child
runs as your user. It keeps a cooperative model away from files, the network and the ground truth; it is not
proof against a determined attacker. `sudo apt install bubblewrap` would allow a real sandbox (option 2).

It is a different condition from sessions without code: compare it with other `--allow-code` sessions.
`compare.py` shows a `code` column and the number of `run_code` calls.

## View and score

```bash
~/venvs/harmonize/bin/python -I experiments/claude_vm/viewer/serve.py      # http://127.0.0.1:8765/
```

The viewer has two tabs. **Compare runs** (`#compare`) lists every scored run in one sortable table: conditions
(mode, search-only and its k, code, SQL), model, items, strict and answered-only accuracy for a chosen category,
coverage, Brier/ECE, and tool use (tool calls, calls per item, searches, candidates shown, code runs), with E3's
BDI-Kit runs as optional reference rows. Below it, a shaded overview of every category per run, and an item-level
diff of any two runs (linkable as `#compare?a=<id>&b=<id>`). Click a run to open its timeline.

**Timeline** lists sessions (study, mode, status), and shows a session's summary (duration, tool calls by tool,
rows returned, items answered, explicit NONEs, accuracy by category, calibration) above a timeline that merges
the transcript, the server log and the hook events. Filter by event type or tool, search, click an event for
its JSON, and follow the links between a tool call, its result and its server-log entry. It rebuilds the
timeline on each load, so it also shows a running session (as far as the hooks have archived it).

Compare sessions (and E3's BDI-Kit agent):

```bash
~/venvs/harmonize/bin/python -I experiments/claude_vm/compare.py --study clark --e3
~/venvs/harmonize/bin/python -I experiments/claude_vm/compare.py --items <session A> <session B>
```

Re-score a session (e.g. after a scoring change) without touching its logs:

```bash
~/venvs/harmonize/bin/python -I experiments/claude_vm/session.py rebuild <session_id>
~/venvs/harmonize/bin/python -m hbench score ~/hbench_raw/claude_vm/runs/<session_id>/predictions.jsonl \
    --vm-mode oracle --out-dir /tmp/rescore          # hbench's own CLI works on the same file
```

## Where everything goes

Under `~/hbench_raw/claude_vm/` (`$MATCH_EVAL_LOG_ROOT`, or `$HBENCH_RAW/claude_vm`), outside the repo:

| Path | Contents |
|---|---|
| `session_index.json` | Per session: start/end time, study, mode, SQL on/off, row cap, Claude Code version, model(s) used, repo commit (and whether the worktree was dirty), status, headline scores, paths |
| `server/<id>.jsonl` | Every MCP tool call: timestamp (ms), tool, full arguments, rows and bytes returned, duration, error, full result |
| `transcripts/<id>/` | `main.jsonl` (the Claude Code transcript), `precompact_*.jsonl` snapshots, `subagents/`, `hook_events.jsonl` |
| `submissions/<id>.jsonl` | Every submission and (un)mapping, append-only; the last answer per item counts |
| `runs/<id>/predictions.jsonl` | Shared format (`hbench.pred/1`, `vm_record`); `targets: []` is an explicit NONE; `confidence` and `rationale` are extra fields |
| `runs/<id>/items.vm.csv`, `scores.json` | hbench's per-item table (with confidence) and the summary: per category, calibration, and for end-to-end the scores on mapped pairs only |
| `runs/<id>/session.json` | The complete merged record the viewer reads |
| `runs/<id>/CLAUDE.md`, `mcp.json`, `workspace_settings.json` | The exact task text and configuration of that session |
| `runs/<id>/stream.jsonl` | `--headless` only: Claude Code's stream-json output |
| `hooks_errors.log` | Errors from the hook script (hooks never fail a session) |

## The tools

| Tool | What it returns |
|---|---|
| `describe_study` | Source columns (type, distinct values, missing cells), row count; in oracle mode the attribute pairs |
| `profile_column` | Distinct count, null rate, top 20 values, lengths, numeric range, value shapes, casing |
| `sample_rows` | Seedable random or stratified sample of source rows |
| `list_values` | The values to map, with counts. `""` stands for empty cells and is a value too |
| `describe_attribute`, `list_permissible` | A GDC attribute's description; its permissible values with descriptions (`list_permissible` is absent with `--search-only`) |
| `search_attributes`, `search_permissible` | Fuzzy search over attribute names/descriptions, or over one attribute's permissible values: `exact`, `normalized`, `ratio`, `partial_ratio`, `token_sort`, `token_set`, `jaro_winkler`, `soundex`, `metaphone` |
| `submit_value_matches` | Validates and stores answers (one value or `null` per item, confidence 0-1, optional rationale) |
| `get_progress` | Per pair: submitted, explicit NONE, remaining. Never correctness |
| `map_attribute`, `unmap_attribute` | End-to-end mode only |
| `query` | `--allow-sql` only: one read-only SELECT over `source`, `attributes`, `permissible_values` |

Every tool caps its rows (`--row-cap`, default 50) and pages with `offset`/`limit`.

## The lockdown

Flags (checked against `claude --help` of Claude Code 2.1.327, and the docs at code.claude.com, 2026-11-18):

- `--tools "ToolSearch,WaitForMcpServers"`: the only built-in tools. They load or wait for the matcher's own
  tools: Claude Code defers MCP tools (tool search) and, in an interactive session, sends the first message
  before the server has connected. With `--tools ""` the session had no way to reach the matcher tools;
- `--strict-mcp-config --mcp-config <workspace>/.mcp.json`: the matcher is the only MCP server;
- `--setting-sources project`: your user and local settings (and their hooks, permissions, MCP servers) are
  not loaded; the workspace's `.claude/settings.json` is;
- `--allowed-tools "mcp__matcher__*,ToolSearch,WaitForMcpServers"`, `--disallowed-tools <every other built-in
  tool>`, `--permission-mode dontAsk`;
- `CLAUDE_CODE_AUTO_CONNECT_IDE=false` and `CLAUDE_CODE_SSE_PORT` unset: launched from a VS Code terminal,
  Claude Code otherwise connects to the IDE's MCP server (`mcp__ide__executeCode` runs code in a Jupyter
  kernel), which `--strict-mcp-config` does not cover.

The task text is the session's opening message (also saved as `runs/<id>/start_prompt.md`). Claude Code
2.1.327 does not load the workspace `CLAUDE.md` under `--setting-sources project`, so `CLAUDE.md` is only a
second copy.

The workspace settings repeat it: `permissions.deny` lists every built-in tool name from the tools reference,
`permissions.allow` is only `mcp__matcher__*`, `ToolSearch` and `WaitForMcpServers`, `defaultMode` is `dontAsk`, `disableClaudeAiConnectors` and
`enableArtifact: false` keep claude.ai connectors and the Artifact tool out, and `showThinkingSummaries: true`
records thinking summaries. The hooks (`Stop`, `SubagentStop`, `PreCompact`, `SessionEnd`) are registered there.
`cleanupPeriodDays` is not set.

The workspace sits outside the repo on purpose: inside it, Claude Code would put the repo's git status (with
commit messages naming the datasets) and parent `CLAUDE.md` files into the model's context.

Ground truth: the server never builds items from the ground truth. The values to map come from the source
table; in oracle mode the attribute pairs are read from the ground-truth file's two attribute columns only
(they are given by design). The SQL database holds the source table and the GDC schema only. Only the scorer
reads the ground truth, after the session.

[verify_lockdown.md](verify_lockdown.md) has prompts to confirm it in a session.

## Known gaps

- **What the model already knows.** GDC's vocabulary and GDC-VM are public; the lockdown cannot stop answers
  from training knowledge. There is no renamed-target control yet.
- **Thinking summaries** are recorded only in interactive sessions (the documented behaviour of
  `showThinkingSummaries`). Headless sessions record a redacted stub, labelled as such in the viewer.
- **Untrusted workspace.** Until you accept the trust dialog once, Claude Code ignores the workspace's
  `permissions.allow` (the CLI's `--allowed-tools` still allows the matcher tools, so sessions work). In the
  headless probe the hooks ran without trust.
- **Not removable:** `EndConversation` (Claude Code keeps it by design; it only ends the session), and an
  `advisor` tool that Claude Code adds on this account (it consults another model about the conversation; it
  has no access to files or other tools). `ToolSearch` and `WaitForMcpServers` are allowed on purpose (above).
- **Organization-managed settings** apply above everything here and could add hooks or servers; `/status` shows
  whether any are in force.
- **Non-enumerated attributes:** the GDC schema does not say which are numeric, so a non-numeric answer for one
  is accepted with a warning rather than rejected.
- **`--allow-code` uses restricted Python, not an OS sandbox** (see above).
- **One session at a time** (shared workspace folder). The MCP server's own stderr is not logged.
- The transcript holds your Claude Code account and organization ids (`session-owner` entries); the logs stay on
  this machine.

## Files

| File | Role |
|---|---|
| `run_eval.sh` | Launches one locked-down session; then finishes it |
| `session.py` | `start` / `finish` / `rebuild` |
| `compare.py` | The same comparison in the terminal (plus `--comparisons` and CSV export) |
| `cvm/comparison.py` | Cross-run rows and item diffs, shared by `compare.py` and the viewer |
| `server/matcher_server.py` | The MCP server (FastMCP, stdio) and its call log |
| `cvm/study.py` | What the server knows: source table, schema, oracle pairs (no ground truth) |
| `cvm/tools.py` | The tools as plain Python |
| `cvm/store.py`, `cvm/sqlguard.py` | Submissions; the read-only SQL guard |
| `cvm/codebox.py`, `cvm/codebox_runner.py` | Model-written code: validation, storage, restricted execution |
| `cvm/results.py` | Shared-format conversion, hbench scoring, calibration |
| `cvm/timeline.py`, `cvm/paths.py` | Timeline merge and paths (standard library only) |
| `hooks/archive_transcript.py` | Transcript archiving hook |
| `viewer/serve.py`, `viewer/index.html` | The viewer |
| `eval_workspace/` | Templates for the workspace: settings, and `CLAUDE.md` per mode |
| `tests/` | Unit tests (fixture data) and an end-to-end smoke test on clark |
