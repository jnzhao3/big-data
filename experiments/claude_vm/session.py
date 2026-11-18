"""Session lifecycle for run_eval.sh (run with the repo venv: ~/venvs/harmonize/bin/python -I session.py ...).

    start  --study S --mode oracle|end_to_end [--allow-sql] [--search-only [--search-top-k K]] [--allow-code] [--model M]
           [--row-cap N] [--claude-version V]
           Creates a session id, renders the eval workspace (settings, CLAUDE.md, .mcp.json), records the session
           in session_index.json and prints shell `export` lines for run_eval.sh.
    finish SESSION_ID [--exit-code N] [--force]
           Archives the transcript, converts submissions to predictions.jsonl (shared format), scores them with
           hbench, writes session.json for the viewer, and closes the index entry.
    rebuild SESSION_ID
           finish --force without copying the transcript again (e.g. after a scoring change).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvm import HARNESS_DIR, REPO_ROOT  # noqa: E402
from cvm.paths import SessionPaths, log_root, read_index, update_index  # noqa: E402
from cvm.study import MODES  # noqa: E402

TEMPLATE_DIR = HARNESS_DIR / "eval_workspace"
SERVER = HARNESS_DIR / "server" / "matcher_server.py"
HOOK = HARNESS_DIR / "hooks" / "archive_transcript.py"
# The task text goes into the opening message itself. Claude Code did not load the workspace CLAUDE.md in an
# interactive session launched with --setting-sources project (2.1.327), so CLAUDE.md is only a second copy.
START_SUFFIX = ("\n\nBegin now. The matcher tools may take a moment to connect; if they are not listed yet, "
                "load or wait for them with ToolSearch / WaitForMcpServers.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


VOCAB_HOW = {
    False: "`list_permissible` lists them; `search_permissible` finds the closest ones",
    True: ("in this session they are not listed: find them with `search_permissible`, which returns at most "
           "{k} results per call. Search for each source value you map, and only submit a value a search returned"),
}


CODE_NOTE = ("- You can write and reuse small Python functions: `save_code` stores a function `run(inputs)`, "
             "`run_code` runs it (it can read a column's values or an attribute's permissible values when you name "
             "them), `list_code` lists them. Code has no file or network access and may import only math, re, "
             "statistics and unicodedata. Submitting is still done with `submit_value_matches`.\n")


def render_workspace(ws: Path, paths: SessionPaths, study: str, mode: str, env: dict) -> dict:
    """Write the eval workspace for one session. The workspace sits outside the repo, so the session gets no git
    context and no CLAUDE.md from the repo's parent directories."""
    (ws / ".claude").mkdir(parents=True, exist_ok=True)
    settings = (TEMPLATE_DIR / ".claude" / "settings.json").read_text()
    settings = settings.replace("{{HOOK_COMMAND}}", f"python3 {shlex.quote(str(HOOK))}")
    json.loads(settings)  # must stay valid JSON
    (ws / ".claude" / "settings.json").write_text(settings)
    search_only = env.get("MATCH_EVAL_SEARCH_ONLY") == "true"
    vocab = VOCAB_HOW[search_only].format(k=env.get("MATCH_EVAL_SEARCH_TOP_K"))
    code_note = CODE_NOTE if env.get("MATCH_EVAL_ALLOW_CODE") == "true" else ""
    task = (TEMPLATE_DIR / f"CLAUDE.{mode}.md").read_text().replace("{{STUDY}}", study) \
        .replace("{{VOCAB_HOW}}", vocab).replace("{{CODE_NOTE}}", code_note)
    (ws / "CLAUDE.md").write_text(task)
    mcp = {"mcpServers": {"matcher": {"type": "stdio", "command": sys.executable,
                                      "args": ["-I", str(SERVER)], "env": env}}}
    (ws / ".mcp.json").write_text(json.dumps(mcp, indent=1) + "\n")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    for name, src in (("workspace_settings.json", ws / ".claude" / "settings.json"),
                      ("CLAUDE.md", ws / "CLAUDE.md"), ("mcp.json", ws / ".mcp.json")):
        shutil.copyfile(src, paths.run_dir / name)
    return mcp


def cmd_start(a) -> int:
    from hbench.data import Data
    from hbench.studies import get_study

    study = get_study(a.study).id
    if a.search_only and a.allow_sql:
        print("--search-only cannot be combined with --allow-sql (SQL exposes permissible_values)", file=sys.stderr)
        return 2
    sid = str(uuid.uuid4())
    root = log_root()
    paths = SessionPaths(sid, root)
    for d in (paths.server_log.parent, paths.transcripts, paths.submissions.parent, paths.run_dir):
        d.mkdir(parents=True, exist_ok=True)
    env = {"MATCH_EVAL_SESSION_ID": sid, "MATCH_EVAL_STUDY": study, "MATCH_EVAL_MODE": a.mode,
           "ALLOW_SQL": "true" if a.allow_sql else "false", "MATCH_EVAL_ROW_CAP": str(a.row_cap),
           "MATCH_EVAL_SEARCH_ONLY": "true" if a.search_only else "false",
           "MATCH_EVAL_SEARCH_TOP_K": str(a.search_top_k),
           "MATCH_EVAL_ALLOW_CODE": "true" if a.allow_code else "false",
           "MATCH_EVAL_LOG_ROOT": str(root), "HBENCH_DATA": str(Data().root)}
    ws = root / "workspace"
    render_workspace(ws, paths, study, a.mode, env)
    if a.allow_sql:
        from cvm.sqlguard import build_db
        from cvm.study import StudyView
        build_db(paths.sql_db, StudyView(Data(), study, a.mode))
    commit, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    update_index(sid, {"session_id": sid, "start_time": _now(), "end_time": None, "status": "running",
                       "study": study, "mode": a.mode, "allow_sql": a.allow_sql, "row_cap": a.row_cap,
                       "search_only": a.search_only, "search_top_k": a.search_top_k if a.search_only else None,
                       "allow_code": a.allow_code,
                       "model_requested": a.model or None, "claude_version": a.claude_version or None,
                       "headless": a.headless, "repo_commit": commit, "repo_dirty": dirty, "workspace": str(ws),
                       "paths": paths.as_dict()}, root)
    start_prompt = (ws / "CLAUDE.md").read_text().strip() + START_SUFFIX
    (paths.run_dir / "start_prompt.md").write_text(start_prompt + "\n")
    exports = {"SESSION_ID": sid, "WORKSPACE": str(ws), "MCP_CONFIG": str(ws / ".mcp.json"), "LOG_ROOT": str(root),
               "RUN_DIR": str(paths.run_dir), "START_PROMPT": start_prompt, **env}
    for k, v in exports.items():
        print(f"export {k}={shlex.quote(v)}")
    return 0


def _claude_projects_dir() -> Path:
    base = Path(os.environ["CLAUDE_CONFIG_DIR"]).expanduser() if os.environ.get("CLAUDE_CONFIG_DIR") else \
        Path.home() / ".claude"
    return base / "projects"


def archive_transcript(paths: SessionPaths) -> list[str]:
    """Copy the session's transcript (and subagent transcripts) from Claude Code's project directory. The hooks
    already copy them during the session; this is the final, complete copy."""
    copied = []
    for t in _claude_projects_dir().glob(f"*/{paths.session_id}.jsonl"):
        paths.transcripts.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(t, paths.transcripts / "main.jsonl")
        copied.append(str(t))
        sub = t.parent / paths.session_id / "subagents"
        if sub.is_dir():
            (paths.transcripts / "subagents").mkdir(exist_ok=True)
            for f in sub.glob("*.jsonl"):
                shutil.copyfile(f, paths.transcripts / "subagents" / f.name)
                copied.append(str(f))
    return copied


def _models_used(paths: SessionPaths) -> list[str]:
    models = []
    p = paths.transcripts / "main.jsonl"
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = (e.get("message") or {}).get("model") if isinstance(e.get("message"), dict) else None
            if e.get("type") == "assistant" and m and m not in models:
                models.append(m)
    return models


def cmd_finish(a, copy_transcript: bool = True, force: bool = False) -> int:
    from cvm import results, timeline

    root = log_root()
    entry = read_index(root).get(a.session_id)
    if entry is None:
        print(f"unknown session {a.session_id} in {root}/session_index.json", file=sys.stderr)
        return 2
    paths = SessionPaths(a.session_id, root)
    out = {"end_time": entry.get("end_time") or _now()}
    if a.exit_code is not None:
        out["claude_exit_code"] = a.exit_code
    if copy_transcript:
        out["transcript_sources"] = archive_transcript(paths)
    out["models_used"] = _models_used(paths)
    problems = []
    try:
        out["predictions"] = results.write_session_predictions(paths, entry["study"], entry["mode"], force=force)
        scores = results.score_session(paths, entry["study"], entry["mode"], force=force)
        out["headline"] = results.headline(scores)
        out["calibration_all"] = {k: scores["calibration"]["all_attempted"].get(k) for k in ("n", "brier", "ece")}
    except Exception as e:  # keep going: the timeline is still useful
        problems.append(f"{type(e).__name__}: {e}")
    out["status"] = "finished" if not problems else "finished_with_errors"
    out["problems"] = problems
    entry = update_index(a.session_id, out, root)
    timeline.write(paths, entry)
    print(json.dumps({k: entry.get(k) for k in ("session_id", "study", "mode", "status", "problems", "headline",
                                                "calibration_all", "models_used")}, indent=1, default=str))
    print(f"run folder: {paths.run_dir}")
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--study", required=True)
    s.add_argument("--mode", choices=MODES, required=True)
    s.add_argument("--allow-sql", action="store_true")
    s.add_argument("--search-only", action="store_true",
                   help="no list_permissible: permissible values only through search_permissible")
    s.add_argument("--search-top-k", type=int, default=5, help="results per search in search-only mode")
    s.add_argument("--allow-code", action="store_true",
                   help="expose save_code / run_code / list_code (restricted Python)")
    s.add_argument("--model", default="")
    s.add_argument("--row-cap", type=int, default=50)
    s.add_argument("--claude-version", default="")
    s.add_argument("--headless", action="store_true")
    f = sub.add_parser("finish")
    f.add_argument("session_id")
    f.add_argument("--exit-code", type=int, default=None)
    f.add_argument("--force", action="store_true", help="rebuild derived files (predictions, scores)")
    r = sub.add_parser("rebuild")
    r.add_argument("session_id")
    a = ap.parse_args()
    if a.cmd == "start":
        return cmd_start(a)
    if a.cmd == "finish":
        return cmd_finish(a, force=a.force)
    a.exit_code = None
    return cmd_finish(a, copy_transcript=False, force=True)


if __name__ == "__main__":
    sys.exit(main())
