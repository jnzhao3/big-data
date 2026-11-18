"""Where a session's files live. Standard library only (the hooks and the viewer import it).

Everything goes under the log root, outside the repo:
    $MATCH_EVAL_LOG_ROOT, else $HBENCH_RAW/claude_vm, else ~/hbench_raw/claude_vm
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
from pathlib import Path

SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def log_root() -> Path:
    if os.environ.get("MATCH_EVAL_LOG_ROOT"):
        return Path(os.environ["MATCH_EVAL_LOG_ROOT"]).expanduser()
    if os.environ.get("HBENCH_RAW"):
        return Path(os.environ["HBENCH_RAW"]).expanduser() / "claude_vm"
    return Path.home() / "hbench_raw" / "claude_vm"


class SessionPaths:
    """All paths of one session under a log root."""

    def __init__(self, session_id: str, root: Path | None = None):
        self.session_id = session_id
        self.root = Path(root) if root is not None else log_root()
        self.server_log = self.root / "server" / f"{session_id}.jsonl"
        self.transcripts = self.root / "transcripts" / session_id
        self.submissions = self.root / "submissions" / f"{session_id}.jsonl"
        self.run_dir = self.root / "runs" / session_id
        self.predictions = self.run_dir / "predictions.jsonl"
        self.items = self.run_dir / "items.vm.csv"
        self.scores = self.run_dir / "scores.json"
        self.session_json = self.run_dir / "session.json"
        self.mcp_config = self.run_dir / "mcp.json"
        self.sql_db = self.run_dir / "matcher.db"
        self.hook_events = self.transcripts / "hook_events.jsonl"

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in vars(self).items() if isinstance(v, Path)}


def index_path(root: Path | None = None) -> Path:
    return (Path(root) if root is not None else log_root()) / "session_index.json"


def hooks_error_log(root: Path | None = None) -> Path:
    return (Path(root) if root is not None else log_root()) / "hooks_errors.log"


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_index(root: Path | None = None) -> dict:
    p = index_path(root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def update_index(session_id: str, fields: dict, root: Path | None = None) -> dict:
    """Merge fields into one session's entry of session_index.json, under a file lock."""
    p = index_path(root)
    with _locked(p):
        idx = read_index(root)
        entry = idx.get(session_id, {})
        entry.update(fields)
        idx[session_id] = entry
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n")
        tmp.replace(p)
    return entry
