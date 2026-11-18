#!/usr/bin/env python3
"""Claude Code hook (Stop, SubagentStop, PreCompact, SessionEnd): archive the session's transcript.

Reads the hook payload from stdin and copies
    transcript_path       -> <log root>/transcripts/<session_id>/main.jsonl   (overwritten with the latest copy)
    PreCompact            -> also precompact_<UTC timestamp>.jsonl            (pre-compaction history kept)
    agent_transcript_path -> subagents/<agent id or file name>.jsonl         (SubagentStop)
and appends the payload to hook_events.jsonl. Standard library only. Never blocks or fails the session: every
error goes to <log root>/hooks_errors.log and the script exits 0.
"""

import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/claude_vm


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def main() -> None:
    from cvm.paths import SessionPaths, update_index

    raw = sys.stdin.read()
    payload = json.loads(raw) if raw.strip() else {}
    sid = payload.get("session_id") or "unknown-session"
    event = payload.get("hook_event_name", "unknown")
    paths = SessionPaths(sid)
    paths.transcripts.mkdir(parents=True, exist_ok=True)
    copied = []

    tp = payload.get("transcript_path")
    if tp and Path(tp).exists():
        shutil.copyfile(tp, paths.transcripts / "main.jsonl")
        copied.append("main.jsonl")
        if event == "PreCompact":
            name = f"precompact_{_now().replace(':', '').replace('.', '_')}.jsonl"
            shutil.copyfile(tp, paths.transcripts / name)
            copied.append(name)

    atp = payload.get("agent_transcript_path")
    if atp and Path(atp).exists():
        sub = paths.transcripts / "subagents"
        sub.mkdir(exist_ok=True)
        name = f"{payload.get('agent_id') or Path(atp).stem}.jsonl"
        shutil.copyfile(atp, sub / name)
        copied.append(f"subagents/{name}")

    rec = {"ts": _now(), "hook_event_name": event, "copied": copied,
           **{k: v for k, v in payload.items() if k not in ("hook_event_name",)}}
    with open(paths.hook_events, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    if event == "SessionEnd":
        update_index(sid, {"hook_session_end": rec["ts"], "session_end_reason": payload.get("reason")})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            from cvm.paths import hooks_error_log

            p = hooks_error_log()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(f"--- {_now()}\n{traceback.format_exc()}\n")
        except Exception:
            pass
    sys.exit(0)
