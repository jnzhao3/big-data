"""Merge a session's Claude Code transcript, server log and hook events into one timeline (session.json).

Standard library only, so the viewer can rebuild session.json without hbench. Transcript JSONL is parsed
defensively: unknown entry types are kept raw, malformed lines are counted and skipped.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from .paths import SessionPaths

MCP_PREFIX = "mcp__matcher__"
# Claude Code bookkeeping entries: kept, but as kind "meta" (hidden by default in the viewer).
META_TYPES = {"api-request", "api-request-blob", "api-request-shape", "file-history-snapshot", "file-history-delta",
              "atis-latch", "session-owner", "last-prompt", "ai-title", "queue-operation", "tools-used", "mode",
              "cost-state", "attachment", "custom-title", "tag"}
PAIR_SLACK_S = 2.0


def _ts(s) -> float | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _read_jsonl(path: Path) -> tuple[list[dict], int]:
    out, bad = [], 0
    if not path.exists():
        return out, bad
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                v = json.loads(line)
                out.append(v if isinstance(v, dict) else {"type": "_non_object", "value": v})
            except json.JSONDecodeError:
                bad += 1
    return out, bad


def _short(x, n=160) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _blocks(content) -> list:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def transcript_events(entries: list[dict], file_label: str) -> list[dict]:
    evs = []
    for e in entries:
        t = e.get("type")
        ts = e.get("timestamp")
        base = {"ts": ts, "source": "transcript", "file": file_label, "uuid": e.get("uuid"),
                "sidechain": bool(e.get("isSidechain"))}
        msg = e.get("message") if isinstance(e.get("message"), dict) else None
        if t in ("user", "assistant") and msg is not None:
            if e.get("isCompactSummary"):
                evs.append({**base, "kind": "compaction", "label": "compaction summary", "raw": e})
                continue
            for i, b in enumerate(_blocks(msg.get("content"))):
                if not isinstance(b, dict):
                    evs.append({**base, "kind": "other", "label": _short(b), "raw": b})
                    continue
                bt = b.get("type")
                ev = {**base, "block": i, "raw": b}
                if t == "user" and bt == "text":
                    ev.update(kind="meta" if e.get("isMeta") else "user_message", label=_short(b.get("text", "")))
                elif t == "user" and bt == "tool_result":
                    ev.update(kind="tool_result", tool_use_id=b.get("tool_use_id"), is_error=bool(b.get("is_error")),
                              label=("ERROR " if b.get("is_error") else "") + _short(b.get("content", "")))
                elif t == "assistant" and bt == "text":
                    ev.update(kind="assistant_text", label=_short(b.get("text", "")))
                elif t == "assistant" and bt in ("thinking", "redacted_thinking"):
                    text = b.get("thinking") or ""
                    ev.update(kind="thinking", thinking_display="summary" if text.strip() else "redacted",
                              label=_short(text) if text.strip() else "(thinking redacted: stub only)")
                elif t == "assistant" and bt == "tool_use":
                    name = b.get("name", "")
                    ev.update(kind="tool_call", tool=name.removeprefix(MCP_PREFIX), tool_name=name,
                              tool_use_id=b.get("id"), label=f"{name}({_short(b.get('input', {}), 120)})")
                else:
                    ev.update(kind="other", label=f"{t}/{bt}")
                evs.append(ev)
        elif t == "system":
            sub = e.get("subtype") or ""
            kind = "compaction" if "compact" in sub else "system"
            evs.append({**base, "kind": kind, "label": sub or _short(e.get("content", "")), "raw": e})
        elif t == "summary":
            evs.append({**base, "kind": "compaction", "label": _short(e.get("summary", "summary")), "raw": e})
        elif t in META_TYPES:
            evs.append({**base, "kind": "meta", "label": t, "raw": e})
        else:
            evs.append({**base, "kind": "other", "label": str(t), "raw": e})
    return evs


def _load_transcripts(tdir: Path) -> tuple[list[dict], dict]:
    """main.jsonl plus pre-compaction snapshots and subagent transcripts, de-duplicated by entry uuid."""
    info = {"files": [], "malformed_lines": 0}
    seen, evs = set(), []
    files = []
    if tdir.exists():
        files = sorted(tdir.glob("precompact_*.jsonl")) + ([tdir / "main.jsonl"] if (tdir / "main.jsonl").exists()
                                                           else []) + sorted((tdir / "subagents").glob("*.jsonl"))
    for f in files:
        entries, bad = _read_jsonl(f)
        info["files"].append(str(f))
        info["malformed_lines"] += bad
        fresh = []
        for e in entries:
            key = e.get("uuid") or json.dumps(e, sort_keys=True, default=str)[:4000]
            if key in seen:
                continue
            seen.add(key)
            fresh.append(e)
        evs += transcript_events(fresh, f.name)
    return evs, info


def _server_events(path: Path) -> list[dict]:
    entries, _ = _read_jsonl(path)
    out = []
    for e in entries:
        if e.get("event") == "tool_call":
            lab = f"{e.get('tool')} -> {e.get('result_rows')} rows, {e.get('duration_ms')} ms" + (
                f", ERROR {e.get('error')}" if e.get("error") else "")
            out.append({"ts": e.get("ts"), "source": "server", "kind": "server_log", "tool": e.get("tool"),
                        "label": lab, "error": e.get("error"), "raw": e})
        else:
            out.append({"ts": e.get("ts"), "source": "server", "kind": "server_log", "tool": None,
                        "label": str(e.get("event")), "raw": e})
    return out


def _hook_events(path: Path) -> list[dict]:
    entries, _ = _read_jsonl(path)
    return [{"ts": e.get("ts"), "source": "hook", "kind": "hook_event", "label": e.get("hook_event_name", "hook"),
             "raw": e} for e in entries]


def _order(evs: list[dict]) -> list[dict]:
    """Sort by time; an event without a timestamp keeps its place after the event before it."""
    last = None
    for i, e in enumerate(evs):
        t = _ts(e.get("ts"))
        if t is None:
            t = last if last is not None else 0.0
        last = t
        e["_t"], e["_i"] = t, i
    evs.sort(key=lambda e: (e["_t"], e["_i"]))
    for n, e in enumerate(evs):
        e["id"] = f"e{n}"
    return evs


def _pair(evs: list[dict]) -> None:
    calls = [e for e in evs if e["kind"] == "tool_call"]
    results = {e.get("tool_use_id"): e for e in evs if e["kind"] == "tool_result"}
    for c in calls:
        r = results.get(c.get("tool_use_id"))
        if r is not None:
            c["result_ref"], r["call_ref"] = r["id"], c["id"]
    server = [e for e in evs if e["source"] == "server" and e.get("tool")]
    used = set()
    for c in calls:
        if not c.get("tool_name", "").startswith(MCP_PREFIX):
            continue
        lo = c["_t"] - PAIR_SLACK_S
        r = next((e for e in evs if e["id"] == c.get("result_ref")), None)
        hi = (r["_t"] if r is not None else c["_t"]) + PAIR_SLACK_S
        for s in server:
            if s["id"] not in used and s["tool"] == c["tool"] and lo <= s["_t"] <= hi:
                used.add(s["id"])
                c["server_ref"], s["call_ref"] = s["id"], c["id"]
                if r is not None:
                    r["server_ref"] = s["id"]
                break


def _submission_counts(path: Path) -> dict:
    entries, _ = _read_jsonl(path)
    latest = {}
    for e in entries:
        if e.get("event") == "match":
            latest[(e.get("source_attribute"), e.get("target_attribute"), e.get("source_value"))] = e
    n_none = sum(1 for e in latest.values() if e.get("target_value") is None)
    return {"submission_events": sum(1 for e in entries if e.get("event") == "match"), "items_answered": len(latest),
            "explicit_none": n_none, "values": len(latest) - n_none}


def build(paths: SessionPaths, index_entry: dict | None = None) -> dict:
    t_evs, tinfo = _load_transcripts(paths.transcripts)
    evs = _order(t_evs + _server_events(paths.server_log) + _hook_events(paths.hook_events))
    _pair(evs)
    times = [e["_t"] for e in evs if e["_t"]]
    server_calls = [e["raw"] for e in evs if e["source"] == "server" and e.get("tool")]
    by_tool = Counter(c.get("tool") for c in server_calls)
    scores = None
    if paths.scores.exists():
        try:
            scores = json.loads(paths.scores.read_text())
        except json.JSONDecodeError:
            scores = None
    summary = {
        "start": min(times) if times else None, "end": max(times) if times else None,
        "duration_s": round(max(times) - min(times), 1) if times else None,
        "tool_calls_by_tool": dict(by_tool), "tool_calls": sum(by_tool.values()),
        "tool_errors": sum(1 for c in server_calls if c.get("error")),
        "rows_returned": sum(c.get("result_rows") or 0 for c in server_calls),
        "events_by_kind": dict(Counter(e["kind"] for e in evs)),
        "submissions": _submission_counts(paths.submissions),
        "transcript": tinfo,
    }
    for e in evs:
        e.pop("_t", None)
        e.pop("_i", None)
    return {"session_id": paths.session_id, "index": index_entry or {}, "summary": summary, "scores": scores,
            "events": evs}


def write(paths: SessionPaths, index_entry: dict | None = None) -> Path:
    doc = build(paths, index_entry)
    paths.session_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.session_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, default=str))
    tmp.replace(paths.session_json)
    return paths.session_json
