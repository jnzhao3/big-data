"""Join a run's LLM calls with what the server actually did, from the server log between the run's start and
end byte offsets (robust to other processes appending, unlike line numbers).

Per call it recovers: effective sampler parameters, prompt tokens reused and recomputed, whether the prompt
state was restored from the host-RAM prompt cache, truncation, and the HTTP status. Requests in the window that
match none of the run's calls are foreign: another client used the server during the run, which changes cache
paths and timing, so E1-style measurements require zero.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GENERATION_PATHS = ("/api/chat", "/api/generate", "/v1/chat/completions", "/v1/completions")
_RX = {
    "ram": re.compile(r"found better prompt with f_keep = ([\d.]+), f_sim = ([\d.]+)"),
    "select": re.compile(r"selected slot by (LCP similarity|LRU)"),
    "sampler": re.compile(r"\| sampler params:"),
    "task": re.compile(r"\| task (\d+) \| processing task"),
    "new": re.compile(r"\| task (\d+) \| new prompt, .*task\.n_tokens = (\d+)"),
    "cached": re.compile(r"\| task (\d+) \| cached n_tokens = (\d+)"),
    "prompt_eval": re.compile(r"\| task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens"),
    "eval": re.compile(r"\| task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens"),
    "stop": re.compile(r"\| task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d)"),
    "gin": re.compile(r'\[GIN\] (\S+) - (\S+) \|\s*(\d+) \|\s*(\S+) \|\s*\S+ \|\s*(\w+)\s+"([^"]+)"'),
    "runner": re.compile(r'msg="starting llama-server"'),
}
_KV = re.compile(r"(\w+) = (-?[\d.]+)")


def read_slice(path: str, start: int, end: int) -> str | None:
    try:
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(max(0, end - start)).decode("utf-8", "replace")
    except OSError:
        return None


def parse_requests(text: str) -> dict:
    """Server-side request records, in order. Each generation task carries what the server did for it."""
    tasks, other, reloads = [], [], 0
    pending: dict = {}
    current = None
    in_sampler = False
    for line in text.splitlines():
        if in_sampler:
            if line.startswith("\t"):
                pending.setdefault("sampler", {}).update({k: float(v) for k, v in _KV.findall(line)})
                continue
            in_sampler = False
        if _RX["runner"].search(line):
            reloads += 1
            continue
        if m := _RX["ram"].search(line):
            pending["ram_restore"] = {"f_keep": float(m[1]), "f_sim": float(m[2])}
        elif m := _RX["select"].search(line):
            pending["slot_select"] = "lcp" if m[1].startswith("LCP") else "lru"
        elif _RX["sampler"].search(line):
            in_sampler = True
        elif m := _RX["task"].search(line):
            current = {"task": int(m[1]), "ram_restore": None, "slot_select": None, "sampler": {}, **pending,
                       "n_tokens": None, "cached": None, "computed": None, "eval_tokens": None, "truncated": None,
                       "http": None}
            tasks.append(current)
            pending = {}
        elif current is not None and (m := _RX["new"].search(line)) and int(m[1]) == current["task"]:
            current["n_tokens"] = int(m[2])
        elif current is not None and (m := _RX["cached"].search(line)) and int(m[1]) == current["task"]:
            if current["cached"] is None:  # later lines of the same task are batch progress, not reuse
                current["cached"] = int(m[2])
        elif current is not None and (m := _RX["prompt_eval"].search(line)) and int(m[1]) == current["task"]:
            current["computed"], current["prompt_eval_ms"] = int(m[3]), float(m[2])
        elif current is not None and (m := _RX["eval"].search(line)) and int(m[1]) == current["task"]:
            current["eval_tokens"] = int(m[3])
        elif current is not None and (m := _RX["stop"].search(line)) and int(m[1]) == current["task"]:
            current["truncated"] = m[3] == "1"
        elif m := _RX["gin"].search(line):
            http = {"date": m[1], "time": m[2], "status": int(m[3]), "latency": m[4], "method": m[5], "path": m[6]}
            if m[6] in GENERATION_PATHS and current is not None and current["http"] is None:
                current["http"] = http
            else:
                other.append(http)
    for t in tasks:
        if t["cached"] is None and t["computed"] is not None and t["n_tokens"] is not None:
            t["cached"] = t["n_tokens"] - t["computed"]
    return {"tasks": tasks, "other_requests": other, "runner_starts": reloads}


def join_calls(calls: list[dict], tasks: list[dict]) -> dict:
    """Align the run's llm_calls (in time order) with server tasks (in log order) by prompt size."""
    matched, foreign, j = [], [], 0
    for t in tasks:
        if j < len(calls):
            c = calls[j]
            pt = (c.get("engine") or {}).get("prompt_tokens")
            if pt is None or pt == t["n_tokens"]:
                matched.append((c, t))
                j += 1
                continue
        foreign.append(t)
    return {"matched": matched, "foreign": foreign, "unmatched_calls": calls[j:]}


def audit_run(run_dir, server_log: str | None = None, end_offset: int | None = None) -> dict:
    """Audit one run; writes engine_audit.json into the run folder and returns its summary."""
    from ..runlog.writer import read_events
    from .snapshot import SERVER_LOG, log_offset

    run_dir = Path(run_dir)
    events = read_events(run_dir)
    header = next((e for e in events if e["type"] == "run_header" and e["proc"] == "driver"), None)
    sl = ((header or {}).get("engine") or {}).get("server_log") or {}
    path, start = server_log or sl.get("path") or SERVER_LOG, sl.get("offset")
    end = end_offset if end_offset is not None else log_offset(path)
    summary = {"server_log": path, "offset_start": start, "offset_end": end}
    text = read_slice(path, start, end) if start is not None and end is not None and end >= start else None
    if text is None:
        summary["unavailable"] = "server log slice not readable (no start offset, missing log, or log rotated)"
        return summary
    parsed = parse_requests(text)
    calls = [e for e in events if e["type"] == "llm_call" and (e.get("api_base") or "").find("://") >= 0
             and not e.get("cloud_call") and e.get("response_raw") is not None]
    j = join_calls(calls, parsed["tasks"])
    rows, mismatches = [], []
    for c, t in j["matched"]:
        want = (c.get("decoding") or {}).get("temperature")
        temp = t["sampler"].get("temp")
        if want is not None and temp is not None and abs(float(want) - temp) > 1e-6:
            mismatches.append({"span_id": c["span_id"], "requested": want, "effective": temp})
        api_cached = (c.get("engine") or {}).get("cached_tokens")
        rows.append({"span_id": c["span_id"], "task": t["task"], "n_tokens": t["n_tokens"], "cached": t["cached"],
                     "computed": t["computed"], "api_cached": api_cached, "truncated": t["truncated"],
                     "ram_restore": t["ram_restore"], "slot_select": t["slot_select"], "sampler": t["sampler"],
                     "http": t["http"]})
    summary.update({
        "n_generation_tasks": len(parsed["tasks"]), "n_calls": len(calls), "matched": len(j["matched"]),
        "foreign_tasks": len(j["foreign"]), "unmatched_calls": [c["span_id"] for c in j["unmatched_calls"]],
        "truncated": [r["span_id"] for r in rows if r["truncated"]],
        "ram_restores": [r["span_id"] for r in rows if r["ram_restore"]],
        "cache_disagreements": [r["span_id"] for r in rows if r["api_cached"] is not None and r["cached"] is not None
                                and r["api_cached"] != r["cached"]],
        "sampler_mismatches": mismatches, "runner_starts": parsed["runner_starts"],
        "other_requests": len(parsed["other_requests"])})
    out = run_dir / "engine_audit.json"
    if not out.exists():
        out.write_text(json.dumps({"summary": summary, "calls": rows,
                                   "foreign": [{k: t.get(k) for k in ("task", "n_tokens", "cached", "http")}
                                               for t in j["foreign"]]}, indent=1))
    return summary


def audit_problems(summary: dict) -> list[str]:
    """Problems that make a run unfit for engine-noise analysis. A run that could not be audited is one: nothing
    shows that it had the server to itself or ran at the requested settings."""
    if not summary:
        return []
    if "unavailable" in summary:
        return [f"engine audit not possible: {summary['unavailable']}"]
    probs = []
    if summary.get("foreign_tasks"):
        probs.append(f"{summary['foreign_tasks']} request(s) from another client during the run")
    if summary.get("unmatched_calls"):
        probs.append(f"{len(summary['unmatched_calls'])} call(s) not found in the server log")
    if summary.get("truncated"):
        probs.append(f"{len(summary['truncated'])} prompt(s) truncated by the server (context shift)")
    if summary.get("sampler_mismatches"):
        probs.append(f"{len(summary['sampler_mismatches'])} call(s) ran at a temperature other than requested")
    return probs
