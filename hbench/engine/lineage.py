"""Computation lineage: how every position of a call's context was computed, reconstructed from the server log.

At T=0 llama-server's output is a deterministic function of the input and of how the KV of each context position
was computed (E1): in a batch of prompt tokens (B, with the batch size b of that forward pass), one prompt token
alone (S: the last token recomputed when the whole prompt was cached), or one generated token at a time (G).
A position also depends on how everything before it was computed, so the lineage is recursive: a single-token
recompute is inherited by every later call that reuses it. A per-call label such as path_class cannot see that
(E1 probe 3: two calls with the same path class, cached count and reused generated tokens differ by 0.0103).

reconstruct() replays the server's prompt state from the log events that change it (parse_requests' stream):

    runner start         slot and host-RAM prompt cache are empty
    save / evict         the slot's current state is stored in / the oldest entry leaves the RAM cache
    restore              the RAM-cache entry the server picked (matched by its f_keep, f_sim) moves into the slot
    task (n, cached c)   positions [0, c) keep their lineage; [c, n) are computed in chunks of the runner's
                         -b/-ub batch sizes (a one-token chunk is S); the slot then holds prompt + generated - 1

canonical() writes a context as runs, e.g. "B237 G16 B54", under an explicit, versioned equivalence rule:

    v1   B: batch size b >= 54 (bitwise identical however such batches split: E1 P2, 54..1024); adjacent ones merge
         b<k>: a batch of 2..53 tokens, kept apart (untested; E1 Stage 4a)
         S, G: kept apart (both one token at a time, but not shown to be equivalent)
         ?: positions whose computation the log does not show (the run did not start from a runner start)

Same input, engine and fingerprint predict bitwise-identical logprobs; check() tests that across calls, so an
unknown noise source or engine drift shows up as a violation instead of passing silently.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

RULES = {"v1": {"large_batch_min": 54}}
RULE = "v1"
N_BATCH = 1024
RUNNER_NEEDLE = b'msg="starting llama-server"'


def _length(segs) -> int:
    return sum(s[1] for s in segs)


def _take(segs, n: int) -> list:
    """The first n positions of a segment list."""
    out, left = [], n
    for kind, length, b, origin, tainted in segs:
        if left <= 0:
            break
        out.append([kind, min(length, left), b, origin, tainted])
        left -= length
    return out


def _chunks(start: int, end: int, n_batch: int, n_ubatch: int, origin, tainted: bool = False) -> list:
    """Positions [start, end) as the server computes them: llama_decode calls of up to n_batch tokens, each split
    into ubatches of up to n_ubatch tokens (the forward passes)."""
    out, pos = [], start
    while pos < end:
        batch = min(n_batch, end - pos)
        done = 0
        while done < batch:
            size = min(n_ubatch, batch - done)
            out.append(["S" if size == 1 else "B", size, size, origin, tainted])
            done += size
        pos += batch
    return out


def canonical(segs, rule: str = RULE) -> str:
    """Run-length text of a context under the rule, e.g. "B236 S1 G16 B54" or "B1024 b6"."""
    big = RULES[rule]["large_batch_min"]
    runs: list[list] = []  # [kind, length, small batch size or None]
    for kind, length, b, *_ in segs:
        if length <= 0:
            continue
        if kind == "B" and b is not None and b < big:
            runs.append(["b", length, b])  # small batches are never merged
        elif runs and runs[-1][0] == kind and runs[-1][2] is None:
            runs[-1][1] += length
        else:
            runs.append([kind, length, None])
    return " ".join(f"{k}{n}" if b is None else (f"b{b}" if n == b else f"b{b}x{n}") for k, n, b in runs)


def fingerprint(canon: str, rule: str = RULE) -> str:
    return hashlib.sha256(f"{rule}|{canon}".encode()).hexdigest()[:12]


def reconstruct(stream: list[dict], start: str = "unknown", rule: str = RULE) -> list[dict]:
    """One lineage record per task of the stream, in order. start: the prompt state before the first event,
    "empty" or "unknown" (a runner start in the stream makes it empty either way)."""
    slot = [] if start == "empty" else None   # None: not known
    cache: list[tuple[int, list | None]] = []
    n_batch = n_ubatch = N_BATCH
    several, reload, candidates, note = False, False, [], None
    records = []
    for ev in stream:
        kind = ev["kind"]
        if kind == "runner_start":
            slot, cache, several, reload, note = [], [], False, True, None
            n_batch, n_ubatch = ev.get("n_batch") or N_BATCH, ev.get("n_ubatch") or ev.get("n_batch") or N_BATCH
        elif kind == "runners_loaded":
            several = several or ev["count"] > 1   # their slot lines cannot be told apart
        elif kind == "save":
            known = slot is not None and _length(slot) >= ev["length"]
            cache.append((ev["length"], _take(slot, ev["length"]) if known else None))
        elif kind == "save_skipped":  # "prompt is already in the cache, skipping": the save did not happen
            if cache:
                cache.pop()
        elif kind == "evict":
            if cache:
                cache.pop(0)
        elif kind == "candidates":
            candidates = ev["items"]
        elif kind == "restore":
            hits = [i for i, c in enumerate(candidates) if c[2] == ev["f_keep"] and c[3] == ev["f_sim"]]
            note = None
            if not hits:
                slot, note = None, "restore not found in the cache listing"
            else:
                i = hits[0]  # the server keeps the first of equal candidates (assumed; E0 v3 design, risks)
                if len(hits) > 1:
                    note = "ambiguous restore"
                length = candidates[i][0]
                in_sync = [c[0] for c in candidates] == [e[0] for e in cache]
                k = i if in_sync else next((j for j, e in enumerate(cache) if e[0] == length), None)
                if not in_sync:
                    note = note or "cache listing out of sync"
                entry = None if k is None else cache.pop(k)  # the server moves a restored entry into the slot
                slot = None if entry is None or entry[1] is None else [list(s) for s in entry[1]]
                if slot is None:
                    note = note or "restored prompt not known"
                elif note:  # a guessed restore: whatever reuses these positions inherits the doubt
                    slot = [[*seg[:4], True] for seg in slot]
        elif kind == "task":
            t, idx = ev["task"], len(records)
            n, c = t.get("n_tokens"), t.get("cached") or 0
            uncertain = (["several runners"] if several else []) + ([note] if note else [])
            if n is None:
                uncertain.append("no token counts in the server log")
            if slot is None:
                prefix = [["?", c, None, None, True]] if c else []
                if c:
                    uncertain.append("unknown start")
            elif c > _length(slot):
                prefix = [["?", c, None, None, True]]
                uncertain.append("cached beyond the known slot")
            else:
                prefix = _take(slot, c)
                if any(s[0] == "?" for s in prefix):
                    uncertain.append("unknown start")
                elif any(s[4] for s in prefix):
                    uncertain.append("inherits an uncertain lineage")
            tainted = bool(uncertain)  # positions computed on an uncertain prefix are uncertain too
            context = prefix + (_chunks(c, n, n_batch, n_ubatch, idx, tainted) if n is not None else [])
            canon = canonical(context, rule)
            records.append({
                "task": t.get("task"), "index": idx, "segments": context, "canonical": canon,
                "fingerprint": fingerprint(canon, rule), "rule": rule, "uncertain": uncertain,
                "gen_reused": sum(s[1] for s in prefix if s[0] == "G"),
                "inherited_single": sum(s[1] for s in prefix if s[0] == "S"),
                "prefix_origins": list(dict.fromkeys(s[3] for s in prefix if s[3] is not None)),
                "reload_before": reload})
            stop = t.get("stop_n_tokens")
            gen = stop - n if stop is not None and n is not None else max(0, (t.get("eval_tokens") or 0) - 1)
            slot = context + ([["G", gen, 1, idx, tainted]] if gen > 0 else [])
            reload, note = False, None
    return records


def last_offset_before(path: str, offset: int, needle: bytes = RUNNER_NEEDLE, chunk: int = 4 << 20,
                       limit: int = 1 << 30) -> int | None:
    """Byte offset of the start of the last line containing needle before `offset` (None if not found)."""
    try:
        with open(path, "rb") as f:
            end, carry = offset, b""
            while end > 0 and offset - end < limit:
                start = max(0, end - chunk)
                f.seek(start)
                data = f.read(end - start) + carry
                i = data.rfind(needle)
                if i >= 0:
                    pos = start + i
                    while pos > 0:  # back to the start of that line
                        f.seek(max(0, pos - 4096))
                        back = f.read(pos - max(0, pos - 4096))
                        j = back.rfind(b"\n")
                        if j >= 0:
                            return max(0, pos - 4096) + j + 1
                        pos = max(0, pos - 4096)
                    return 0
                carry = data[: len(needle)]
                end = start
    except OSError:
        return None
    return None


def for_window(path: str, start: int, end: int) -> tuple[list[dict] | None, str | None, int]:
    """(lineage records of the tasks in [start, end), why not, number of earlier tasks read). The log is read
    from the last runner start before `start`, so prompt state left by earlier runs is known; without one the
    start state is unknown. A record's index and prefix_origins count tasks from where reading began."""
    from .audit import parse_requests, read_slice

    first = last_offset_before(path, start)
    begin = start if first is None else first
    text = read_slice(path, begin, end)
    if text is None:
        return None, "server log slice not readable", 0
    before = 0 if begin == start else len(parse_requests(read_slice(path, begin, start) or "")["tasks"])
    records = reconstruct(parse_requests(text)["stream"], start="unknown")
    return records[before:], None, before


def labelled(rec: dict, span_at: dict, before: int) -> dict:
    """A record with task indices replaced by the span ids of the run's calls ("pre-run" for tasks before the
    run, "foreign" for other clients' requests)."""
    name = lambda o: o if o is None else span_at.get(o, "pre-run" if o < before else "foreign")  # noqa: E731
    return {**rec, "prefix_origins": [name(o) for o in rec["prefix_origins"]],
            "segments": [[k, n, b, name(o), u] for k, n, b, o, u in rec["segments"]]}


# ------------------------------------------------------------------ checks across calls

def _audit_rows(run_dir: Path, lineage_dir: Path | None = None) -> dict:
    """span_id -> lineage record, from <lineage_dir>/<run>.lineage.json when there is one, else from the run's
    engine_audit.json."""
    p = Path(lineage_dir) / f"{Path(run_dir).name}.lineage.json" if lineage_dir else None
    if p is None or not p.exists():
        p = Path(run_dir) / "engine_audit.json"
    if not p.exists():
        return {}
    return {r["span_id"]: r["lineage"] for r in json.loads(p.read_text()).get("calls", []) if r.get("lineage")}


def _engine_key(header: dict) -> str | None:
    from ..runlog.writer import sha256_json
    from .lock import lock_from_snapshot
    eng = header.get("engine")
    return sha256_json(lock_from_snapshot(eng))[:12] if eng else None


def check(runs, lineage_dir: str | Path | None = None) -> dict:
    """Group the T=0 calls of the given runs by (input_sha256, engine, lineage fingerprint): within a group the
    logprobs must be bitwise identical. Calls with an uncertain lineage are left out. Also lists inputs whose
    calls had different fingerprints but identical logprobs: evidence that the rule could merge those classes."""
    from ..runlog.writer import read_events

    groups: dict[tuple, list] = {}
    coverage = {}
    for run_dir in runs:
        events = read_events(run_dir)
        header = next((e for e in events if e["type"] == "run_header" and e.get("proc") == "driver"), {})
        ekey, lin = _engine_key(header), _audit_rows(run_dir, lineage_dir)
        cov = coverage.setdefault(Path(run_dir).name, {"t0_calls": 0, "checked": 0, "uncertain": 0, "no_lineage": 0})
        for e in events:
            if e["type"] != "llm_call" or (e.get("decoding") or {}).get("temperature") != 0 or not e.get("logprobs_sha256"):
                continue
            rec = lin.get(e["span_id"])
            cov["t0_calls"] += 1
            cov["checked" if rec and not rec.get("uncertain") else "uncertain" if rec else "no_lineage"] += 1
            if rec and not rec.get("uncertain"):
                key = (e.get("input_sha256"), ekey, rec["fingerprint"])
                groups.setdefault(key, []).append({"run": Path(run_dir).name, "span_id": e["span_id"],
                                                   "logprobs_sha256": e["logprobs_sha256"],
                                                   "lineage": rec["canonical"]})
    violations = [{"input_sha256": k[0], "fingerprint": k[2], "calls": v} for k, v in groups.items()
                  if len({c["logprobs_sha256"] for c in v}) > 1]
    by_input: dict[tuple, dict] = {}
    for (inp, ekey, fp), v in groups.items():
        by_input.setdefault((inp, ekey), {})[fp] = v
    merged = []
    for (inp, _), fps in by_input.items():
        if len(fps) > 1:
            hashes = {fp: {c["logprobs_sha256"] for c in v} for fp, v in fps.items()}
            seen = [fp for fp in fps if any(hashes[fp] & hashes[o] for o in fps if o != fp)]
            if seen:
                merged.append({"input_sha256": inp, "lineages": sorted({fps[fp][0]["lineage"] for fp in seen})})
    return {"n_groups": len(groups), "n_calls": sum(len(v) for v in groups.values()), "violations": violations,
            "equal_across_fingerprints": merged, "coverage": coverage}


def rebuild(run_dir: str | Path, out_path: str | Path, server_log: str | None = None) -> dict:
    """Lineage of an existing run's calls (its engine_audit.json predates lineage), written to out_path; the run
    folder is not touched. Returns the summary."""
    from ..runlog.writer import read_events
    from .audit import join_calls, parse_requests, read_slice
    from .snapshot import SERVER_LOG

    run_dir = Path(run_dir)
    audit = json.loads((run_dir / "engine_audit.json").read_text())
    s = audit["summary"]
    path = server_log or s.get("server_log") or SERVER_LOG
    records, why, before = for_window(path, s["offset_start"], s["offset_end"])
    if records is None:
        summary = {"unavailable": why}
        rows = []
    else:
        tasks = parse_requests(read_slice(path, s["offset_start"], s["offset_end"]) or "")["tasks"]
        index = {id(t): i for i, t in enumerate(tasks)}
        events = read_events(run_dir)
        calls = [e for e in events if e["type"] == "llm_call" and e.get("response_raw") is not None
                 and (e.get("api_base") or "").find("://") >= 0 and not e.get("cloud_call")]
        matched = join_calls(calls, tasks)["matched"]
        span_at = {before + index[id(t)]: c["span_id"] for c, t in matched}
        rows = [{"span_id": c["span_id"], "task": t["task"],
                 "lineage": labelled(records[index[id(t)]], span_at, before)} for c, t in matched]
        summary = {"n_calls": len(rows), "uncertain": [r["span_id"] for r in rows if r["lineage"]["uncertain"]]}
    out = Path(out_path)
    with open(out, "x") as f:
        f.write(json.dumps({"run": run_dir.name, "summary": summary, "calls": rows}, indent=1) + "\n")
    return summary


def run_dirs(paths) -> list[Path]:
    """Run folders among paths, each either a run folder or a raw-results root holding run folders."""
    out = []
    for p in map(Path, paths):
        if any(p.glob("events.*.jsonl")):
            out.append(p)
        else:
            out += sorted(d for d in p.iterdir() if d.is_dir() and any(d.glob("events.*.jsonl"))) if p.is_dir() else []
    return out
