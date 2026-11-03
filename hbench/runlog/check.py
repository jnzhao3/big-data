"""Integrity checks for a run directory (hbench log validate)."""

from __future__ import annotations

from pathlib import Path

from .writer import EVENT_TYPES, LOG_SCHEMA, BlobStore, is_blob_ref, read_events


def _blob_refs(x):
    if is_blob_ref(x):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _blob_refs(v)
    elif isinstance(x, list):
        for v in x:
            if isinstance(v, str) and v.startswith("sha256:"):
                yield {"blob": v, "bytes": None}
            else:
                yield from _blob_refs(v)


def check_run(run_dir: str | Path) -> dict:
    """{"problems": [...], "counts": {...}}. No problems means: every process starts with a run_header, sequence
    numbers are contiguous, every blob exists, the driver ended with run_end, and no call left the machine."""
    run_dir = Path(run_dir)
    events = read_events(run_dir)
    blobs = BlobStore(run_dir.parent / "_blobs")
    problems, counts = [], {}
    by_proc: dict[str, list] = {}
    for e in events:
        by_proc.setdefault(e["proc"], []).append(e)
        counts[e["type"]] = counts.get(e["type"], 0) + 1
        if e.get("v") != LOG_SCHEMA or e.get("type") not in EVENT_TYPES:
            problems.append(f"{e.get('span_id')}: unknown schema or type")
        for ref in _blob_refs(e):
            h = ref["blob"].split(":", 1)[1]
            if not (blobs.root / "sha256" / h[:2] / f"{h}.json.gz").exists():
                problems.append(f"{e['span_id']}: missing blob {h[:12]}")
    for proc, evs in by_proc.items():
        evs = sorted(evs, key=lambda e: e["seq"])
        if evs[0]["type"] != "run_header":
            problems.append(f"{proc}: first event is {evs[0]['type']}, not run_header")
        if [e["seq"] for e in evs] != list(range(1, len(evs) + 1)):
            problems.append(f"{proc}: sequence numbers are not contiguous")
    if "driver" in by_proc and max(by_proc["driver"], key=lambda e: e["seq"])["type"] != "run_end":
        problems.append("driver: no run_end (run crashed or is still running)")
    cloud = [e["span_id"] for e in events if e["type"] == "llm_call" and e.get("cloud_call")]
    if cloud:
        problems.append(f"{len(cloud)} llm_call(s) went to a non-local api_base: {cloud[:5]}")
    failed = sum(1 for e in events if e["type"] == "llm_call" and e.get("error"))
    return {"run_dir": str(run_dir), "problems": problems, "counts": counts, "failed_llm_calls": failed,
            "processes": sorted(by_proc)}
