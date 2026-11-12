"""BDI-Kit chat agent run -> prediction records (E0 design 5 and 6.3).

The agent's final mapping is the MCP server's state after its last tool call: the last state_snapshot's
schema_matching_results (SM) and value_matching_results (VM). The agent's own final text is a claim, kept apart
for E7 (what the agent says vs. what it did), not a prediction.

- SM: one top1 record per source column in the snapshot (a NaN target is "no match"). update_schema_matching
  edits rows in place, so later edits are already in the snapshot.
- VM: match_values appends to value_matching_results on every call, so a key (source attribute, target attribute,
  source value) seen twice keeps its last row. If match_values never succeeded, VM is "not_attempted" and no VM
  records are written (the scorer then counts the items as missing).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..runlog.writer import BlobStore, read_events
from .bdikit import from_bdikit_schema_df, from_bdikit_value_df

SM_COLS = ["source_attribute", "target_attribute", "similarity"]
VM_COLS = ["source_attribute", "target_attribute", "source_value", "target_value", "similarity"]


def last_snapshot(events: list[dict]) -> dict | None:
    snaps = [e for e in events if e["type"] == "state_snapshot"]
    return max(snaps, key=lambda e: (e["ts_ns"], e["seq"])) if snaps else None


def _tool_ok(events: list[dict], tool: str) -> bool:
    return any(e["type"] == "tool_call" and e.get("side") == "server" and e.get("tool") == tool and not e.get("error")
               for e in events)


def _df(rows, cols) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def _sv_key(v):
    return "<NA>" if v is None or (isinstance(v, float) and v != v) else repr(v)


def predictions_from_run(run_dir: str | Path, study: str, run_id: str | None = None) -> tuple[list[dict], dict]:
    """(records, info). info says which parts were attempted, the snapshot used, and row counts."""
    run_dir = Path(run_dir)
    events = read_events(run_dir)
    run_id = run_id or run_dir.name
    blobs = BlobStore(run_dir.parent / "_blobs")
    snap = last_snapshot(events)
    info = {"snapshot_span": None if snap is None else snap["span_id"], "sm": "not_attempted", "vm": "not_attempted",
            "sm_records": 0, "vm_records": 0, "vm_duplicate_rows_dropped": 0}
    records: list[dict] = []
    if snap is None:
        return records, info
    sm_ref, vm_ref = snap.get("schema_matching_results"), snap.get("value_matching_results")
    if sm_ref is not None:
        df = _df(blobs.get(sm_ref), SM_COLS).drop_duplicates("source_attribute", keep="last")
        sm = from_bdikit_schema_df(df, run_id, study, output_type="top1")
        for r in sm:
            r["evidence"] = {"span_id": snap["span_id"], "source": "mcp_state_snapshot"}
        records += sm
        info.update(sm="ok", sm_records=len(sm))
    if vm_ref is not None and _tool_ok(events, "match_values"):
        df = _df(blobs.get(vm_ref), VM_COLS)
        keys = df.apply(lambda r: (r.source_attribute, r.target_attribute, _sv_key(r.source_value)), axis=1)
        dedup = df[~keys.duplicated(keep="last")] if len(df) else df
        info["vm_duplicate_rows_dropped"] = int(len(df) - len(dedup))
        vm = from_bdikit_value_df(dedup, run_id, study, output_type="top1")
        records += vm
        info.update(vm="ok", vm_records=len(vm))
    return records, info
