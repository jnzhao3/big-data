"""BDI-Kit DataFrames (match_schema / rank_schema_matches / match_values / rank_value_matches) -> records.

A NaN target means BDI-Kit found no match. Source values come from pandas' default CSV parsing (floats such
as 1031.0, NaN for empty cells); the scorer canonicalizes them.
"""

from __future__ import annotations

import math

import pandas as pd

from ..predictions import sm_record, vm_record


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def from_bdikit_schema_df(df: pd.DataFrame, run_id: str, study: str, output_type: str = "top1") -> list[dict]:
    """Rows: source_attribute, target_attribute, similarity. rank_schema_matches output -> output_type="ranked"."""
    recs: dict[str, dict] = {}
    for r in df.itertuples(index=False):
        rec = recs.setdefault(r.source_attribute, sm_record(run_id, study, r.source_attribute, [],
                                                            output_type=output_type, scores=[], shown="*"))
        if not _missing(r.target_attribute):
            rec["targets"].append(r.target_attribute)
            rec["scores"].append(float(r.similarity))
    for rec in recs.values():
        if output_type == "ranked":  # BDI-Kit returns each column's candidates in score order already
            pairs = sorted(zip(rec["scores"], range(len(rec["targets"])), rec["targets"]), key=lambda x: (-x[0], x[1]))
            rec["targets"], rec["scores"] = [t for _, _, t in pairs], [s for s, _, _ in pairs]
    return list(recs.values())


def from_bdikit_value_df(df: pd.DataFrame, run_id: str, study: str, output_type: str = "top1") -> list[dict]:
    """Rows: source_attribute, target_attribute, source_value, target_value, similarity.
    rank_value_matches output (several rows per source value) -> output_type="ranked"."""
    recs: dict[tuple, dict] = {}
    for r in df.itertuples(index=False):
        sv = None if _missing(r.source_value) else r.source_value
        key = (r.source_attribute, r.target_attribute, "<NA>" if sv is None else repr(sv))
        rec = recs.setdefault(key, vm_record(run_id, study, r.source_attribute, r.target_attribute, sv, [],
                                             output_type=output_type, scores=[]))
        if not _missing(r.target_value):
            rec["targets"].append(r.target_value)
            rec["scores"].append(None if _missing(r.similarity) else float(r.similarity))
    for rec in recs.values():
        if rec["scores"] and any(s is None for s in rec["scores"]):
            rec["scores"] = None
    return list(recs.values())
