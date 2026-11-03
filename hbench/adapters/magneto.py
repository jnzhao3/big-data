"""Magneto output -> SM records.

from_magneto_matches keeps the full ranking with scores (use it for new runs). from_magneto_txt parses the
old magneto_ollama.py stdout, which only kept the top-1 target per column.
"""

from __future__ import annotations

import re

from ..predictions import sm_record

_LINE = re.compile(r"^(\S+)\s+-> (\S+)\s+[✓✗] gold_rank=(\S+)\s*$", re.M)


def from_magneto_matches(matches, run_id: str, study: str, source_columns=None) -> list[dict]:
    """matches: Magneto.get_matches() result, {((src_table, src_col), (tgt_table, tgt_col)): score}.
    One ranked record per source column, best score first (ties keep Magneto's order). Columns in
    source_columns without any candidate get an empty ranking."""
    by_col: dict[str, list] = {}
    for i, (((_, s), (_, t)), score) in enumerate(matches.items()):
        by_col.setdefault(s, []).append((-float(score), i, t, float(score)))
    out = []
    for s in (source_columns if source_columns is not None else by_col):
        ranked = sorted(by_col.get(s, []))
        out.append(sm_record(run_id, study, s, [t for _, _, t, _ in ranked], output_type="ranked",
                             scores=[sc for _, _, _, sc in ranked], shown="*"))
    return out


def from_magneto_txt(text: str, run_id: str = "magneto", study: str = "clark") -> list[dict]:
    return [sm_record(run_id, study, s, [t]) for s, t, _ in _LINE.findall(text)]
