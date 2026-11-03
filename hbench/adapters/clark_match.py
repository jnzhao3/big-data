"""experiments/qwen_local/clark_match.py CSV -> SM records.

Older CSVs have no candidate list: `in_shortlist` was computed against the old ground truth only, so
candidate recall under another ground truth needs `shown`, which new runs write to predictions.jsonl.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..predictions import sm_record

_STATUS = {"ok": "ok", "hallucinated": "ok", "hallucinated_in_target": "ok", "invalid_json": "invalid_output"}


def from_clark_match_csv(path: str | Path, run_id: str | None = None, study: str = "clark") -> list[dict]:
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    run_id = run_id or path.stem
    out = []
    for r in df.itertuples(index=False):
        pred = r.pred.strip()
        out.append(sm_record(run_id, study, r.source, [] if pred.upper() in ("", "NONE") else [pred],
                             status=_STATUS.get(r.status, "error"), latency_s=float(r.secs) if r.secs else None,
                             legacy_status=r.status))
    return out
