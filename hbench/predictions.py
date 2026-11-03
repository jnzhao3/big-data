"""The shared prediction format (predictions.jsonl, one decision per line).

Every method writes, or is adapted to, these records:

  {"schema": "hbench.pred/1", "run_id": ..., "kind": "sm", "study": "clark", "source_column": "Tumor_Site",
   "output_type": "ranked", "targets": ["site_of_resection_or_biopsy", "primary_site"], "scores": [0.81, 0.77],
   "shown": null, "status": "ok"}
  {"schema": "hbench.pred/1", "run_id": ..., "kind": "vm", "study": "clark", "source_attribute": "Race",
   "target_attribute": "race", "source_value": "White", "output_type": "top1", "targets": ["white"], "status": "ok"}

- output_type: top1 | ranked | set. targets == [] means "no match".
- shown: the candidates the method was shown (list), "*" for the whole target schema, or null if unknown.
  Without it, a valid but unshown target cannot be told apart from a plain wrong one.
- status: ok | invalid_output | error. Raw model output belongs in the run log, not here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "hbench.pred/1"
KINDS = ("sm", "vm")
OUTPUT_TYPES = ("top1", "ranked", "set")
STATUSES = ("ok", "invalid_output", "error")
_KEY_FIELDS = {"sm": ("source_column",), "vm": ("source_attribute", "target_attribute", "source_value")}


def sm_record(run_id: str, study: str, source_column: str, targets, output_type: str = "top1", scores=None,
              shown=None, status: str = "ok", **extra) -> dict:
    return dict(schema=SCHEMA, run_id=run_id, kind="sm", study=study, source_column=source_column,
                output_type=output_type, targets=list(targets), scores=None if scores is None else list(scores),
                shown=shown, status=status, **extra)


def vm_record(run_id: str, study: str, source_attribute: str, target_attribute: str, source_value, targets,
              output_type: str = "top1", scores=None, status: str = "ok", **extra) -> dict:
    return dict(schema=SCHEMA, run_id=run_id, kind="vm", study=study, source_attribute=source_attribute,
                target_attribute=target_attribute, source_value=source_value, output_type=output_type,
                targets=list(targets), scores=None if scores is None else list(scores), status=status, **extra)


def jsonable(x):
    """Plain JSON types: NaN and pd.NA become null, numpy scalars become Python scalars."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, float):
        return None if math.isnan(x) else float(x)
    if isinstance(x, int):
        return int(x)
    if hasattr(x, "item"):  # numpy scalar
        return jsonable(x.item())
    try:
        import pandas as pd
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return str(x)


def write_predictions(path: str | Path, records) -> Path:
    """Write records to a new file; refuses to overwrite an existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(jsonable(r), ensure_ascii=False) + "\n")
    return path


def read_predictions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def item_key(r: dict) -> tuple:
    return (r.get("kind"), r.get("study")) + tuple(str(r.get(k)) for k in _KEY_FIELDS.get(r.get("kind"), ()))


def validate(records, target_names=None) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors break scoring; warnings are worth a look (e.g. unknown target names)."""
    errors, warnings = [], []
    valid_targets = None if target_names is None else {t.casefold() for t in target_names}
    seen = {}
    for i, r in enumerate(records, 1):
        where = f"record {i}"
        if r.get("schema") != SCHEMA:
            errors.append(f"{where}: schema is {r.get('schema')!r}, expected {SCHEMA!r}")
        kind = r.get("kind")
        if kind not in KINDS:
            errors.append(f"{where}: kind must be one of {KINDS}")
            continue
        missing = [k for k in ("run_id", "study", "output_type", "targets", "status") + _KEY_FIELDS[kind] if k not in r]
        if missing:
            errors.append(f"{where}: missing {missing}")
            continue
        if r["output_type"] not in OUTPUT_TYPES:
            errors.append(f"{where}: output_type must be one of {OUTPUT_TYPES}")
        if r["status"] not in STATUSES:
            errors.append(f"{where}: status must be one of {STATUSES}")
        if not isinstance(r["targets"], list):
            errors.append(f"{where}: targets must be a list ([] means no match)")
            continue
        if r.get("scores") is not None and len(r["scores"]) != len(r["targets"]):
            errors.append(f"{where}: scores and targets differ in length")
        if r["output_type"] == "top1" and len(r["targets"]) > 1:
            errors.append(f"{where}: top1 output with {len(r['targets'])} targets")
        if kind == "sm" and valid_targets is not None:
            unknown = [t for t in r["targets"] if isinstance(t, str) and t.strip().casefold() not in valid_targets]
            if unknown:
                warnings.append(f"{where}: targets not in the GDC schema (scored as hallucinated): {unknown}")
        k = item_key(r)
        if k in seen and seen[k] != (r["targets"], r["run_id"]):
            warnings.append(f"{where}: duplicate decision for {k[2:]} (the first one is scored)")
        seen.setdefault(k, (r["targets"], r["run_id"]))
    return errors, warnings
