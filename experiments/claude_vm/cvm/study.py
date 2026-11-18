"""What the MCP server knows about a study: the source table, the GDC schema and, in oracle mode, the attribute
pairs. Nothing else.

Ground-truth isolation: this module never calls Data.vm_items or Data.vm_rows. Those build items from the
ground truth and would reveal which values have answers, each item's category and the age unit. The values to
map come from the source table. In oracle mode the attribute pairs are read from the ground-truth file with
only its two attribute columns (`usecols`); the pairs are given to the model by design.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property

import pandas as pd

from . import REPO_ROOT  # noqa: F401  (puts the repo's hbench on sys.path)
from hbench.data import Data
from hbench.normalize import canon_value, to_decimal
from hbench.studies import get_study

MODES = ("oracle", "end_to_end")


def oracle_pairs(data: Data, study) -> list[tuple[str, str]]:
    """GDC-VM's (source attribute, target attribute) pairs, in file order. Reads only those two columns."""
    df = pd.read_csv(data.vm_gt_path(study), dtype=str, keep_default_na=False,
                     usecols=["source_attribute", "target_attribute"])
    seen, out = set(), []
    for p in zip(df.source_attribute, df.target_attribute):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class StudyView:
    def __init__(self, data: Data, study: str, mode: str):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.data = data
        self.study = get_study(study).id
        self.mode = mode
        self.table = data.source(self.study)  # every cell a string; '' is a missing cell
        self.columns = list(self.table.columns)
        self.pairs = oracle_pairs(data, self.study) if mode == "oracle" else None

    # ------------------------------------------------------------------ source side

    @cached_property
    def _value_counts(self) -> dict[str, Counter]:
        return {c: Counter(canon_value(v) for v in self.table[c]) for c in self.columns}

    def value_counts(self, column: str) -> Counter:
        """Canonical source value -> count. '' is the missing cell (it can be an item to map)."""
        return self._value_counts[column]

    def has_value(self, column: str, value) -> bool:
        return canon_value(value) in self._value_counts[column]

    def inferred_type(self, column: str) -> str:
        vals = [v for v in self._value_counts[column] if v != ""]
        if not vals:
            return "empty"
        if all(to_decimal(v) is not None for v in vals):
            return "numeric"
        return "text" if len(vals) > 50 else "categorical"

    # ------------------------------------------------------------------ target side (GDC schema)

    @property
    def schema(self) -> dict:
        return self.data.schema

    @property
    def permissible(self) -> dict:
        return self.data.permissible

    def has_attribute(self, attribute: str) -> bool:
        return attribute in self.schema

    def permissible_values(self, attribute: str) -> list[tuple[str, str]]:
        """(value, description) pairs in schema order; empty for a non-enumerated attribute."""
        vd = (self.schema.get(attribute) or {}).get("value_data") or {}
        return [(v, d or "") for v, d in vd.items()]
