"""Loaders for the GDC suite: GDC_new source tables, the GDC target, GDC-SM and GDC-VM ground truth.

Default data root: $HBENCH_DATA, else <repo>/datasets (see datasets/README.md for downloads).
"""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

import pandas as pd

from .normalize import canon_value, to_decimal
from .studies import STUDIES, get_study

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ground-truth variants for schema matching. Record the id with every score.
SM_GT_VARIANTS = {
    "gdc-sm": "gdc-sm-v1.0@zenodo:14963588",
    # The older 197-row GDC_new groundtruth.csv, stripped and deduplicated (what the qwen_local scripts did).
    # Use it only to compare with old numbers; it lacks most 1:n pairs and has the nonexistent target `purity`.
    "gdc-new-raw": "gdc-new-197@vida-data-harmonization-benchmark:3207e37",
    # Same, with `purity` (Vasaikar TumorPurity) mapped to `tumor_purity` as in the gdc-data branch.
    "gdc-new-clean": "gdc-new-197-clean@vida-data-harmonization-benchmark:3207e37",
}
VM_GT_ID = "gdc-vm@zenodo:18557064"

_SM_COLS = ["original_paper_variable_names", "GDC_format_variable_names"]


def default_root() -> Path:
    return Path(os.environ.get("HBENCH_DATA", REPO_ROOT / "datasets"))


class Data:
    """Read-only access to the benchmark files. Methods take any study spelling (see studies.study_id)."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_root()

    # ------------------------------------------------------------------ paths

    def exists(self) -> bool:
        return (self.root / "GDC_new").is_dir() and (self.root / "gdc_vm").is_dir()

    def source_path(self, study) -> Path:
        return self.root / "GDC_new" / get_study(study).gdc_new_dir / "source.csv"

    def old_gt_path(self, study) -> Path:
        return self.root / "GDC_new" / get_study(study).gdc_new_dir / "groundtruth.csv"

    def sm_gt_path(self, study) -> Path:
        return self.root / "gdc_sm" / "extracted" / "data" / "ground-truth" / f"{get_study(study).name}.csv"

    def vm_gt_path(self, study) -> Path:
        return self.root / "gdc_vm" / f"{get_study(study).name}.csv"

    @property
    def target_table_path(self) -> Path:
        return self.root / "gdc_sm" / "extracted" / "data" / "target-tables" / "gdc_unique_columns_concat_values.csv"

    @property
    def schema_path(self) -> Path:
        return self.root / "gdc_vm" / "gdc_schema.json"

    # ------------------------------------------------------------------ tables

    def source(self, study, raw: bool = True) -> pd.DataFrame:
        """raw=True keeps every cell as the original string. raw=False reproduces pandas' default parsing,
        which is what BDI-Kit's MCP server and VIDA's scripts see (e.g. 1031.0 instead of '1031')."""
        p = self.source_path(study)
        return pd.read_csv(p, dtype=str, keep_default_na=False) if raw else pd.read_csv(p, low_memory=False)

    def source_columns(self, study) -> list[str]:
        return list(pd.read_csv(self.source_path(study), nrows=0).columns)

    @cached_property
    def target_columns(self) -> list[str]:
        return list(pd.read_csv(self.target_table_path, nrows=0).columns)

    def target_table(self) -> pd.DataFrame:
        """Each column stacks the permissible values of that attribute; rows are not records, so never
        sample or shuffle them. na_values=[""] keeps the literal "None" values."""
        return pd.read_csv(self.target_table_path, dtype=str, keep_default_na=False, na_values=[""])

    @cached_property
    def schema(self) -> dict:
        return json.loads(self.schema_path.read_text())

    @cached_property
    def permissible(self) -> dict[str, frozenset]:
        """Attribute -> permissible values (empty for numeric or free-text attributes)."""
        return {a: frozenset((e.get("value_data") or {}).keys()) for a, e in self.schema.items()}

    # ------------------------------------------------------------------ schema-matching ground truth

    def sm_pairs(self, study, variant: str = "gdc-sm") -> pd.DataFrame:
        if variant not in SM_GT_VARIANTS:
            raise ValueError(f"unknown SM ground-truth variant {variant!r}; use one of {sorted(SM_GT_VARIANTS)}")
        if variant == "gdc-sm":
            p = self.sm_gt_path(study)
            with open(p, encoding="utf-8") as f:  # the gdc-data branch copy of Vasaikar.csv has no header
                header = 0 if f.readline().startswith(_SM_COLS[0]) else None
            df = pd.read_csv(p, dtype=str, keep_default_na=False, header=header)
        else:
            df = pd.read_csv(self.old_gt_path(study), dtype=str, keep_default_na=False)
        df = df.iloc[:, :2].copy()
        df.columns = ["source", "target"]
        df = df.apply(lambda c: c.str.strip())
        if variant == "gdc-new-clean":
            df["target"] = df["target"].replace({"purity": "tumor_purity"})
        return df.drop_duplicates().reset_index(drop=True)

    def sm_gold(self, study, variant: str = "gdc-sm") -> dict[str, frozenset]:
        """Source column -> set of correct target columns (casefolded). Columns not listed are unannotated."""
        df = self.sm_pairs(study, variant)
        return {s: frozenset(t.casefold() for t in g) for s, g in df.groupby("source")["target"]}

    # ------------------------------------------------------------------ value-matching ground truth

    def vm_rows(self, study) -> pd.DataFrame:
        """GDC-VM rows as strings. An empty target_value means no permissible GDC value; an empty
        source_value means a missing source cell. Exact duplicate rows (2 in Huang) are dropped."""
        df = pd.read_csv(self.vm_gt_path(study), dtype=str, keep_default_na=False)
        return df.drop_duplicates().reset_index(drop=True)

    def vm_items(self, study) -> tuple[pd.DataFrame, dict]:
        """Scorable value-matching items: one per (source attribute, target attribute, canonical source value)
        that appears both in GDC-VM and in the source table.

        Returns (items, info). info counts the keys that cannot be scored and describes each task
        (attribute pair): its category and, for age conversions, the unit inferred from the ground truth.
        """
        sid = get_study(study).id
        rows = self.vm_rows(sid)
        src = self.source(sid)
        perm = self.permissible
        items = []
        info = {"study": sid, "gt": VM_GT_ID, "gt_only_keys": [], "table_only_keys": [], "key_collisions": [],
                "tasks": {}}
        for (sa, ta), grp in rows.groupby(["source_attribute", "target_attribute"], sort=False):
            gold: dict[str, set] = {}
            raw_by_key: dict[str, set] = {}
            for sv, tv in zip(grp.source_value, grp.target_value):
                k = canon_value(sv)
                raw_by_key.setdefault(k, set()).add(sv)
                gold.setdefault(k, set())
                if canon_value(tv) != "":
                    gold[k].add(canon_value(tv))
            for k, raws in raw_by_key.items():
                if len(raws) > 1:  # e.g. Dou Race ' White' and 'White'; their gold sets are merged
                    info["key_collisions"].append((sa, ta, k, sorted(raws)))
            category, unit = _vm_category(ta, gold, perm)
            info["tasks"][(sa, ta)] = {"category": category, "age_unit": unit}
            table_keys = {canon_value(v) for v in src[sa]} if sa in src.columns else set()
            info["gt_only_keys"] += [(sa, ta, k) for k in sorted(set(gold) - table_keys)]
            info["table_only_keys"] += [(sa, ta, k) for k in sorted(table_keys - set(gold))]
            for k in sorted(set(gold) & table_keys):
                items.append(dict(item_id=vm_item_id(sid, sa, ta, k), study=sid, source_attribute=sa,
                                  target_attribute=ta, key=k, gold=frozenset(gold[k]), category=category))
        return pd.DataFrame(items, columns=["item_id", "study", "source_attribute", "target_attribute", "key",
                                            "gold", "category"]), info


def sm_item_id(study, source_column: str) -> str:
    return f"sm:{get_study(study).id}:{source_column}"


def vm_item_id(study, source_attribute: str, target_attribute: str, key: str) -> str:
    return f"vm:{get_study(study).id}:{source_attribute}->{target_attribute}:{key}"


VM_CATEGORIES = ("categorical", "numeric_identity", "age_conversion", "no_legal_target")


def _vm_category(target_attribute, gold, permissible) -> tuple[str, str | None]:
    """categorical: the target attribute has permissible values. Otherwise no_legal_target when every gold
    set is empty, age_conversion when some gold number differs from its source number, numeric_identity else."""
    if permissible.get(target_attribute):
        return "categorical", None
    if all(not g for g in gold.values()):
        return "no_legal_target", None
    ratios = []
    for k, g in gold.items():
        kd = to_decimal(k)
        for x in g:
            xd = to_decimal(x)
            if kd is not None and xd is not None and xd != kd and kd != 0:
                ratios.append(float(xd / kd))
    if not ratios:
        return "numeric_identity", None
    med = sorted(ratios)[len(ratios) // 2]
    unit = "years" if abs(med / 365.25 - 1) < 0.01 else "months" if abs(med / 30.4375 - 1) < 0.01 else "other"
    return "age_conversion", unit


def all_study_ids() -> list[str]:
    return list(STUDIES)
