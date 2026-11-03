"""Value-matching scores against GDC-VM, reported per task category.

Rules (E0 design doc, section 4.3):
- An item is (source attribute, target attribute, canonical source value). It is correct if the canonical
  prediction is in the gold set; an empty gold set means only "no match" is correct.
- categorical: exact string match (GDC permissible values are case-sensitive codes); the lenient variant
  also accepts case and inner-whitespace differences.
- numeric_identity: numeric equality.
- age_conversion: |prediction - gold| <= age_tol_days (default 1 day); the lenient variant accepts a
  relative error up to age_rel_lenient (default 2%, covers 365-day years and 30-day months).
- Report categories separately; never as one headline number.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from . import SCORING_VERSION
from .data import VM_GT_ID, Data
from .normalize import NONE, SM_NONE_TOKENS, canon_value, lenient_value, to_decimal
from .studies import get_study

AGE_TOL_DAYS = 1.0
AGE_REL_LENIENT = 0.02
AGE_RATIOS = {"not_converted_years": 1 / 365.25, "not_converted_months": 1 / 30.4375,
              "months_read_as_years": 12.0, "years_read_as_months": 1 / 12.0}


def score_vm(records, data: Data, study, mode: str = "oracle", age_tol_days: float = AGE_TOL_DAYS,
             age_rel_lenient: float = AGE_REL_LENIENT) -> tuple[pd.DataFrame, dict]:
    """Score the VM records of one study. mode is recorded in the summary: "oracle" when the method was given
    the ground-truth attribute pairs, "end_to_end" when it chose the target attributes itself (items of
    attribute pairs it never mapped then count as not_attempted)."""
    sid = get_study(study).id
    recs = [r for r in records if r.get("kind") == "vm" and get_study(r["study"]).id == sid]
    items, info = data.vm_items(sid)
    df, s = score_vm_core(recs, items, data.permissible, age_tol_days, age_rel_lenient)
    s.update(study=sid, gt=VM_GT_ID, mode=mode, gt_only_keys=len(info["gt_only_keys"]),
             table_only_keys=len(info["table_only_keys"]), key_collisions=len(info["key_collisions"]))
    return df, s


def classify_age_error(pred: Decimal, gold: Decimal) -> str:
    if gold == 0:
        return "wrong_number"
    r = float(pred / gold)
    for name, target in AGE_RATIOS.items():
        if abs(r / target - 1) < 0.02:
            return name
    return "wrong_number"


def _pred_value(r, permissible_values):
    """First predicted target as a canonical string, or NONE. "None"-like strings are values only where the
    target attribute permits them (8 GDC attributes have a literal "None")."""
    ts = [t for t in (canon_value(t) for t in r.get("targets", [])) if t != ""]
    if not ts:
        return NONE
    p = ts[0]
    if p.casefold() in SM_NONE_TOKENS and p not in permissible_values:
        return NONE
    return p


def score_vm_core(recs, items: pd.DataFrame, permissible, age_tol_days: float = AGE_TOL_DAYS,
                  age_rel_lenient: float = AGE_REL_LENIENT) -> tuple[pd.DataFrame, dict]:
    by_key, dups, conflicts = {}, 0, 0
    for r in recs:
        k = (r["source_attribute"], r["target_attribute"], canon_value(r["source_value"]))
        if k in by_key:
            dups += 1
            conflicts += _pred_value(by_key[k], permissible.get(k[1]) or set()) != _pred_value(
                r, permissible.get(k[1]) or set())
            continue
        by_key[k] = r
    tol = Decimal(str(age_tol_days))
    rows = []
    for it in items.itertuples(index=False):
        r = by_key.get((it.source_attribute, it.target_attribute, it.key))
        perm = permissible.get(it.target_attribute) or set()
        attempted = r is not None
        pred = NONE if r is None else _pred_value(r, perm)
        g = it.gold
        row = dict(item_id=it.item_id, study=getattr(it, "study", ""), category=it.category,
                   source_attribute=it.source_attribute, target_attribute=it.target_attribute, key=it.key,
                   gold="|".join(sorted(g)), pred="NONE" if pred is NONE else pred, attempted=attempted,
                   age_exact=None)
        lenient = False
        if not g:
            correct = pred is NONE
            err = "" if correct else ("not_attempted" if not attempted else "false_match")
        elif pred is NONE:
            correct, err = False, ("not_attempted" if not attempted else "missed")
        elif it.category == "categorical":
            correct = pred in g
            lenient = correct or lenient_value(pred) in {lenient_value(x) for x in g}
            err = "" if correct else ("case_or_space_variant" if lenient else
                                      "not_permissible" if pred not in perm else "wrong_value")
        else:
            p = to_decimal(pred)
            golds = [d for d in (to_decimal(x) for x in g) if d is not None]
            if p is None:
                correct, err = False, "non_numeric"
            elif not golds:
                correct, err = False, "wrong_number"
            elif it.category == "numeric_identity":
                correct = any(abs(p - d) <= Decimal("1e-9") * max(1, abs(d)) for d in golds)
                err = "" if correct else "wrong_number"
            else:  # age_conversion
                diff = min(abs(p - d) for d in golds)
                correct = diff <= tol
                row["age_exact"] = bool(diff <= Decimal("1e-6"))
                lenient = correct or any(d != 0 and abs(float((p - d) / d)) <= age_rel_lenient for d in golds)
                err = "" if correct else ("convention_or_rounding" if lenient else classify_age_error(p, golds[0]))
        row["correct"], row["lenient_correct"], row["error"] = bool(correct), bool(correct or lenient), err
        rows.append(row)
    df = pd.DataFrame(rows)
    item_keys = {(i.source_attribute, i.target_attribute, i.key) for i in items.itertuples(index=False)}
    s = {"scoring_version": SCORING_VERSION, "kind": "vm", "n_items": len(df), "duplicate_predictions": dups,
         "conflicting_duplicates": conflicts, "unmatched_predictions": len(set(by_key) - item_keys),
         "age_tol_days": age_tol_days, "age_rel_lenient": age_rel_lenient}
    if len(df):
        df["task"] = df.source_attribute + "->" + df.target_attribute
        groups = [("all_but_no_legal_target", df[df.category != "no_legal_target"])] + list(df.groupby("category"))
        for cat, sub in groups:
            if not len(sub):
                continue
            s[cat] = {"n_items": len(sub), "n_tasks": int(sub.task.nunique()), "micro": float(sub.correct.mean()),
                      "macro_by_task": float(sub.groupby("task").correct.mean().mean()),
                      "micro_lenient": float(sub.lenient_correct.mean()), "coverage": float(sub.attempted.mean()),
                      "errors": {k: int(v) for k, v in sub.error[sub.error != ""].value_counts().items()}}
        df = df.drop(columns="task")
    return df, s
