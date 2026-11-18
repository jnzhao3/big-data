"""After a session: submissions -> shared-format predictions -> hbench scores + calibration.

This is the only module of the harness that reads the ground truth (through hbench.score_vm).
"""

from __future__ import annotations

import json

import pandas as pd

from .paths import SessionPaths
from .store import SubmissionStore, item_key
from .study import oracle_pairs
from hbench import SCORING_VERSION
from hbench.data import Data
from hbench.normalize import canon_value
from hbench.predictions import read_predictions, validate, vm_record, write_predictions
from hbench.score_vm import score_vm
from hbench.studies import get_study

BUCKETS = [i / 10 for i in range(11)]


def final_pairs(store: SubmissionStore, data: Data, study: str, mode: str) -> list[tuple[str, str]]:
    return oracle_pairs(data, study) if mode == "oracle" else store.mapped_pairs()


def to_records(store: SubmissionStore, session_id: str, study: str, pairs) -> list[dict]:
    """One vm_record per submitted item of a final pair. target_value None (explicit NONE) -> targets [].
    Items never submitted get no record, so hbench counts them not_attempted."""
    pairs = set(pairs)
    recs = []
    for (sa, ta, sv), s in store.matches().items():
        if (sa, ta) not in pairs:
            continue
        tv = s.get("target_value")
        targets = [] if tv is None else [tv]
        recs.append(vm_record(session_id, get_study(study).id, sa, ta, sv, targets, output_type="top1",
                              scores=None if tv is None else [s["confidence"]], status="ok",
                              confidence=s["confidence"], rationale=s.get("rationale"), submitted_at=s.get("ts")))
    return recs


def write_session_predictions(paths: SessionPaths, study: str, mode: str, data: Data | None = None,
                              force: bool = False) -> dict:
    data = data or Data()
    store = SubmissionStore(paths.submissions)
    pairs = final_pairs(store, data, study, mode)
    recs = to_records(store, paths.session_id, study, pairs)
    errors, warnings = validate(recs)
    if errors:
        raise ValueError(f"predictions failed hbench validation: {errors[:5]}")
    if paths.predictions.exists():
        if not force:
            raise FileExistsError(f"{paths.predictions} exists (use --force to rebuild derived files)")
        paths.predictions.unlink()
    write_predictions(paths.predictions, recs)
    return {"n_records": len(recs), "n_explicit_none": sum(1 for r in recs if not r["targets"]),
            "pairs": [list(p) for p in pairs], "validation_warnings": warnings}


def _calibration(df: pd.DataFrame) -> dict:
    """Confidence buckets vs observed accuracy, and the Brier score, over attempted items with a confidence."""
    d = df[df.attempted & df.confidence.notna()]
    if d.empty:
        return {"n": 0}
    y = d.correct.astype(float)
    c = d.confidence.astype(float)
    buckets = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        m = (c >= lo) & ((c < hi) if hi < 1 else (c <= hi))
        if m.any():
            buckets.append({"bucket": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_confidence": round(float(c[m].mean()), 4),
                            "accuracy": round(float(y[m].mean()), 4)})
    ece = sum(b["n"] * abs(b["mean_confidence"] - b["accuracy"]) for b in buckets) / len(d)
    return {"n": int(len(d)), "brier": round(float(((c - y) ** 2).mean()), 4), "ece": round(ece, 4),
            "mean_confidence": round(float(c.mean()), 4), "accuracy": round(float(y.mean()), 4), "buckets": buckets}


def _category_summary(df: pd.DataFrame) -> dict:
    """Per category: hbench's strict and lenient accuracy, plus the same counting only answered items.

    hbench counts an item whose gold set is empty as correct when it was not submitted at all (no prediction
    reads as "no match"). This harness keeps an explicit NONE apart from a skipped item, so `micro_answered`
    counts an item correct only if the session submitted it."""
    out = {}
    for cat, sub in [("all_but_no_legal_target", df[df.category != "no_legal_target"])] + list(df.groupby("category")):
        if len(sub):
            att = sub.attempted.astype(bool)
            out[cat] = {"n_items": int(len(sub)), "micro": round(float(sub.correct.mean()), 4),
                        "micro_answered": round(float((sub.correct.astype(bool) & att).mean()), 4),
                        "micro_lenient": round(float(sub.lenient_correct.mean()), 4),
                        "micro_lenient_answered": round(float((sub.lenient_correct.astype(bool) & att).mean()), 4),
                        "coverage": round(float(att.mean()), 4)}
    return out


def score_session(paths: SessionPaths, study: str, mode: str, data: Data | None = None, force: bool = False) -> dict:
    data = data or Data()
    recs = read_predictions(paths.predictions)
    items, summary = score_vm(recs, data, study, mode=mode)
    conf = {item_key(r["source_attribute"], r["target_attribute"], r["source_value"]): r.get("confidence")
            for r in recs}
    items["confidence"] = [conf.get((a, t, k)) for a, t, k in zip(items.source_attribute, items.target_attribute,
                                                                  items.key)]
    item_keys = set(zip(items.source_attribute, items.target_attribute, items.key))
    unscored = [r for r in recs if (r["source_attribute"], r["target_attribute"], canon_value(r["source_value"]))
                not in item_keys]

    calibration = {"all_attempted": _calibration(items)}
    for cat, sub in items.groupby("category"):
        calibration[cat] = _calibration(sub)

    out = {"scoring_version": SCORING_VERSION, "session_id": paths.session_id, "study": get_study(study).id,
           "mode": mode, "hbench_summary": summary, "by_category": _category_summary(items),
           "calibration": calibration,
           "n_submitted_records": len(recs), "n_unscored_submissions": len(unscored),
           "unscored_note": "submissions for items GDC-VM does not score (no GT row, or a pair outside the GT)"}
    if mode == "end_to_end":
        mapped = set(SubmissionStore(paths.submissions).mapped_pairs())
        on_mapped = items[[(a, t) in mapped for a, t in zip(items.source_attribute, items.target_attribute)]]
        gt_pairs = set(zip(items.source_attribute, items.target_attribute))
        out["mapped_pairs"] = {"n_mapped": len(mapped), "n_mapped_in_gt": len(mapped & gt_pairs),
                               "n_gt_pairs": len(gt_pairs)}
        out["restricted_to_mapped_pairs"] = _category_summary(on_mapped) if len(on_mapped) else {}

    for p in (paths.items, paths.scores):
        if p.exists() and not force:
            raise FileExistsError(f"{p} exists (use --force to rebuild derived files)")
    items.to_csv(paths.items, index=False)
    paths.scores.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


def headline(scores: dict) -> dict:
    """Per-category numbers for the console and the index (never one number across categories)."""
    return {k: {f: v.get(f) for f in ("n_items", "micro", "micro_answered", "micro_lenient", "coverage")}
            for k, v in (scores.get("by_category") or {}).items()}
