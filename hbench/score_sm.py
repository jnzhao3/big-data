"""Schema-matching scores against a set of correct targets per source column.

Rules (E0 design doc, section 4.2):
- A prediction is correct if it is any member of the column's gold set (1:n ground truth).
- Primary metrics use annotated columns only. Unannotated columns are reported separately:
  answering NONE there counts as "abstained"; a named target there is a claim to audit, not an error.
  closed_world_correct additionally treats every unannotated column as "should be NONE".
- Target names compare case-insensitively; names outside the 736 GDC columns are hallucinations.
"""

from __future__ import annotations

import pandas as pd

from . import SCORING_VERSION
from .data import SM_GT_VARIANTS, Data, sm_item_id
from .normalize import NONE, canon_target_col
from .studies import get_study

KS = (1, 3, 5, 10, 20)


def score_sm(records, data: Data, study, variant: str = "gdc-sm", ks=KS) -> tuple[pd.DataFrame, dict]:
    """Score the SM records of one study. Returns (per-item table, summary)."""
    sid = get_study(study).id
    recs = [r for r in records if r.get("kind") == "sm" and get_study(r["study"]).id == sid]
    items, summary = score_sm_core(recs, data.source_columns(sid), data.sm_gold(sid, variant), data.target_columns,
                                   study=sid, ks=ks)
    summary["gt"] = SM_GT_VARIANTS[variant]
    return items, summary


def _match_source_columns(recs, source_columns):
    """Map each record to a source column: exact name first, else a unique case/space-insensitive match."""
    exact = set(source_columns)
    loose = {}
    for c in source_columns:
        loose.setdefault(" ".join(c.split()).casefold(), []).append(c)
    by_col, unknown, dups = {}, [], 0
    for r in recs:
        c = r["source_column"]
        if c not in exact:
            cands = loose.get(" ".join(str(c).split()).casefold(), [])
            if len(cands) != 1:
                unknown.append(c)
                continue
            c = cands[0]
        if c in by_col:
            dups += 1
            continue
        by_col[c] = r
    return by_col, unknown, dups


def score_sm_core(recs, source_columns, gold, target_names, study: str = "", ks=KS) -> tuple[pd.DataFrame, dict]:
    valid = {t.casefold() for t in target_names}
    by_col, unknown, dups = _match_source_columns(recs, source_columns)
    rows = []
    for col in source_columns:
        r = by_col.get(col)
        g = gold.get(col)
        targets = [] if r is None else [t for t in (canon_target_col(x) for x in r.get("targets", [])) if t is not NONE]
        top1 = targets[0] if targets else NONE
        status = "no_prediction" if r is None else r.get("status", "ok")
        shown = None if r is None else r.get("shown")
        shown_set = valid if shown == "*" else (None if shown is None else {canon_target_col(x) for x in shown})
        row = dict(item_id=sm_item_id(study, col) if study else f"sm:{col}", study=study, source_column=col,
                   annotated=g is not None, gold="|".join(sorted(g)) if g else "",
                   pred="NONE" if top1 is NONE else top1, n_targets=len(targets), status=status,
                   output_type=None if r is None else r.get("output_type"),
                   hallucinated=top1 is not NONE and top1 not in valid)
        if g is not None:
            hits = [i for i, t in enumerate(targets) if t in g]
            row["correct"] = top1 is not NONE and top1 in g
            row["rank"] = hits[0] + 1 if hits else None
            row["rr"] = 1.0 / row["rank"] if hits else 0.0
            for k in ks:
                row[f"hit@{k}"] = bool(hits) and hits[0] < k
            if row["correct"]:
                row["error"] = ""
            elif status in ("invalid_output", "error", "no_prediction"):
                row["error"] = status
            elif top1 is NONE:
                row["error"] = "missed"
            elif row["hallucinated"]:
                row["error"] = "hallucinated"
            elif shown_set is not None and top1 not in shown_set:
                row["error"] = "not_in_candidates"
            else:
                row["error"] = "wrong_target"
            row["closed_world_correct"] = row["correct"]
            row["candidate_recall"] = None if shown_set is None else bool(g & shown_set)
        else:
            row["correct"] = None
            row["closed_world_correct"] = top1 is NONE
            row["error"] = ("abstained" if top1 is NONE else
                            "hallucinated" if row["hallucinated"] else "claimed_unannotated")
            row["candidate_recall"] = None
        if r is not None and r.get("output_type") == "set":
            ts = set(targets)
            row["set_tp"] = len(ts & g) if g else 0
            row["set_fp"] = len(ts - g) if g else len(ts)
            row["set_fn"] = len(g - ts) if g else 0
        rows.append(row)
    df = pd.DataFrame(rows)
    ann = df[df.annotated]
    n_pairs = sum(len(v) for v in gold.values())
    with_target = df[df.pred != "NONE"]
    s = {"scoring_version": SCORING_VERSION, "kind": "sm", "study": study, "n_columns": len(df),
         "n_annotated": len(ann), "n_unannotated": int((~df.annotated).sum()), "n_gold_pairs": n_pairs,
         "n_predicted_columns": int((df.status != "no_prediction").sum()),
         "top1_correct": int(ann.correct.sum()), "top1_acc": float(ann.correct.mean()) if len(ann) else None,
         "mrr": float(ann.rr.mean()) if len(ann) else None,
         "pair_recall_top1": float(ann.correct.sum() / n_pairs) if n_pairs else None,
         "closed_world_correct": int(df.closed_world_correct.sum()),
         "unannotated_abstained": int((df.error == "abstained").sum()),
         "unannotated_claims": df.loc[df.error == "claimed_unannotated", ["source_column", "pred"]].values.tolist(),
         "hallucinated": int(df.hallucinated.sum()),
         "hallucination_rate": float(with_target.hallucinated.mean()) if len(with_target) else 0.0,
         "candidate_recall": (float(ann.candidate_recall.mean()) if len(ann) and ann.candidate_recall.notna().all()
                              else None),
         "errors": {k: int(v) for k, v in df.error.value_counts().items()},
         "unknown_source_columns": unknown, "duplicate_predictions": dups}
    for k in ks:
        s[f"hit@{k}"] = float(ann[f"hit@{k}"].mean()) if len(ann) else None
    ranked = {c: [t for t in (canon_target_col(x) for x in r.get("targets", [])) if t is not NONE]
              for c, r in by_col.items()}
    pairs = [(c, t) for c, g in gold.items() for t in sorted(g)]
    s["vida_pair_mrr"] = vida_pair_mrr(ranked, pairs) if pairs else None
    s["recall_at_gt"] = recall_at_gt(by_col, pairs)
    if "set_tp" in df:
        tp = df.set_tp.sum()
        fn = n_pairs - tp  # every gold pair not predicted, including columns the method left out entirely
        for name, fp in [("closed", df.set_fp.sum()), ("open", df[df.annotated].set_fp.sum())]:
            p = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            s[f"set_{name}_P"], s[f"set_{name}_R"] = float(p), float(rec)
            s[f"set_{name}_F1"] = float(2 * p * rec / (p + rec)) if p + rec else 0.0
    return df, s


def vida_pair_mrr(ranked: dict[str, list[str]], gold_pairs: list[tuple[str, str]]) -> float:
    """MRR as VIDA's utils/mrr.py computes it: averaged over ground-truth pairs, so a column with two
    correct targets can earn at most 1 + 1/2. Kept only for comparison with published numbers."""
    total = 0.0
    for src, tgt in gold_pairs:
        lst = [t.casefold() for t in ranked.get(src, [])]
        if tgt.casefold() in lst:
            total += 1.0 / (lst.index(tgt.casefold()) + 1)
    return total / len(gold_pairs)


def recall_at_gt(by_col: dict, gold_pairs: list[tuple[str, str]]):
    """Valentine's recall at size of ground truth: rank every predicted (column, target) pair by score,
    keep the top len(ground truth) pairs, and count how many are correct. None when scores are missing."""
    scored = []
    for col, r in by_col.items():
        targets, scores = r.get("targets") or [], r.get("scores")
        if targets and (scores is None or len(scores) != len(targets)):
            return None
        for i, (t, sc) in enumerate(zip(targets, scores or [])):
            t = canon_target_col(t)
            if t is not NONE:
                scored.append((-float(sc), i, col, t))
    if not gold_pairs or not scored:
        return None
    top = {(c, t) for _, _, c, t in sorted(scored)[: len(gold_pairs)]}
    return len(top & {(c, t.casefold()) for c, t in gold_pairs}) / len(gold_pairs)
