"""Run-to-run stability (RQ1) from per-item tables of several runs of the same condition.

Input is the long table that score_sm / score_vm produce, stacked over runs:
columns run_id, item_id, pred (canonical; "NONE" for no match) and correct (bool, or None when unscored).
Prediction-level metrics need no ground truth, so they also cover unannotated columns.
"""

from __future__ import annotations

from collections import Counter
from math import comb

import numpy as np
import pandas as pd


def pairwise_disagreement(labels) -> float:
    """Probability that two different runs (drawn without replacement) disagree; 0 when all agree."""
    labels = list(labels)
    n = len(labels)
    if n < 2:
        return 0.0
    counts = np.array(list(Counter(labels).values()))
    return 1.0 - float((counts * (counts - 1)).sum()) / (n * (n - 1))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one of k draws is correct) from n runs with c correct (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased P(all k draws are correct) from n runs with c correct (tau-bench pass^k)."""
    return comb(c, k) / comb(n, k) if k <= n else float("nan")


def fleiss_kappa(table) -> float:
    """Fleiss' kappa for an N subjects x K categories count table (same number of raters per subject).
    NaN when chance agreement is 1 (every rating in one category)."""
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or len(t) == 0:
        return float("nan")
    n = t[0].sum()
    if n < 2 or not (t.sum(axis=1) == n).all():
        raise ValueError("every subject needs the same number (>= 2) of ratings")
    p_j = t.sum(axis=0) / t.sum()
    p_i = ((t ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float("nan") if p_e == 1 else float((p_bar - p_e) / (1 - p_e))


def stability_report(long: pd.DataFrame) -> dict:
    """Prediction flips, Fleiss' kappa, pass@k / pass^k and always/sometimes/never-correct counts.
    Items missing from some runs are dropped (reported as n_items_incomplete)."""
    runs = sorted(long.run_id.unique())
    n_runs = len(runs)
    piv = long.pivot_table(index="item_id", columns="run_id", values="pred", aggfunc="first")
    complete = piv.dropna()
    out = {"n_runs": n_runs, "n_items": len(complete), "n_items_incomplete": int(len(piv) - len(complete))}
    if n_runs < 2 or not len(complete):
        return out
    flips = complete.apply(lambda r: pairwise_disagreement(r.tolist()), axis=1)
    out["pred_flip_rate"] = float(flips.mean())
    out["items_with_any_change"] = int((flips > 0).sum())
    out["raw_agreement"] = float(1 - flips.mean())
    labels = sorted({str(x) for x in complete.values.ravel()})
    index = {lab: j for j, lab in enumerate(labels)}
    counts = np.zeros((len(complete), len(labels)))
    for i, row in enumerate(complete.itertuples(index=False)):
        for x in row:
            counts[i, index[str(x)]] += 1
    out["fleiss_kappa_pred"] = fleiss_kappa(counts)
    scored = long.dropna(subset=["correct"])
    if len(scored):
        pc = scored.pivot_table(index="item_id", columns="run_id", values="correct", aggfunc="first").dropna()
        pc = pc.astype(bool)
        c = pc.sum(axis=1)
        n = pc.shape[1]
        out["n_scored_items"] = len(pc)
        out["accuracy_per_run"] = {k: float(v) for k, v in pc.mean().items()}
        accs = pc.mean().to_numpy()
        out["accuracy_mean"], out["accuracy_sd"] = float(accs.mean()), float(accs.std(ddof=1)) if n > 1 else 0.0
        out["correct_flip_rate"] = float(pc.apply(lambda r: pairwise_disagreement(r.tolist()), axis=1).mean())
        out["always_correct"] = int((c == n).sum())
        out["sometimes_correct"] = int(((c > 0) & (c < n)).sum())
        out["never_correct"] = int((c == 0).sum())
        for k in range(1, n + 1):
            out[f"pass@{k}"] = float(c.map(lambda x: pass_at_k(n, int(x), k)).mean())
            out[f"pass^{k}"] = float(c.map(lambda x: pass_hat_k(n, int(x), k)).mean())
        out["fleiss_kappa_correct"] = fleiss_kappa([[int(x), n - int(x)] for x in c])
    if "margin" in long and long["margin"].notna().any():
        m = long.groupby("item_id")["margin"].min().dropna()
        out["margin_quantiles"] = {q: float(m.quantile(q)) for q in (0.1, 0.25, 0.5)}
        out["flips_by_margin"] = flips_by_margin(long)
    return out


MARGIN_BINS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf"))


def flips_by_margin(long: pd.DataFrame, bins=MARGIN_BINS) -> list[dict]:
    """How often items changed across runs, by decision margin (each item's smallest margin over the runs,
    i.e. the most fragile observation). This is the curve that links E1's noise floor to observed flips:
    with path-induced logit shifts of size eps, items with margin < eps are the ones that can flip at T=0."""
    if "margin" not in long or long["margin"].isna().all():
        return []
    per = long.groupby("item_id").agg(margin=("margin", "min"), n_preds=("pred", "nunique"))
    per = per.dropna(subset=["margin"])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = per[(per.margin >= lo) & (per.margin < hi)]
        out.append({"margin_from": lo, "margin_to": hi, "n_items": len(sub), "n_changed": int((sub.n_preds > 1).sum()),
                    "flip_rate": float((sub.n_preds > 1).mean()) if len(sub) else None})
    return out


def fragile_items(long: pd.DataFrame, eps: float) -> int:
    """Items whose smallest observed margin is below eps (e.g. the path-induced shift measured in E1)."""
    if "margin" not in long:
        return 0
    m = long.groupby("item_id")["margin"].min().dropna()
    return int((m < eps).sum())


def stability_classes(long: pd.DataFrame) -> pd.Series:
    """item_id -> always / sometimes / never (correct across the runs). Used for E6 transitions."""
    pc = long.dropna(subset=["correct"]).pivot_table(index="item_id", columns="run_id", values="correct",
                                                    aggfunc="first").dropna().astype(bool)
    c, n = pc.sum(axis=1), pc.shape[1]
    return c.map(lambda x: "always" if x == n else "never" if x == 0 else "sometimes")


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, groups=None) -> tuple[float, float]:
    """Percentile CI for the mean of per-item values. With groups (e.g. study per item), items are resampled
    within each group so every study keeps its size."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        means = v[idx].mean(axis=1)
    else:
        g = np.asarray(groups)
        parts = [np.flatnonzero(g == x) for x in pd.unique(g)]
        means = np.array([np.concatenate([v[rng.choice(p, size=len(p))] for p in parts]).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def majority_vote(runs: list[list[dict]], run_id: str) -> list[dict]:
    """Combine k runs' top-1 decisions into one virtual run (E6). Each item gets the most common canonical
    top-1 target; ties go to the decision seen first in run order. Scored with the usual scorer."""
    from .normalize import canon_value, canon_target_col
    from .predictions import item_key

    votes: dict[tuple, list] = {}
    first: dict[tuple, dict] = {}
    for recs in runs:
        for r in recs:
            k = item_key(r)
            first.setdefault(k, r)
            t = r["targets"][0] if r.get("targets") else None
            norm = canon_target_col(t) if r["kind"] == "sm" else (canon_value(t) or None)
            votes.setdefault(k, []).append((norm, t))
    out = []
    for k, vs in votes.items():
        counts = Counter(n for n, _ in vs)
        best = max(counts.values())
        winner = next(n for n, _ in vs if counts[n] == best)
        raw = next(t for n, t in vs if n == winner)
        rec = {key: val for key, val in first[k].items() if key not in ("scores", "span_id", "latency_s")}
        rec.update(run_id=run_id, output_type="top1", targets=[] if winner is None else [raw], status="ok",
                   votes={str(n): c for n, c in counts.items()})
        out.append(rec)
    return out
