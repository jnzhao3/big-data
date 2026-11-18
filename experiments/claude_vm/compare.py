"""Compare value-matching scores across sessions, and against E3's BDI-Kit agent.

    python -I experiments/claude_vm/compare.py                    # every scored session
    python -I experiments/claude_vm/compare.py --study clark --e3 # plus E3's runs on clark, as reference rows
    python -I experiments/claude_vm/compare.py --items A B        # item-level differences between two sessions
    python -I experiments/claude_vm/compare.py --csv out.csv      # also write the table
    python -I experiments/claude_vm/compare.py --comparisons A B [--trace]   # searches and comparisons

Session ids can be given by their first characters. Scores are per category, never one number across
categories. "strict" is hbench's; "answered" counts an item correct only if the session submitted it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvm import comparison  # noqa: E402
from cvm.comparison import CATS  # noqa: E402
from cvm.paths import SessionPaths, log_root, read_index  # noqa: E402


def _flat(r: dict) -> dict:
    f = comparison.flatten(r)
    f["brier"], f["ece"] = f.pop("cal.brier", None), f.pop("cal.ece", None)
    f["start"] = (f.get("start") or "")[:16].replace("T", " ")
    return f


def session_rows(root: Path, study: str | None) -> list[dict]:
    """Scored sessions, one flat row each (the same data the viewer's Compare tab shows)."""
    return [_flat(r) for r in comparison.run_rows(root, study)]


def e3_rows(study: str | None) -> list[dict]:
    """E3's BDI-Kit agent runs that did value matching (clark, end to end, categorical attributes only)."""
    return [_flat(r) for r in comparison.e3_rows(study)]


def resolve(root: Path, prefix: str) -> str:
    hits = [s for s in read_index(root) if s.startswith(prefix)]
    if len(hits) != 1:
        sys.exit(f"{prefix!r} matches {len(hits)} sessions")
    return hits[0]


def item_diff(root: Path, a: str, b: str) -> None:
    a, b = resolve(root, a), resolve(root, b)
    cols = ["item_id", "category", "source_attribute", "target_attribute", "key", "gold", "pred", "correct",
            "attempted", "confidence"]
    ta, tb = (pd.read_csv(SessionPaths(s, root).items, dtype=str, keep_default_na=False)[cols] for s in (a, b))
    m = ta.merge(tb, on=["item_id", "category", "source_attribute", "target_attribute", "key", "gold"],
                 suffixes=("_a", "_b"))
    ca, cb = m.correct_a == "True", m.correct_b == "True"
    print(f"A = {a}\nB = {b}\n")
    summary = pd.crosstab(m.category, [ca.map({True: "A right", False: "A wrong"}),
                                       cb.map({True: "B right", False: "B wrong"})])
    print(summary.to_string(), "\n")
    diff = m[(m.pred_a != m.pred_b) | (ca != cb)]
    if diff.empty:
        print("The two sessions gave the same answer on every item.")
        return
    show = diff[["category", "source_attribute", "target_attribute", "key", "gold", "pred_a", "correct_a",
                 "pred_b", "correct_b"]]
    with pd.option_context("display.max_rows", None, "display.width", 220, "display.max_colwidth", 40):
        print(f"{len(diff)} items where the answers or their correctness differ:\n")
        print(show.to_string(index=False))


def _server_calls(root: Path, sid: str) -> list[dict]:
    p = SessionPaths(sid, root).server_log
    out = []
    for line in (p.read_text().splitlines() if p.exists() else []):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "tool_call":
            out.append(e)
    return out


def comparisons(root: Path, prefixes: list[str], trace: bool) -> None:
    """How many comparisons the logs show, per session.

    searches              search_permissible calls (the model asking for candidates)
    candidates_shown      permissible values those searches returned to the model
    server_comparisons    string comparisons the server made (each search scores every value of the attribute)
    vocab_rows_read       permissible values the model read in bulk with list_permissible (normal mode)
    implicit_upper_bound  normal mode: values to map x vocabulary size, summed over enumerated pairs. The model
                          compares in its head, so this is only the most it could have compared; nothing logs it.
    """
    from hbench.data import Data
    from hbench.normalize import canon_value

    data = Data()
    vocab = {a: len(v) for a, v in data.permissible.items()}
    rows, traces = [], []
    for pre in prefixes:
        sid = resolve(root, pre)
        entry = read_index(root)[sid]
        calls = _server_calls(root, sid)
        searches = [c for c in calls if c["tool"] == "search_permissible" and not c.get("error")]
        lists = [c for c in calls if c["tool"] == "list_permissible" and not c.get("error")]
        subs = {}
        for c in calls:
            if c["tool"] == "submit_value_matches" and isinstance(c.get("result"), dict):
                for m in c["arguments"].get("matches", []):
                    subs[(m["source_attribute"], m["target_attribute"], canon_value(m["source_value"]))] = m
        cat = {k: m for k, m in subs.items() if vocab.get(k[1])}
        # values to map per enumerated pair the session answered (for the normal-mode upper bound)
        per_pair = {}
        for (sa, ta, _sv) in cat:
            per_pair[(sa, ta)] = per_pair.get((sa, ta), 0) + 1
        rows.append({
            "run": sid[:8], "mode": entry.get("mode"), "search_only": bool(entry.get("search_only")),
            "top_k": entry.get("search_top_k"), "tool_calls": len(calls),
            "categorical_items_answered": len(cat),
            "searches": len(searches),
            "searches_per_item": round(len(searches) / len(cat), 2) if cat else None,
            "candidates_shown": sum(len(c["result"].get("results", [])) for c in searches),
            "server_comparisons": sum(vocab.get(c["result"].get("attribute"), 0) for c in searches),
            "vocab_rows_read": sum(c.get("result_rows") or 0 for c in lists),
            "implicit_upper_bound": sum(n * vocab[ta] for (sa, ta), n in per_pair.items()) if lists else None,
        })
        if trace:
            for (sa, ta, sv), m in sorted(cat.items()):
                own = [c for c in searches if c["result"].get("attribute") == ta
                       and str(c["result"].get("query", "")).casefold() == sv.casefold()]
                anyq = [c for c in searches if c["result"].get("attribute") == ta]
                tv = m.get("target_value")
                rank = next((i + 1 for c in reversed(anyq) for i, r in enumerate(c["result"]["results"])
                             if r["value"] == tv), None)
                traces.append({"run": sid[:8], "pair": f"{sa} -> {ta}", "source_value": sv,
                               "answer": "NONE" if tv is None else tv,
                               "searches_for_this_value": len(own), "searches_on_attribute": len(anyq),
                               "rank_in_search": rank})
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.max_rows", None,
                           "display.max_colwidth", 45):
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nsearches: search_permissible calls · candidates_shown: values those searches showed the model · "
              "server_comparisons: query-vs-value scorings on the server · vocab_rows_read: values read in bulk with "
              "list_permissible · implicit_upper_bound: normal mode only, values x vocabulary size (not observed)")
        if trace and traces:
            print("\nper item (categorical):\n" + pd.DataFrame(traces).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study")
    ap.add_argument("--e3", action="store_true", help="add E3's BDI-Kit agent runs as reference rows")
    ap.add_argument("--items", nargs=2, metavar=("A", "B"), help="item-level diff of two sessions")
    ap.add_argument("--comparisons", nargs="+", metavar="SESSION",
                    help="searches, candidates shown and comparisons per session (from the server logs)")
    ap.add_argument("--trace", action="store_true", help="with --comparisons: one row per categorical item")
    ap.add_argument("--csv")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root).expanduser() if a.root else log_root()
    if a.items:
        item_diff(root, *a.items)
        return 0
    if a.comparisons:
        comparisons(root, a.comparisons, a.trace)
        return 0
    rows = session_rows(root, a.study) + (e3_rows(a.study) if a.e3 else [])
    if not rows:
        print(f"no scored sessions in {root}")
        return 0
    df = pd.DataFrame(rows).sort_values(["study", "source", "start"])
    if a.csv:
        df.to_csv(a.csv, index=False)
    meta = ["run", "source", "start", "study", "mode", "model", "sql", "search_only", "code"]
    pct = lambda x: "" if pd.isna(x) else f"{100 * x:.1f}"  # noqa: E731
    with pd.option_context("display.width", 220, "display.max_columns", None):
        for c in CATS:
            cols = [f"{c}.{k}" for k in ("n", "strict", "answered", "coverage") if f"{c}.{k}" in df]
            if not cols or df[cols].isna().all().all():
                continue
            t = df[meta + cols].copy()
            t[cols[0]] = t[cols[0]].astype("Int64")
            for k in cols[1:]:
                t[k] = t[k].map(pct)
            t.columns = meta + [k.split(".", 1)[1] for k in cols]
            print(f"\n{c} (% of items)\n" + t.to_string(index=False))
        cal = df[meta + ["brier", "ece"]].dropna(subset=["brier"])
        if len(cal):
            print("\ncalibration (attempted items)\n" + cal.to_string(index=False))
        if "tool_calls" in df:
            t = df[df.source == "claude_vm"][meta + ["tool_calls", "searches", "code_runs"]].copy()
            n_items = df.get("all_but_no_legal_target.n")
            if n_items is not None:
                t["calls_per_item"] = (t.tool_calls / df.loc[t.index, "all_but_no_legal_target.n"]).round(2)
            print("\ntool use (from the server log)\n" + t.to_string(index=False))
    if a.e3:
        print("\nE3 rows: BDI-Kit agent on local Qwen, end to end, asked only for categorical attributes; "
              "compare them with end_to_end sessions, not oracle ones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
