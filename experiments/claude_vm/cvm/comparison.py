"""Cross-run comparison data, shared by compare.py (terminal) and the viewer (/api/compare, /api/diff).
Standard library only, so the viewer stays dependency-free.

Scores are per category, never one number across categories. "strict" is hbench's; "answered" counts an item
correct only if the session submitted it (a value or an explicit NONE).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import REPO_ROOT
from .paths import SessionPaths, read_index

CATS = ("categorical", "numeric_identity", "age_conversion", "no_legal_target", "all_but_no_legal_target")
E3_RUNS = REPO_ROOT / "experiments" / "e3_agent" / "results" / "runs.json"


def _calls(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "tool_call":
                out.append(e)
    return out


def tool_use(path: Path) -> dict:
    """Tool use from a session's server log."""
    calls = _calls(path)
    ok = [c for c in calls if not c.get("error")]
    searches = [c for c in ok if c.get("tool") == "search_permissible"]
    return {
        "tool_calls": len(calls),
        "tool_errors": len(calls) - len(ok),
        "searches": len(searches),
        "candidates_shown": sum(len((c.get("result") or {}).get("results") or []) for c in searches),
        "vocab_rows_read": sum(c.get("result_rows") or 0 for c in ok if c.get("tool") == "list_permissible"),
        "code_saves": sum(1 for c in ok if c.get("tool") == "save_code"),
        "code_runs": sum(1 for c in calls if c.get("tool") == "run_code"),
        "submit_calls": sum(1 for c in ok if c.get("tool") == "submit_value_matches"),
    }


def session_row(root: Path, sid: str, entry: dict) -> dict | None:
    p = SessionPaths(sid, root)
    if not p.scores.exists():
        return None
    s = json.loads(p.scores.read_text())
    cats = {}
    for c, v in (s.get("by_category") or {}).items():
        cats[c] = {"n": v.get("n_items"), "strict": v.get("micro"), "answered": v.get("micro_answered"),
                   "lenient_answered": v.get("micro_lenient_answered"), "coverage": v.get("coverage")}
    cal = (s.get("calibration") or {}).get("all_attempted") or {}
    tools = tool_use(p.server_log)
    n_all = (cats.get("all_but_no_legal_target") or {}).get("n")
    n_cat = (cats.get("categorical") or {}).get("n")
    tools["calls_per_item"] = round(tools["tool_calls"] / n_all, 3) if n_all else None
    tools["searches_per_categorical_item"] = round(tools["searches"] / n_cat, 3) if n_cat else None
    return {
        "run": sid[:8], "session_id": sid, "source": "claude_vm", "start": entry.get("start_time"),
        "study": entry.get("study"), "mode": entry.get("mode"), "status": entry.get("status"),
        "model": ", ".join(entry.get("models_used") or []) or entry.get("model_requested") or "default",
        "sql": bool(entry.get("allow_sql")), "search_only": bool(entry.get("search_only")),
        "search_top_k": entry.get("search_top_k"), "code": bool(entry.get("allow_code")),
        "categories": cats,
        "calibration": {"n": cal.get("n"), "brier": cal.get("brier"), "ece": cal.get("ece")},
        "tools": tools,
        "restricted_to_mapped_pairs": s.get("restricted_to_mapped_pairs") or None,
    }


def e3_rows(study: str | None = None) -> list[dict]:
    """E3's BDI-Kit agent runs that did value matching (clark, end to end, asked for categorical attributes only).
    Only the numbers E3 committed are available: categorical strict accuracy and coverage, all-items accuracy."""
    if not E3_RUNS.exists() or (study and study != "clark"):
        return []
    rows = []
    for label, r in json.loads(E3_RUNS.read_text()).items():
        vm = r.get("vm")
        if not vm:
            continue
        fin = vm.get("final") or {}
        rows.append({
            "run": label, "session_id": None, "source": "e3_bdikit", "start": None, "study": "clark",
            "mode": "end_to_end", "status": r.get("outcome"), "model": r.get("model"),
            "sql": False, "search_only": False, "search_top_k": None, "code": False,
            "categories": {"categorical": {"n": fin.get("n_items"), "strict": fin.get("micro"), "answered": None,
                                           "lenient_answered": None, "coverage": fin.get("coverage")},
                           "all_but_no_legal_target": {"n": None, "strict": vm.get("all_items_micro"),
                                                       "answered": None, "lenient_answered": None,
                                                       "coverage": None}},
            "calibration": {}, "tools": {}, "restricted_to_mapped_pairs": None,
            "note": "BDI-Kit agent on local Qwen (E3); categorical attributes only"})
    return rows


def run_rows(root: Path, study: str | None = None, include_e3: bool = False) -> list[dict]:
    rows = []
    for sid, e in read_index(root).items():
        if study and e.get("study") != study:
            continue
        r = session_row(root, sid, e)
        if r is not None:
            rows.append(r)
    rows.sort(key=lambda r: (r["study"] or "", r["start"] or ""))
    if include_e3:
        rows += e3_rows(study)
    return rows


def flatten(row: dict) -> dict:
    """One flat dict per run (for tables and CSV): categories as '<cat>.<metric>', tools and calibration inline."""
    out = {k: v for k, v in row.items() if k not in ("categories", "calibration", "tools",
                                                     "restricted_to_mapped_pairs")}
    for c, v in (row.get("categories") or {}).items():
        for m, x in v.items():
            out[f"{c}.{m}"] = x
    out.update({f"cal.{k}": v for k, v in (row.get("calibration") or {}).items()})
    out.update(row.get("tools") or {})
    return out


def _items(root: Path, sid: str) -> dict[str, dict]:
    p = SessionPaths(sid, root).items
    with open(p, newline="", encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def item_diff(root: Path, a: str, b: str) -> dict:
    """Per category: both right / only A right / only B right / both wrong; and the items whose answer or
    correctness differs."""
    ia, ib = _items(root, a), _items(root, b)
    summary, rows = {}, []
    for iid in sorted(set(ia) & set(ib)):
        x, y = ia[iid], ib[iid]
        ca, cb = x.get("correct") == "True", y.get("correct") == "True"
        s = summary.setdefault(x["category"], {"both_right": 0, "only_a": 0, "only_b": 0, "both_wrong": 0})
        s["both_right" if ca and cb else "only_a" if ca else "only_b" if cb else "both_wrong"] += 1
        if x.get("pred") != y.get("pred") or ca != cb:
            rows.append({"category": x["category"], "source_attribute": x["source_attribute"],
                         "target_attribute": x["target_attribute"], "source_value": x["key"], "gold": x["gold"],
                         "pred_a": x.get("pred"), "correct_a": ca, "confidence_a": x.get("confidence"),
                         "pred_b": y.get("pred"), "correct_b": cb, "confidence_b": y.get("confidence")})
    return {"a": a, "b": b, "n_common": len(set(ia) & set(ib)), "only_in_a": len(set(ia) - set(ib)),
            "only_in_b": len(set(ib) - set(ia)), "summary": summary, "rows": rows}
