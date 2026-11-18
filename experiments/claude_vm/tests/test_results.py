"""Submissions -> predictions.jsonl -> hbench scores, for value / explicit NONE / not submitted."""

import json

from cvm import results
from cvm.paths import SessionPaths
from cvm.store import SubmissionStore
from hbench.predictions import read_predictions, validate

SID = "00000000-0000-4000-8000-000000000001"


def _submit(t, rows):
    r = t.submit_value_matches([dict(zip(("source_attribute", "target_attribute", "source_value", "target_value",
                                          "confidence"), x)) for x in rows])
    assert r["errors"] == [], r


def test_convert_and_score_oracle(data, tmp_path, make_tools):
    paths = SessionPaths(SID, tmp_path / "logs")
    t = make_tools("oracle")
    t.store = SubmissionStore(paths.submissions)
    _submit(t, [
        ("Race", "race", "White", "white", 0.9),         # correct
        ("Race", "race", "WHITE", "white", 0.9),         # correct (case variant of the source)
        ("Race", "race", "", "not reported", 0.6),       # correct (second member of the gold set)
        ("Race", "race", "Klingon", None, 0.8),          # explicit NONE, empty gold -> correct
        ("Race", "race", "Asian", None, 0.3),            # explicit NONE, gold asian -> missed
        ("Age", "age_at_diagnosis", "30", "10957.5", 0.7),   # correct
        ("Age", "age_at_diagnosis", "41", "14965", 0.4),     # 365-day year -> lenient only
        # Age 55 not submitted -> not_attempted
    ])
    info = results.write_session_predictions(paths, "clark", "oracle", data=data)
    assert info["n_records"] == 7 and info["n_explicit_none"] == 2
    recs = read_predictions(paths.predictions)
    assert validate(recs) == ([], [])
    none_rec = next(r for r in recs if r["source_value"] == "Klingon")
    assert none_rec["targets"] == [] and none_rec["scores"] is None and none_rec["confidence"] == 0.8

    s = results.score_session(paths, "clark", "oracle", data=data)
    items = {(r["source_attribute"], r["key"]): r for r in __import__("pandas").read_csv(
        paths.items, keep_default_na=False).to_dict("records")}
    assert items[("Race", "Klingon")]["correct"] and items[("Race", "")]["correct"]
    assert items[("Race", "Asian")]["error"] == "missed"
    assert items[("Age", "55")]["error"] == "not_attempted"
    assert items[("Age", "41")]["lenient_correct"] and not items[("Age", "41")]["correct"]
    hs = s["hbench_summary"]
    assert hs["categorical"]["n_items"] == 5 and hs["categorical"]["micro"] == 0.8
    assert hs["age_conversion"]["coverage"] == round(2 / 3, 10) or abs(hs["age_conversion"]["coverage"] - 2 / 3) < 1e-9
    cal = s["calibration"]["all_attempted"]
    assert cal["n"] == 7 and 0 <= cal["brier"] <= 1 and cal["buckets"]
    assert json.loads(paths.scores.read_text())["mode"] == "oracle"


def test_end_to_end_only_mapped_pairs_count(data, tmp_path, make_tools):
    paths = SessionPaths(SID, tmp_path / "logs")
    t = make_tools("end_to_end")
    t.store = SubmissionStore(paths.submissions)
    t.map_attribute("Race", "race")
    t.map_attribute("Race", "gender")
    _submit(t, [("Race", "race", "White", "white", 0.9), ("Race", "gender", "White", None, 0.5)])
    t.unmap_attribute("Race", "gender")
    info = results.write_session_predictions(paths, "clark", "end_to_end", data=data)
    assert info["n_records"] == 1 and info["pairs"] == [["Race", "race"]]
    s = results.score_session(paths, "clark", "end_to_end", data=data)
    assert s["mapped_pairs"] == {"n_mapped": 1, "n_mapped_in_gt": 1, "n_gt_pairs": 2}
    assert s["restricted_to_mapped_pairs"]["categorical"]["n_items"] == 5
    assert s["hbench_summary"]["age_conversion"]["coverage"] == 0.0


def test_unsubmitted_empty_gold_item_is_not_counted_as_answered(data, tmp_path, make_tools):
    """hbench counts an unsubmitted item with an empty gold set (Klingon) as correct; the harness's
    answered-only score does not."""
    paths = SessionPaths(SID, tmp_path / "logs")
    t = make_tools("oracle")
    t.store = SubmissionStore(paths.submissions)
    results.write_session_predictions(paths, "clark", "oracle", data=data)   # nothing submitted
    s = results.score_session(paths, "clark", "oracle", data=data)
    cat = s["by_category"]["categorical"]
    assert cat["coverage"] == 0.0 and cat["micro"] == 0.2 and cat["micro_answered"] == 0.0
    assert s["calibration"]["all_attempted"] == {"n": 0}
    assert results.headline(s)["categorical"]["micro_answered"] == 0.0
