import pytest

from cvm.tools import METHODS, ToolError, similarity, shape


def test_describe_study_modes(make_tools):
    o = make_tools("oracle").describe_study()
    assert o["mode"] == "oracle" and o["n_rows"] == 6
    assert {(p["source_attribute"], p["target_attribute"]) for p in o["pairs"]} == {
        ("Race", "race"), ("Age", "age_at_diagnosis")}
    assert [c["name"] for c in o["columns"]] == ["Case", "Race", "Age"]
    e = make_tools("end_to_end").describe_study()
    assert "pairs" not in e


def test_list_values_includes_missing_and_counts(make_tools):
    r = make_tools().list_values("Race")
    vals = {v["value"]: v for v in r["values"]}
    assert vals["White"]["count"] == 2 and vals[""]["missing"] is True
    assert "GTONLYSENTINEL" not in vals


def test_row_cap_and_pagination(make_tools):
    t = make_tools(cap=2)
    p1 = t.list_values("Race", offset=0, limit=100)
    assert p1["returned"] == 2 and p1["row_cap"] == 2 and p1["next_offset"] == 2
    p2 = t.list_values("Race", offset=p1["next_offset"], limit=2)
    p3 = t.list_values("Race", offset=p2["next_offset"], limit=2)
    seen = [v["value"] for p in (p1, p2, p3) for v in p["values"]]
    assert len(seen) == len(set(seen)) == 5 and p3["next_offset"] is None
    assert t.sample_rows(n=1000)["returned"] == 2
    assert len(t.list_permissible("race", limit=99)["values"]) == 2
    assert t.search_attributes("race", top_k=99)["returned"] <= 2
    assert len(t.describe_study()["columns"]) == 2


def test_profile_column(make_tools):
    p = make_tools().profile_column("Race")
    assert p["n_distinct_values"] == 4 and p["null_rate"] == round(1 / 6, 4)
    assert {"upper", "title"} <= set(p["casing"])
    assert p["top_values"][0] == {"value": "White", "count": 2, "missing": False}
    assert make_tools().profile_column("Age")["numeric"] == {"min": 30.0, "max": 55.0, "mean": 42.0}


def test_sample_rows_seeded_and_stratified(make_tools):
    t = make_tools()
    assert t.sample_rows(3, seed=7)["rows"] == t.sample_rows(3, seed=7)["rows"]
    s = t.sample_rows(5, seed=1, stratify_by="Age")["rows"]
    assert {r["Age"] for r in s} == {"30", "41", "55"}
    with pytest.raises(ToolError):
        t.sample_rows(3, stratify_by="nope")


def test_vocabulary_tools(make_tools):
    t = make_tools()
    assert t.describe_attribute("age_at_diagnosis")["enumerated"] is False
    assert t.search_attributes("race")["results"][0]["attribute"] == "race"
    assert t.search_permissible("race", "WHITE", method="normalized")["results"][0]["value"] == "white"
    for m in METHODS:
        assert t.search_permissible("race", "Asian", method=m)["returned"] >= 1
    with pytest.raises(ToolError):
        t.search_permissible("age_at_diagnosis", "30")
    with pytest.raises(ToolError, match="closest names"):
        t.describe_attribute("racee")
    with pytest.raises(ToolError):
        similarity("a", "b", "nope")


def test_shape():
    assert shape("White") == "Aa" and shape("WHITE") == "A" and shape("C3L-00004") == "A9A-9"


def test_submit_validation_and_overwrite(make_tools):
    t = make_tools()
    r = t.submit_value_matches([
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "White", "target_value": "white",
         "confidence": 0.9},
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "WHITE", "target_value": "White",
         "confidence": 0.9},                                        # case error with a hint
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "Klingon", "target_value": None,
         "confidence": 0.7},                                        # explicit NONE
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "Asian", "target_value": "NONE",
         "confidence": 0.2},                                        # NONE spelled as a string -> NONE + warning
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "Martian", "target_value": "white",
         "confidence": 0.5},                                        # not a source value
        {"source_attribute": "Race", "target_attribute": "gender", "source_value": "White", "target_value": "male",
         "confidence": 0.5},                                        # not a given pair
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "", "target_value": "Unknown"},
        {"source_attribute": "Age", "target_attribute": "age_at_diagnosis", "source_value": 30,
         "target_value": "thirty", "confidence": 1.5},
        {"source_attribute": "Age", "target_attribute": "age_at_diagnosis", "source_value": "41",
         "target_value": "about 41 years", "confidence": 0.3},     # not a number -> recorded with a warning
    ])
    assert r["accepted"] == 4
    errs = {e["index"]: e["error"] for e in r["errors"]}
    assert "did you mean 'white'" in errs[1] and "not a value" in errs[4] and "not one of the given" in errs[5]
    assert "confidence" in errs[6] and "between 0 and 1" in errs[7]
    assert {w["index"] for w in r["warnings"]} == {3, 8}
    state = t.store.matches()
    assert state[("Race", "race", "Klingon")]["target_value"] is None
    assert state[("Race", "race", "Asian")]["target_value"] is None
    t.submit_value_matches([{"source_attribute": "Race", "target_attribute": "race", "source_value": "Klingon",
                             "target_value": "asian", "confidence": 0.1}])
    assert t.store.matches()[("Race", "race", "Klingon")]["target_value"] == "asian"
    with pytest.raises(ToolError):
        t.submit_value_matches([])


def test_progress(make_tools):
    t = make_tools()
    t.submit_value_matches([
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "White", "target_value": "white",
         "confidence": 0.9},
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "", "target_value": None,
         "confidence": 0.5}])
    p = {(r["source_attribute"], r["target_attribute"]): r for r in t.get_progress()["pairs"]}
    race = p[("Race", "race")]
    assert race == {"source_attribute": "Race", "target_attribute": "race", "n_values": 5, "submitted_values": 1,
                    "explicit_none": 1, "remaining": 3}
    assert p[("Age", "age_at_diagnosis")]["remaining"] == 3


def test_end_to_end_mapping(make_tools):
    t = make_tools("end_to_end")
    m = {"source_attribute": "Race", "target_attribute": "race", "source_value": "White", "target_value": "white",
         "confidence": 0.9}
    assert "map_attribute first" in t.submit_value_matches([m])["errors"][0]["error"]
    t.map_attribute("Race", "race")
    t.map_attribute("Race", "gender")
    assert t.submit_value_matches([m])["accepted"] == 1
    assert len(t.get_progress()["pairs"]) == 2 and "Age" in t.get_progress()["unmapped_source_columns"]
    t.unmap_attribute("Race", "gender")
    assert t.store.mapped_pairs() == [("Race", "race")]
    with pytest.raises(ToolError):
        t.unmap_attribute("Race", "gender")
    with pytest.raises(ToolError):
        make_tools("oracle").map_attribute("Race", "race")


def test_search_only_mode(data, tmp_path):
    from cvm.sqlguard import SqlGuard, build_db
    from cvm.store import SubmissionStore
    from cvm.study import StudyView
    from cvm.tools import MatcherTools

    view = StudyView(data, "clark", "oracle")
    t = MatcherTools(view, SubmissionStore(tmp_path / "s.jsonl"), search_only=True, search_top_k=2)
    with pytest.raises(ToolError, match="disabled"):
        t.list_permissible("race")
    assert "search_only" in t.describe_study()
    r = t.search_permissible("race", "White", top_k=50)
    assert r["returned"] == 2 and r["row_cap"] == 2
    found = r["results"][0]["value"]
    m = {"source_attribute": "Race", "target_attribute": "race", "source_value": "White", "confidence": 0.9}
    # a permissible value that no search returned is rejected, and the error names no vocabulary value
    other = next(v for v in ("white", "asian", "Unknown", "not reported") if (("race", v) not in t.searched))
    res = t.submit_value_matches([{**m, "target_value": other}, {**m, "source_value": "WHITE", "target_value": found},
                                  {**m, "source_value": "Asian", "target_value": None}])
    assert res["accepted"] == 2 and "search for it first" in res["errors"][0]["error"]
    assert "did you mean" not in res["errors"][0]["error"]
    with pytest.raises(ValueError, match="SQL"):
        MatcherTools(view, SubmissionStore(tmp_path / "s2.jsonl"), sql=SqlGuard(build_db(tmp_path / "m.db", view), 5),
                     search_only=True)
