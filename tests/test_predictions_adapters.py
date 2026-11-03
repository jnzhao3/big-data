import json

import numpy as np
import pandas as pd
import pytest

from hbench.adapters.bdikit import from_bdikit_schema_df, from_bdikit_value_df
from hbench.adapters.clark_match import from_clark_match_csv
from hbench.adapters.magneto import from_magneto_matches, from_magneto_txt
from hbench.normalize import canon_value
from hbench.predictions import jsonable, read_predictions, sm_record, validate, vm_record, write_predictions


def test_write_read_roundtrip_and_never_overwrite(tmp_path):
    recs = [sm_record("r1", "clark", "Age", ["age_at_diagnosis"], scores=[np.float64(0.9)]),
            vm_record("r1", "clark", "BMI", "bmi", np.float64(17.22), [np.int64(17)])]
    p = write_predictions(tmp_path / "predictions.jsonl", recs)
    back = read_predictions(p)
    assert back[0]["scores"] == [0.9] and back[1]["source_value"] == 17.22 and back[1]["targets"] == [17]
    with pytest.raises(FileExistsError):
        write_predictions(p, recs)


def test_jsonable_handles_nan_and_numpy():
    assert jsonable({"a": float("nan"), "b": np.int64(3), "c": [np.float32(1.5), pd.NA]}) == {"a": None, "b": 3,
                                                                                              "c": [1.5, None]}


def test_validate_reports_errors_and_warnings():
    good = sm_record("r1", "clark", "Age", ["age_at_diagnosis"])
    bad = [dict(good, output_type="best"), dict(good, targets="age"), {"kind": "xx"},
           dict(good, targets=["a", "b"]), sm_record("r1", "clark", "Age", ["made_up"])]
    errors, warnings = validate([good] + bad, target_names=["age_at_diagnosis"])
    assert len(errors) == 5  # {"kind": "xx"} has both a wrong schema and a wrong kind
    assert any("not in the GDC schema" in w for w in warnings) and any("duplicate" in w for w in warnings)
    assert validate([good], target_names=["age_at_diagnosis"]) == ([], [])


def test_clark_match_csv(tmp_path):
    p = tmp_path / "clark_full_qwen2.5-14b_run1.csv"
    p.write_text("source,gold,in_shortlist,pred,correct,status,secs,reason\n"
                 "Case_ID,NONE,True,submitter_id,False,ok,20.7,x\n"
                 "Tumor_Site,a|b,True,anatomic_site,False,hallucinated,2.3,y\n"
                 "Grade,tumor_grade,True,NONE,False,ok,1.0,z\n"
                 "BMI,bmi,True,,False,invalid_json,1.0,{bad\n")
    recs = from_clark_match_csv(p)
    assert [r["targets"] for r in recs] == [["submitter_id"], ["anatomic_site"], [], []]
    assert [r["status"] for r in recs] == ["ok", "ok", "ok", "invalid_output"]
    assert recs[0]["run_id"] == "clark_full_qwen2.5-14b_run1" and validate(recs)[0] == []


def test_magneto_matches_keep_full_ranking():
    matches = {(("source", "Age"), ("target", "age_at_onset")): 0.8, (("source", "Age"), ("target", "age_at_diagnosis")): 0.9,
               (("source", "BMI"), ("target", "bmi")): 1.0}
    recs = from_magneto_matches(matches, "m1", "clark", source_columns=["Age", "BMI", "Case_ID"])
    by = {r["source_column"]: r for r in recs}
    assert by["Age"]["targets"] == ["age_at_diagnosis", "age_at_onset"] and by["Age"]["scores"] == [0.9, 0.8]
    assert by["Case_ID"]["targets"] == [] and by["Age"]["output_type"] == "ranked" and by["Age"]["shown"] == "*"


def test_magneto_txt_top1_only():
    txt = ("Age                              -> age_at_onset                        ✗ gold_rank=2\n"
           "Histologic_Type                  -> papillary_renal_cell_type           ✗ gold_rank=None\n")
    assert [r["targets"] for r in from_magneto_txt(txt)] == [["age_at_onset"], ["papillary_renal_cell_type"]]


def test_bdikit_frames():
    sm = pd.DataFrame([["Age", "age_at_diagnosis", 0.9], ["Notes", np.nan, np.nan]],
                      columns=["source_attribute", "target_attribute", "similarity"])
    recs = from_bdikit_schema_df(sm, "b1", "clark")
    assert recs[0]["targets"] == ["age_at_diagnosis"] and recs[1]["targets"] == []
    ranked = pd.DataFrame([["Age", "b", 0.5], ["Age", "a", 0.9]], columns=["source_attribute", "target_attribute",
                                                                            "similarity"])
    assert from_bdikit_schema_df(ranked, "b1", "clark", "ranked")[0]["targets"] == ["a", "b"]
    vm = pd.DataFrame([["Race", "race", "White", "white", 0.9], ["Race", "race", np.nan, np.nan, np.nan],
                       ["Size", "s", 1031.0, np.nan, np.nan]],
                      columns=["source_attribute", "target_attribute", "source_value", "target_value", "similarity"])
    recs = from_bdikit_value_df(vm, "b1", "clark")
    assert [r["targets"] for r in recs] == [["white"], [], []]
    assert canon_value(recs[1]["source_value"]) == "" and canon_value(recs[2]["source_value"]) == "1031"
    assert json.loads(json.dumps(jsonable(recs)))  # serializable
