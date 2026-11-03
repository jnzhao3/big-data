import pytest

from hbench.predictions import sm_record
from hbench.score_sm import recall_at_gt, score_sm_core, vida_pair_mrr

TARGETS = ["gender", "race", "ethnicity", "site_of_resection_or_biopsy", "tissue_or_organ_of_origin",
           "primary_site", "submitter_id", "tumor_grade"]
GOLD = {"Gender": frozenset({"gender"}), "Ethnicity": frozenset({"ethnicity", "race"}),
        "Tumor_Site": frozenset({"site_of_resection_or_biopsy", "tissue_or_organ_of_origin", "primary_site"}),
        "Grade": frozenset({"tumor_grade"})}
COLS = ["Case_ID", "Gender", "Ethnicity", "Tumor_Site", "Grade", "Notes"]


def sm(recs):
    items, s = score_sm_core(recs, COLS, GOLD, TARGETS, study="clark")
    return items.set_index("source_column"), s


def p(col, *targets, otype="top1", **kw):
    return sm_record("r1", "clark", col, list(targets), output_type=otype, **kw)


def test_any_member_of_gold_set_is_correct():
    items, _ = sm([p("Ethnicity", "race"), p("Tumor_Site", "primary_site"), p("Gender", "Gender")])
    assert items.loc["Ethnicity", "correct"] and items.loc["Tumor_Site", "correct"]
    assert items.loc["Gender", "correct"]  # target names compare case-insensitively
    assert items.loc["Gender", "item_id"] == "sm:clark:Gender"


@pytest.mark.parametrize("tok", ["NONE", "none", " None ", "null", "", "N/A", "no match"])
def test_none_tokens(tok):
    items, _ = sm([p("Case_ID", tok)])
    assert items.loc["Case_ID", "pred"] == "NONE"
    assert items.loc["Case_ID", "closed_world_correct"] and items.loc["Case_ID", "error"] == "abstained"


def test_unannotated_columns_are_reported_separately():
    items, s = sm([p("Case_ID", "submitter_id"), p("Notes", "NONE")])
    assert items.loc["Case_ID", "correct"] is None              # not in the annotated accuracy
    assert not items.loc["Case_ID", "closed_world_correct"]     # wrong only under the closed-world reading
    assert items.loc["Case_ID", "error"] == "claimed_unannotated"
    assert s["n_annotated"] == 4 and s["n_unannotated"] == 2 and s["unannotated_abstained"] == 1
    assert s["unannotated_claims"] == [["Case_ID", "submitter_id"]]


def test_error_classes():
    items, s = sm([p("Gender", "anatomic_site"), p("Grade", "race", shown=["tumor_grade", "gender"]),
                   p("Ethnicity", "gender"), p("Tumor_Site", "NONE")])
    assert items.loc["Gender", "error"] == "hallucinated" and s["hallucinated"] == 1
    assert items.loc["Grade", "error"] == "not_in_candidates"
    assert items.loc["Ethnicity", "error"] == "wrong_target"
    assert items.loc["Tumor_Site", "error"] == "missed"


def test_missing_and_invalid_predictions_count_as_wrong():
    items, s = sm([p("Gender", "gender"), sm_record("r1", "clark", "Grade", [], status="invalid_output")])
    assert items.loc["Ethnicity", "error"] == "no_prediction" and items.loc["Ethnicity", "correct"] is False
    assert items.loc["Grade", "error"] == "invalid_output"
    assert s["top1_correct"] == 1 and s["n_predicted_columns"] == 2


def test_ranked_mrr_hits_and_candidate_recall():
    items, s = sm([p("Grade", "race", "gender", "tumor_grade", otype="ranked", shown="*"),
                   p("Gender", "gender", "race", otype="ranked", shown=["race"])])
    assert items.loc["Grade", "rank"] == 3 and items.loc["Grade", "rr"] == pytest.approx(1 / 3)
    assert not items.loc["Grade", "hit@1"] and items.loc["Grade", "hit@3"]
    assert s["mrr"] == pytest.approx((1 / 3 + 1) / 4)  # Ethnicity and Tumor_Site have no prediction -> 0
    assert items.loc["Grade", "candidate_recall"] and not items.loc["Gender", "candidate_recall"]


def test_source_column_matching_and_duplicates():
    items, s = sm([p("gender", "gender"), p("Gender", "race"), p("Unknown_Col", "race")])
    assert items.loc["Gender", "correct"]       # unique case-insensitive match; the later duplicate is ignored
    assert s["duplicate_predictions"] == 1 and s["unknown_source_columns"] == ["Unknown_Col"]


def test_set_output_open_vs_closed_precision():
    _, s = sm([p("Ethnicity", "race", "ethnicity", otype="set"), p("Tumor_Site", "primary_site", "gender", otype="set"),
               p("Case_ID", "submitter_id", otype="set")])
    # TP=3, FP on annotated columns=1 (gender), FP on unannotated=1 (submitter_id)
    assert s["set_closed_P"] == pytest.approx(3 / 5) and s["set_open_P"] == pytest.approx(3 / 4)
    assert s["set_closed_R"] == pytest.approx(3 / 7)


def test_vida_pair_mrr_penalizes_one_to_many():
    assert vida_pair_mrr({"c": ["a", "b"]}, [("c", "a"), ("c", "b")]) == pytest.approx(0.75)


def test_recall_at_gt():
    by_col = {"a": {"targets": ["x", "y"], "scores": [0.9, 0.2]}, "b": {"targets": ["z"], "scores": [0.5]}}
    # top-2 pairs by score: (a,x) 0.9 and (b,z) 0.5; gold has (a,x) and (a,y) -> 1/2
    assert recall_at_gt(by_col, [("a", "x"), ("a", "y")]) == pytest.approx(0.5)
    assert recall_at_gt({"a": {"targets": ["x"], "scores": None}}, [("a", "x")]) is None
