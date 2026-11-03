import pandas as pd
import pytest

from hbench.predictions import vm_record
from hbench.score_vm import score_vm_core

PERM = {"race": {"white", "Unknown", "unknown", "not reported"}, "risk_factor": {"None", "Smoking"},
        "gender": {"male", "female"}}


def items(rows):
    return pd.DataFrame([dict(item_id=f"vm:x:{a}->{t}:{k}", study="x", source_attribute=a, target_attribute=t, key=k,
                              gold=frozenset(g), category=c) for a, t, k, g, c in rows])


def vp(a, t, v, *targets):
    return vm_record("r1", "clark", a, t, v, [x for x in targets if x is not None])


def vm(recs, rows):
    d, s = score_vm_core(recs, items(rows), PERM)
    return d.set_index("key"), s


def test_empty_gold_means_only_none_is_correct():
    rows = [("Race", "race", "Other", [], "categorical")]
    assert vm([vp("Race", "race", "Other", None)], rows)[0].loc["Other", "correct"]
    assert vm([vp("Race", "race", "Other", "white")], rows)[0].loc["Other", "error"] == "false_match"


def test_none_string_is_a_value_only_where_permitted():
    rows = [("Risk", "risk_factor", "no risk", ["None"], "categorical"), ("Race", "race", "x", [], "categorical")]
    d, _ = vm([vp("Risk", "risk_factor", "no risk", "None"), vp("Race", "race", "x", "None")], rows)
    assert d.loc["no risk", "correct"] and d.loc["no risk", "pred"] == "None"   # a real GDC value here
    assert d.loc["x", "correct"] and d.loc["x", "pred"] == "NONE"               # coerced to no match here


def test_one_to_many_missing_cell_and_case_variant():
    rows = [("Race", "race", "", ["Unknown", "not reported"], "categorical")]
    assert vm([vp("Race", "race", float("nan"), "Unknown")], rows)[0].loc["", "correct"]  # NaN source = missing cell
    d, _ = vm([vp("Race", "race", "", "unknown")], rows)
    assert not d.loc["", "correct"] and d.loc["", "lenient_correct"] and d.loc["", "error"] == "case_or_space_variant"


def test_not_permissible_vs_wrong_value():
    rows = [("Race", "race", "Whte", ["white"], "categorical")]
    assert vm([vp("Race", "race", "Whte", "Whiteish")], rows)[0].loc["Whte", "error"] == "not_permissible"
    assert vm([vp("Race", "race", "Whte", "Unknown")], rows)[0].loc["Whte", "error"] == "wrong_value"


def test_n_to_one_items_are_independent():
    rows = [("Sex", "gender", "M", ["male"], "categorical"), ("Sex", "gender", "Male", ["male"], "categorical")]
    d, s = vm([vp("Sex", "gender", "M", "male"), vp("Sex", "gender", "Male", "male")], rows)
    assert d.correct.all() and s["categorical"]["n_items"] == 2


def test_numeric_identity():
    rows = [("BMI", "bmi", "17.22", ["17.22"], "numeric_identity"), ("Size", "s", "1031", ["1031"], "numeric_identity")]
    d, _ = vm([vp("BMI", "bmi", "17.22", "17.220"), vp("Size", "s", 1031.0, 1031)], rows)
    assert d.loc["17.22", "correct"] and d.loc["1031", "correct"]
    d, _ = vm([vp("BMI", "bmi", "17.22", "17.2"), vp("Size", "s", "1031", "n/a value")], rows)
    assert d.loc["17.22", "error"] == "wrong_number" and d.loc["1031", "error"] == "non_numeric"


@pytest.mark.parametrize("pred,correct,lenient,err", [
    ("11322.75", True, True, ""),
    ("11323", True, True, ""),                         # whole days, within 1 day
    ("11315", False, True, "convention_or_rounding"),  # 365-day years
    ("31", False, False, "not_converted_years"),
    ("943.5625", False, False, "years_read_as_months"),
    (None, False, False, "missed"),
])
def test_age_in_years(pred, correct, lenient, err):
    rows = [("Age", "age_at_diagnosis", "31", ["11322.75"], "age_conversion")]
    d, _ = vm([vp("Age", "age_at_diagnosis", "31", pred)], rows)
    assert (d.loc["31", "correct"], d.loc["31", "lenient_correct"], d.loc["31", "error"]) == (correct, lenient, err)


def test_age_in_months_and_unconvertible_value():
    rows = [("Age", "age_at_diagnosis", "700", ["21306.25"], "age_conversion"),
            ("Age", "age_at_diagnosis", ">=90", [], "age_conversion")]
    d, _ = vm([vp("Age", "age_at_diagnosis", "700", str(700 * 365.25)), vp("Age", "age_at_diagnosis", ">=90", None)],
              rows)
    assert d.loc["700", "error"] == "months_read_as_years" and d.loc[">=90", "correct"]
    assert not d.loc["700", "age_exact"]


def test_duplicate_conflicting_and_unmatched_predictions():
    rows = [("Race", "race", "White", ["white"], "categorical")]
    d, s = vm([vp("Race", "race", "White", "white"), vp("Race", "race", " White", "Unknown"),
               vp("Race", "race", "Asian", "asian")], rows)
    assert d.loc["White", "correct"]
    assert (s["duplicate_predictions"], s["conflicting_duplicates"], s["unmatched_predictions"]) == (1, 1, 1)


def test_not_attempted_micro_and_macro():
    rows = [("A", "gender", "M", ["male"], "categorical"), ("A", "gender", "F", ["female"], "categorical"),
            ("B", "race", "W", ["white"], "categorical")]
    d, s = vm([vp("A", "gender", "M", "male"), vp("A", "gender", "F", "male")], rows)
    assert d.loc["W", "error"] == "not_attempted"
    assert s["categorical"]["micro"] == pytest.approx(1 / 3)
    assert s["categorical"]["macro_by_task"] == pytest.approx((0.5 + 0.0) / 2)
    assert s["categorical"]["coverage"] == pytest.approx(2 / 3)


def test_headline_excludes_no_legal_target():
    rows = [("A", "gender", "M", ["male"], "categorical"), ("T", "tumor_purity", "Low", [], "no_legal_target")]
    _, s = vm([vp("T", "tumor_purity", "Low", None)], rows)
    assert s["all_but_no_legal_target"]["n_items"] == 1 and s["no_legal_target"]["micro"] == 1.0
