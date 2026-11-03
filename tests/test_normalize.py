import numpy as np
import pandas as pd
import pytest

from hbench.normalize import NONE, canon_target_col, canon_value, lenient_value, num, to_decimal
from hbench.studies import STUDIES, get_study, study_id


@pytest.mark.parametrize("raw,want", [
    ("1031.0", "1031"), ("-1.0", "-1"), ("0.00", "0"), ("-0.0", "0"), ("+5.0", "5"),
    ("12345678901234567890.0", "12345678901234567890"),  # no float round-trip, no precision loss
    ("1031.5", "1031.5"), ("1.50", "1.50"), ("5.12e-08", "5.12e-08"), ("1e3", "1e3"),
    ("007", "007"), ("na", "na"), ("NaN", "NaN"), ("", ""), (">=90", ">=90"), ("1_000", "1_000"), ("inf", "inf"),
])
def test_num(raw, want):
    assert num(raw) == want


@pytest.mark.parametrize("raw,want", [
    (None, ""), (float("nan"), ""), (np.nan, ""), (pd.NA, ""),
    (1031.0, "1031"), (np.float64(17.22), "17.22"), (5, "5"), (np.int64(-9), "-9"),
    (" White", "White"), ("pT2 ", "pT2"), ("\u00a0x\u00a0", "x"), ("e\u0301", "\u00e9"),
    ("Cytokeratin 7 \u2013 positive", "Cytokeratin 7 \u2013 positive"),  # en-dash kept
    ("R0:  No residual tumor", "R0:  No residual tumor"),               # inner spaces kept, as in GDC-VM
])
def test_canon_value(raw, want):
    assert canon_value(raw) == want


def test_float_round_trip_would_be_wrong():
    assert str(int(float("12345678901234567890.0"))) != "12345678901234567890"
    assert float("1_000") == 1000.0


@pytest.mark.parametrize("raw,want", [("Gender", "gender"), (" `race` ", "race"), ('"BMI"', "bmi"),
                                      ("NONE", NONE), ("none", NONE), ("N/A", NONE), ("", NONE), (None, NONE),
                                      (float("nan"), NONE), ("no match", NONE)])
def test_canon_target_col(raw, want):
    assert canon_target_col(raw) == want


def test_lenient_and_decimal():
    assert lenient_value("Not  Reported") == lenient_value("not reported")
    assert to_decimal("17.220") == to_decimal("17.22")
    assert to_decimal("inf") is None and to_decimal("abc") is None and to_decimal(None) is None


@pytest.mark.parametrize("name", ["Gillette", "gilette", "GILLETTE", " gillete ", "Gil lette"])
def test_study_id_gillette(name):
    assert study_id(name) == "gillette"


def test_study_table():
    assert len(STUDIES) == 10
    assert get_study("McDermott").id == "mcdermott"
    assert get_study("gillette").gdc_new_dir == "gilette" and get_study("gillette").name == "Gillette"
    with pytest.raises(KeyError):
        study_id("gdc")
