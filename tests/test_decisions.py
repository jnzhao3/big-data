import json
from pathlib import Path

import pandas as pd
import pytest

from hbench.decisions import (decision_margin, evidence, json_field_parser, p_flip, parser_from_spec,
                              position_margin, token_span, value_char_span)
from hbench.predictions import sm_record
from hbench.score_sm import score_sm_core
from hbench.stability import flips_by_margin, fragile_items, stability_report

PILOT = json.loads((Path(__file__).parent / "fixtures" / "pilot_case_id_logprobs.json").read_text())["paths"]


def _tokens(rec):
    return [[t, next(lp for c, lp in top if c == t), top] for t, top in zip(rec["tokens"], rec["top"])]


@pytest.mark.parametrize("path,want", [("cold", 0.1266), ("repeat_1tok_suffix", 0.1196)])
def test_margin_reproduces_the_e1_pilot(path, want):
    rec = PILOT[path]
    d = decision_margin(_tokens(rec), rec["content"], "bcr_id", json_key="target")
    assert d["found"] and d["aligned"] and round(d["margin"], 4) == want
    assert (d["chosen_token"], d["runner_up"]) == ("b", "submit")  # bcr_id vs submitter_id, decided at token 1


def test_span_margin_differs_from_whole_output_minimum():
    tl = _tokens(PILOT["repeat_1tok_suffix"])
    whole = min(m for m, _, _ in (position_margin(t) for t in tl) if m is not None)
    span = decision_margin(tl, PILOT["repeat_1tok_suffix"]["content"], "bcr_id", "target")["margin"]
    assert round(whole, 4) == 0.1145 and round(span, 4) == 0.1196  # the whole-output minimum is a non-decision token


def test_value_located_after_its_json_key_not_in_the_reason():
    text = '{"reason": "gender looks right", "target": "gender"}'
    a, b = value_char_span(text, "gender", json_key="target")
    assert text[a:b] == "gender" and a > text.index("target")
    assert token_span(["{\"", "reason", "\": \"", "gender"], 12, 18) == [3]


def test_p_flip():
    entry = ["a", -0.1, [["a", -0.1], ["b", -0.2]]]
    assert p_flip(entry, 0) == 0.0 and 0.4 < p_flip(entry, 1.0) < 0.5
    assert p_flip(entry, 0.1) < p_flip(entry, 1.0)


def test_missing_logprobs_and_values_are_explicit():
    assert decision_margin(None, "x", "x")["reason"] == "no token logprobs"
    assert decision_margin([["x", -0.1, [["x", -0.1]]]], "x", "x")["reason"] == "no alternative candidates in top_logprobs"
    assert decision_margin([["y", -0.1, []]], "y", "zzz")["reason"] == "value not found in the output tokens"
    assert json_field_parser("target")('{"target": "bmi"}') == [{"key": "target", "value": "bmi", "json_key": "target"}]
    assert json_field_parser("target")("not json") == []


def test_parser_from_spec():
    assert parser_from_spec({"type": "json_field", "field": "target"})('{"target": "bmi"}')[0]["value"] == "bmi"
    assert parser_from_spec({"type": "text"})(" Yes\n") == [{"key": "text", "value": "Yes", "json_key": None}]
    assert parser_from_spec(None) is None
    with pytest.raises(ValueError):
        parser_from_spec({"type": "regex"})


def test_evidence_flows_into_items_and_stability():
    tl = _tokens(PILOT["cold"])
    ev = evidence("driver:5", tl, PILOT["cold"]["content"], "bcr_id", json_key="target", temperature=0.8)
    assert ev["span_id"] == "driver:5" and round(ev["margin"], 4) == 0.1266 and 0.4 < ev["p_flip"] < 0.6
    rec = sm_record("r1", "clark", "Case_ID", ["bcr_id"], evidence=ev)
    items, _ = score_sm_core([rec], ["Case_ID"], {}, ["bcr_id"], study="clark")
    assert round(items.loc[0, "margin"], 4) == 0.1266 and items.loc[0, "evidence_span"] == "driver:5"

    long = pd.DataFrame([dict(run_id=f"r{r}", item_id=i, pred=p, correct=None, margin=m)
                         for r in range(2) for i, p, m in [("a", "x", 0.01), ("b", "y" if r else "z", 0.03),
                                                           ("c", "w", 0.8)]])
    rows = {r["margin_from"]: r for r in flips_by_margin(long)}
    assert rows[0.0]["n_items"] == 2 and rows[0.0]["n_changed"] == 1 and rows[0.5]["flip_rate"] == 0.0
    assert fragile_items(long, eps=0.02) == 1
    assert stability_report(long)["margin_quantiles"][0.5] == pytest.approx(0.03)
