import json
from pathlib import Path

import pandas as pd
import pytest

from hbench.decisions import (decision_margin, evidence, json_field_parser, parser_from_spec, position_margin,
                              token_span, value_char_span)
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
    ev = evidence("driver:5", tl, PILOT["cold"]["content"], "bcr_id", json_key="target", temperature=0)
    assert ev["span_id"] == "driver:5" and round(ev["margin"], 4) == 0.1266
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


TOOL = json.loads((Path(__file__).parent / "fixtures" / "probe3_tool_call.json").read_text())
TOOL_SPEC = [{"type": "tool_call", "field": "name"}, {"type": "tool_call", "field": "arguments.source_column"}]


def _normalized(transport):
    from hbench.llm.normalize import NORMALIZERS
    return NORMALIZERS[transport](TOOL[transport]["request"], TOOL[transport]["response"])


@pytest.mark.parametrize("transport,name_margin,arg_margin", [("ollama", 10.53, 8.96), ("openai_v1", 10.52, 8.92)])
def test_tool_call_decisions_from_spec(transport, name_margin, arg_margin):
    n = _normalized(transport)
    assert n["output"]["text"] == ""  # the tool call exists only in the generated tokens
    ds = parser_from_spec(TOOL_SPEC)(n["output"])
    assert [(d["key"], d["value"], d["channel"], d["index"]) for d in ds] == [
        ("name", "list_target_columns", "tool_call", 0), ("arguments.source_column", "Tumor_Size_cm", "tool_call", 0)]
    got = [decision_margin(n["token_logprobs"], n["output"]["text"], d["value"], decision=d) for d in ds]
    assert all(g["found"] for g in got)
    assert [round(g["margin_value"], 2) for g in got] == [name_margin, arg_margin]


def _synthetic(pieces):
    """Token i has a gap of i + 1 to its runner-up, so a margin names the first token of the span it covers."""
    return [[t, -0.1, [[t, -0.1], ["<alt>", -0.1 - (i + 1)]]] for i, t in enumerate(pieces)]


SYN = _synthetic(["Let me check. ", "<tool_call>", '\n{"name": "', "a", '", "arguments": {"k": ', "3",
                  "}}\n</tool_call>", "<tool_call>", '\n{"name": "', "b", '", "arguments": {"x": {"y": "', "deep",
                  '"}}}\n</tool_call>'])
SYN_OUTPUT = {"text": "Let me check. ", "tool_calls": [{"name": "a", "arguments": {"k": 3}},
                                                     {"name": "b", "arguments": {"x": {"y": "deep"}}}]}


@pytest.mark.parametrize("spec,span", [
    ({"type": "tool_call", "field": "name"}, [3]),
    ({"type": "tool_call", "field": "arguments.k"}, [5]),                 # an integer value, after text content
    ({"type": "tool_call", "field": "name", "index": 1}, [9]),           # the second call, not the first
    ({"type": "tool_call", "field": "arguments.x.y", "index": 1}, [11]),  # a nested argument
    ({"type": "tool_call", "field": "*"}, [1, 2, 3, 4, 5, 6]),           # the whole first call
])
def test_tool_call_locator_edge_cases(spec, span):
    (d,) = parser_from_spec(spec)(SYN_OUTPUT)
    got = decision_margin(SYN, SYN_OUTPUT["text"], d["value"], decision=d)
    assert got["found"] and got["span"] == span


def test_tool_call_locator_reports_what_it_cannot_find():
    assert parser_from_spec({"type": "tool_call", "field": "arguments.missing"})(SYN_OUTPUT) == []
    assert parser_from_spec({"type": "tool_call", "field": "name", "index": 5})(SYN_OUTPUT) == []
    d = {"key": "name", "value": "zzz", "json_key": "name", "channel": "tool_call", "index": 0, "field": "name"}
    got = decision_margin(SYN, SYN_OUTPUT["text"], "zzz", decision=d)
    assert not got["found"] and got["reason"]


def test_prefix_margin_covers_tokens_before_the_value():
    d = decision_margin(_tokens(PILOT["cold"]), PILOT["cold"]["content"], "bcr_id", json_key="target")
    assert d["margin_def"] == "prefix/1" and round(d["margin"], 4) == round(d["margin_value"], 4) == 0.1266
    toks = [["{\"", -0.01, [["{\"", -0.01], ["x", -0.02]]],      # gap 0.01 before the value
            ["target", -0.1, [["target", -0.1], ["t", -5.0]]], ["\": \"", -0.1, [["\": \"", -0.1], [":", -6.0]]],
            ["bmi", -0.1, [["bmi", -0.1], ["age", -2.0]]], ["\"}", -0.1, [["\"}", -0.1], ["\",", -0.2]]]]
    d = decision_margin(toks, '{"target": "bmi"}', "bmi", json_key="target")
    assert d["margin"] == pytest.approx(0.01) and d["margin_pos"] == 0 and d["margin_value"] == pytest.approx(1.9)
    assert d["end"] == 3  # the gap of 0.1 after the value does not count


def test_constrained_position_gets_a_conservative_gap():
    # format=json: the grammar forbade "}\n\n", so "}" was chosen although it is not the top logprob
    toks = [["{\"", -0.1, [["{\"", -0.1], ["x", -9.0]]], ["}", -0.99, [["}\n\n", -0.46], ["}", -0.99], ["\"", -3.0]]]]
    d = decision_margin(toks, '{"}', "}", decision={"key": "t", "value": "}", "json_key": None}, greedy=True)
    assert d["n_constrained"] == 1 and d["margin"] == pytest.approx(2.01)  # chosen minus the best below it
    alone = [["}", -0.99, [["}\n\n", -0.46], ["}", -0.99]]]]
    d = decision_margin(alone, "}", "}", decision={"key": "t", "value": "}", "json_key": None}, greedy=True)
    assert d["n_constrained"] == 1 and d["margin"] == 0.0                   # nothing visible below: only >= 0
    d = decision_margin(alone, "}", "}", decision={"key": "t", "value": "}", "json_key": None}, greedy=False)
    assert d["n_constrained"] == 0 and d["margin"] == pytest.approx(-0.53)  # sampled runs keep the plain gap


def test_evidence_v2_columns():
    from hbench.decisions import evidence_columns
    tl = _tokens(PILOT["cold"])
    ev = evidence("driver:5", tl, PILOT["cold"]["content"], "bcr_id", json_key="target", temperature=0,
                  prompt_sha256="abc")
    assert ev["margin_def"] == "prefix/1" and set(ev["p_path_default"]) == {"/api/chat", "/v1/chat/completions"}
    assert "p_flip" not in ev and "p_path" not in ev        # T=0: the actual sampler reproduces the path surely
    cols = evidence_columns(ev)
    assert round(cols["margin"], 4) == 0.1266 and cols["prompt_sha256"] == "abc"
    assert 0 < cols["p_path_v1"] < cols["p_path_api_chat"] <= 1
    old = evidence_columns({"span_id": "driver:1", "margin": 0.2})  # a v2 record: margin kept, no value margin
    assert old["margin"] == 0.2 and old["margin_value"] is None and old["p_path_api_chat"] is None
    hot = evidence("driver:6", tl, PILOT["cold"]["content"], "bcr_id", json_key="target", temperature=0.8,
                   sampler_params={"temp": 0.8, "top_k": 40, "top_p": 0.9, "min_p": 0.0})
    assert 0 < hot["p_path"] <= 1 and hot["greedy_path"] is True and hot["p_path"] == hot["p_path_default"]["/api/chat"]["p"]


def test_tool_call_decisions_of_different_calls_have_different_keys():
    spec = [{"type": "tool_call", "field": "name"}, {"type": "tool_call", "field": "name", "index": 1}]
    assert [d["key"] for d in parser_from_spec(spec)(SYN_OUTPUT)] == ["name", "name@1"]


def test_a_registered_parser_that_reads_outputs_sees_tool_calls():
    from hbench.decisions import decision_parser, register_decision_parser, tool_call_parser
    register_decision_parser("agent:test", tool_call_parser("name"))
    assert [d["value"] for d in decision_parser("agent:test")(SYN_OUTPUT)] == ["a"]
    register_decision_parser("legacy:test", lambda text: [{"key": "k", "value": text, "json_key": None}])
    assert decision_parser("legacy:test")({"text": "hi", "tool_calls": None})[0]["value"] == "hi"


def test_tool_call_decision_needs_no_message_text():
    (d,) = parser_from_spec({"type": "tool_call", "field": "name"})(SYN_OUTPUT)
    got = decision_margin(SYN, None, d["value"], decision=d)  # OpenAI-style replies carry content null
    assert got["found"] and got["span"] == [3]


def test_an_argument_key_is_not_confused_with_an_equal_value():
    toks = _synthetic(["<tool_call>", '\n{"name": "', "f", '", "arguments": {"x": "', "b", '", "c": "', "zz",
                       '", "b": "', "zz", '"}}\n</tool_call>'])
    out = {"text": "", "tool_calls": [{"name": "f", "arguments": {"x": "b", "c": "zz", "b": "zz"}}]}
    (d,) = parser_from_spec({"type": "tool_call", "field": "arguments.b"})(out)
    assert decision_margin(toks, "", d["value"], decision=d)["span"] == [8]  # b's value, not c's
