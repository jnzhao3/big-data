"""E1 Stage 2 tools: the synthetic tool loop's decisions and messages, run plans, and the analysis helpers.
No model calls: the loop itself is exercised against a live server by running a plan."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hbench.data import REPO_ROOT
from hbench.decisions import parser_from_spec

sys.path.insert(0, str(REPO_ROOT / "experiments" / "e1_noise_floor"))
import analyze_stage2  # noqa: E402
import run_plan  # noqa: E402
import tool_loop  # noqa: E402


def test_decision_spec_names_parallel_tool_calls():
    out = {"text": "", "tool_calls": [
        {"name": "get_column_values", "arguments": {"column": "tumor_depth", "side": "target"}},
        {"name": "submit_match", "arguments": {"source_column": "Tumor_Size_cm", "target": "NONE"}}]}
    found = {d["key"]: d["value"] for d in parser_from_spec(tool_loop.DECISION)(out)}
    assert found == {"tool": "get_column_values", "column": "tumor_depth", "tool@1": "submit_match",
                     "target@1": "NONE"}
    text_only = {"text": '{"target": "bmi"}', "tool_calls": None}
    assert {d["key"]: d["value"] for d in parser_from_spec(tool_loop.DECISION)(text_only)} == {"text_target": "bmi"}


def test_text_target():
    assert tool_loop.text_target('{"target": " age_at_index "}') == "age_at_index"
    assert tool_loop.text_target("not json") is None
    assert tool_loop.text_target('["a"]') is None


def test_next_messages_follow_each_transport():
    calls = [{"name": "list_candidate_columns", "arguments": {"source_column": "x"}}]
    native = SimpleNamespace(tool_calls=calls, response={"message": {"content": "", "tool_calls": [
        {"function": {"name": "list_candidate_columns", "arguments": {"source_column": "x"}}}]}})
    msgs = tool_loop.next_messages(native, "ollama", ['["a", "b"]'])
    assert msgs[0]["role"] == "assistant" and msgs[0]["tool_calls"][0]["function"]["name"] == "list_candidate_columns"
    assert msgs[1] == {"role": "tool", "content": '["a", "b"]', "tool_name": "list_candidate_columns"}
    v1 = SimpleNamespace(tool_calls=calls, response={"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "list_candidate_columns", "arguments": '{"source_column":"x"}'}}]}}]})
    msgs = tool_loop.next_messages(v1, "openai_v1", ['["a"]'])
    assert msgs[0]["content"] == "" and msgs[1] == {"role": "tool", "tool_call_id": "call_1", "content": '["a"]'}


def test_tool_returns_are_a_function_of_the_arguments(data):
    t = tool_loop.Tables(data, "clark")
    col = t.src.columns[0]
    first = t.tool_result("list_candidate_columns", {"source_column": col})
    assert first == tool_loop.Tables(data, "clark").tool_result("list_candidate_columns", {"source_column": col})
    assert len(json.loads(first)) == tool_loop.SHORTLIST_K
    assert "error" in json.loads(t.tool_result("get_column_values", {"column": col, "side": "both"}))
    assert "error" in json.loads(t.tool_result("no_such_tool", {}))
    assert json.loads(t.tool_result("submit_match", {"target": "NONE"})) == {"recorded": True}


def test_tool_loop_condition_groups_replicates():
    from hbench.design import condition_id
    kw = dict(block="back_to_back", order=3, protocol="natural", transport="ollama", model="m", temperature=0,
              seed=1, max_steps=5)
    a = tool_loop.condition("clark", "c1", replicate=1, **kw)
    b = tool_loop.condition("clark", "c1", replicate=2, **{**kw, "order": 9})
    assert condition_id(a) == condition_id(b)
    assert condition_id(a) != condition_id(tool_loop.condition("clark", "c2", replicate=1, **kw))


def test_stage2_plan_orders():
    cols = [("s", "a"), ("s", "b")]
    plan = run_plan.make_stage2(cols, repeats=3)
    names = [s["name"] for s in plan["steps"]]
    assert len(names) == len(set(names))
    eps = [s for s in plan["steps"] if s["kind"] == "tool_loop"]
    assert len(eps) == 3 * len(cols) * 3
    inter = [(s["column"], s["replicate"]) for s in eps if s["block"] == "interleaved"]
    assert inter[:3] == [("a", 1), ("b", 1), ("a", 2)]
    b2b = [(s["column"], s["replicate"]) for s in eps if s["block"] == "back_to_back"]
    assert b2b[:3] == [("a", 1), ("a", 2), ("a", 3)]
    assert {s["protocol"] for s in eps if s["block"] == "unload_each"} == {"cold_start"}
    assert {s["protocol"] for s in eps if s["block"] != "unload_each"} == {"natural"}
    for s in plan["steps"]:
        if s["kind"] == "replay":
            assert s["of"] in names and names.index(s["of"]) < names.index(s["name"])
    # every block starts from an unload
    for block in ("unload_each", "back_to_back", "interleaved"):
        first = next(i for i, s in enumerate(plan["steps"]) if s.get("block") == block)
        assert plan["steps"][first - 1]["kind"] == "unload"


def test_execute_records_and_resumes(tmp_path):
    plan = {"plan_version": 1, "name": "t", "steps": [
        {"name": "hello", "kind": "cmd", "argv": ["{python}", "-c", "print('hi')"]},
        {"name": "fails", "kind": "cmd", "argv": ["{python}", "-c", "raise SystemExit(3)"]}]}
    root = tmp_path / "root"
    assert run_plan.execute(plan, root) == 1
    recs = [json.loads(x) for x in (root / "execution.jsonl").read_text().splitlines()]
    assert [(r["name"], r["status"], r.get("rc")) for r in recs] == [("hello", "ok", 0), ("fails", "failed", 3)]
    assert recs[0]["out"].strip() == "hi" and recs[0]["runs"] == []
    with pytest.raises(SystemExit):
        run_plan.execute(plan, root)  # same root without --resume
    assert run_plan.execute(plan, root, resume=True) == 1  # reruns only the failed step
    recs = [json.loads(x) for x in (root / "execution.jsonl").read_text().splitlines()]
    assert [r["name"] for r in recs] == ["hello", "fails", "fails"]
    with pytest.raises(SystemExit):
        run_plan.execute({**plan, "name": "other"}, root, resume=True)  # a different plan


def test_run_plan_refuses_a_root_inside_the_repo(tmp_path):
    plan_file = tmp_path / "p.json"
    plan_file.write_text(json.dumps({"steps": []}))
    with pytest.raises(SystemExit):
        run_plan.main(["run", str(plan_file), "--root", str(REPO_ROOT / "tmp_e1")])


def test_analysis_helpers():
    a = [["x", -0.1, [["x", -0.1], ["y", -2.0]]], ["p", -0.5, [["p", -0.5], ["q", -0.6]]], ["z", 0, [["z", 0]]]]
    b = [["x", -0.2, [["x", -0.2], ["y", -1.9]]], ["q", -0.5, [["q", -0.5], ["p", -0.55]]], ["z", 0, [["z", 0]]]]
    assert analyze_stage2.margin_of(a[0], "x") == pytest.approx(1.9)
    assert analyze_stage2.margin_of(b[1], "p") == pytest.approx(-0.05)
    # position 0 (1.9 -> 1.7), then the divergence at position 1 (0.1 -> -0.05); nothing after it
    assert analyze_stage2.shifts(a, b) == pytest.approx([-0.2, -0.15])
    assert analyze_stage2.lineage_class("B237 G16 B54") == "B+G"
    assert analyze_stage2.lineage_class("B505 S1") == "B+S"
    assert analyze_stage2.lineage_class(None) == "unknown"


def _run(rep, hashes, first="B300"):
    calls = [{"span_id": f"driver:{i}", "logprobs_sha256": h, "output": {"tool_calls": [{"name": "t"}], "text": ""}}
             for i, h in enumerate(hashes)]
    return {"cond": {"replicate": rep}, "calls": calls, "lineage": {"driver:0": {"canonical": first}}}


def test_compare_replicates_shapes():
    same = analyze_stage2.compare_replicates([_run(1, ["a", "b"]), _run(2, ["a", "b"]), _run(3, ["a", "b"])])
    assert same["shape"] == "all_bitwise_equal" and same["same_sequence"]
    p4 = analyze_stage2.compare_replicates([_run(1, ["a", "b"]), _run(2, ["c", "d"], "B299 S1"),
                                            _run(3, ["c", "d"], "B299 S1")])
    assert p4["shape"] == "first_differs_rest_equal" and p4["pairs_bitwise"] == 1
    assert p4["first_call_lineage"] == ["B", "B+S", "B+S"]
    other = analyze_stage2.compare_replicates([_run(1, ["a"]), _run(2, ["c"]), _run(3, ["e"])])
    assert other["shape"] == "other"
