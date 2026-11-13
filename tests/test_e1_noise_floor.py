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
        {"id": "call_x1vkyext", "type": "function",
         "function": {"name": "list_candidate_columns", "arguments": '{"source_column":"x"}'}}]}}]})
    msgs = tool_loop.next_messages(v1, "openai_v1", ['["a"]'], step=2)
    # the server's random id is replaced, so a repeated episode sends the same prompt (and gets the same seeds)
    assert msgs[0]["content"] == "" and msgs[0]["tool_calls"][0]["id"] == "call_2_0"
    assert msgs[1] == {"role": "tool", "tool_call_id": "call_2_0", "content": '["a"]'}
    native.response["message"]["tool_calls"][0]["id"] = "call_4q34kdk2"
    assert tool_loop.next_messages(native, "ollama", ["[]"], step=1)[0]["tool_calls"][0]["id"] == "call_1_0"


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


def _run(rep, hashes, first="B300", final="x", text=""):
    calls = [{"span_id": f"driver:{i}", "logprobs_sha256": h, "output": {"tool_calls": [{"name": "t"}], "text": text}}
             for i, h in enumerate(hashes)]
    return {"cond": {"replicate": rep}, "calls": calls, "lineage": {"driver:0": {"canonical": first}},
            "final": final}


def test_compare_replicates_shapes():
    same = analyze_stage2.compare_replicates([_run(1, ["a", "b"]), _run(2, ["a", "b"]), _run(3, ["a", "b"])])
    assert same["shape"] == "all_bitwise_equal" and same["same_sequence"]
    p4 = analyze_stage2.compare_replicates([_run(1, ["a", "b"]), _run(2, ["c", "d"], "B299 S1"),
                                            _run(3, ["c", "d"], "B299 S1")])
    assert p4["shape"] == "first_differs_rest_equal" and p4["pairs_bitwise"] == 1
    assert p4["first_call_lineage"] == ["B", "B+S", "B+S"]
    assert (p4["pairs_diverged"], p4["pairs_text_changed"], p4["pairs_final_changed"]) == (2, 0, 0)
    other = analyze_stage2.compare_replicates([_run(1, ["a"]), _run(2, ["c"], text="t"), _run(3, ["e"], final="y")])
    assert other["shape"] == "other" and not other["same_final"]
    assert (other["pairs_diverged"], other["pairs_text_changed"], other["pairs_final_changed"]) == (3, 2, 2)


# ------------------------------------------------------------------ Stage 3

import analyze_stage3  # noqa: E402
import pandas as pd  # noqa: E402
import resample  # noqa: E402


def test_stage3_plan_shape():
    plan = run_plan.make_stage3([("clark", "Age"), ("krug", "PAM50")], "/stage1", chat_seeds=3, v1_seeds=2,
                                p6_seeds=2, agent_seeds=2)
    steps = plan["steps"]
    names = [s["name"] for s in steps]
    assert len(names) == len(set(names))
    assert names[0] == "engine_check_before" and names[-3:] == ["engine_check_after", "validate", "lineage_check"]
    refs = [s for s in steps if s.get("block") == "v1_ref"]
    assert all(s["transport"] == "openai_v1" and s["temperature"] == 0 and s["protocol"] == "cold_start" for s in refs)
    cal = [s for s in steps if s["kind"] == "resample" and s["block"] == "chat_cal"]
    assert [s["seed"] for s in cal] == [1, 2, 3]
    assert all(s["protocol"] == "cold_start" and s["subset"] == "all" and s["overrides"] == {"temperature": 0.8}
               for s in cal)
    v1 = [s for s in steps if s["kind"] == "resample" and s["block"].startswith("v1_")]
    assert {s["overrides"]["temperature"] for s in v1} == {1.0}
    # resend passes come after an unload and use the P6 subset; repeats are whole passes with the same seeds
    i = names.index("chat:resend:unload")
    assert steps[i + 1]["protocol"] == "resend" and steps[i + 1]["subset"] == "p6"
    assert [s["seed"] for s in steps if s.get("block") == "chat_repeat"] == [1, 2]
    sampled = [s for s in steps if s.get("block") == "v1_sampled"]
    assert len(sampled) == 4 and all(s["seed"] == s["replicate"] and s["temperature"] == 1.0 for s in sampled)
    assert all(s["seed"] == 1 for s in steps if s.get("block") == "v1_sampled_repeat")
    # selections are written before any pass reads them
    assert names.index("select:v1") < names.index("v1:cal:s1") and names.index("select:chat") < names.index("chat:cal:s1")


def test_path_kept():
    assert resample.path_kept(True, None, 5) is True
    assert resample.path_kept(False, 6, 5) is True    # diverged after the value
    assert resample.path_kept(False, 5, 5) is False   # diverged at the value's last token
    assert resample.path_kept(False, 0, 5) is False
    assert resample.path_kept(None, None, 5) is None and resample.path_kept(True, None, None) is None


def test_selection_order_is_fixed():
    items = [{"span_id": f"driver:{i}", "run": f"/r/{i % 3}", "call_index": i} for i in range(12)]
    assert resample.hash_order(items) == resample.hash_order(list(reversed(items)))
    sel = resample.finish_selection(items[::-1], items[:3], endpoint="x")
    assert [(c["run"], c["call_index"]) for c in sel["sources"]] == sorted((c["run"], c["call_index"]) for c in items)
    assert {c["span_id"] for c in sel["p6"]} == {"driver:0", "driver:1", "driver:2"}
    assert sel["sha"] == resample.finish_selection(items, items[:3], endpoint="x")["sha"]


def test_calibration_stats():
    rows = []
    for i in range(10):  # p = 1: never changes
        rows += [{"source_span_id": f"a{i}", "decision_key": "target", "p_path": 1.0, "p_exact": True,
                  "path_kept": True, "value_kept": True}]
    for i in range(20):  # p = 0.5: half the samples change their path, a quarter their value
        rows += [{"source_span_id": f"b{i // 4}", "decision_key": "target", "p_path": 0.5, "p_exact": True,
                  "path_kept": i % 2 == 0, "value_kept": i % 4 != 1}]
    c = analyze_stage3.calibration(pd.DataFrame(rows))
    assert c["n_samples"] == 30 and c["n_decisions"] == 15
    assert c["expected_changes"] == 10 and c["observed_changes"] == 10 and c["z"] == 0
    assert c["observed_value_changes"] == 5 and c["path_changed_value_kept"] == 5
    assert [b["n"] for b in c["bins"]] == [10, 20] and c["bins"][0]["observed"] == 0
    assert 0 < c["skill"] <= 1


def test_wilson_and_key_class():
    lo, hi = analyze_stage3.wilson(0, 150)
    assert lo == 0 and 0 < hi < 0.03
    assert analyze_stage3.wilson(5, 10)[0] < 0.5 < analyze_stage3.wilson(5, 10)[1]
    assert analyze_stage3.key_class("target@1") == "target" and analyze_stage3.key_class(None) == "none"


def test_value_local_forecast_ignores_the_free_prefix():
    # a near-uniform reasoning token before the value, then a confident value token
    tokens = [["To", -0.7, [["To", -0.7], ["Let", -0.8]]], ["x", -0.01, [["x", -0.01], ["y", -4.6]]]]
    params = {"temp": 1.0, "top_k": 40, "top_p": 1.0, "min_p": 0.0}
    m = {"span": [1], "end": 1, "constrained_positions": []}
    pv = resample.p_value_local(tokens, m, params)
    pp = resample.sampler.p_path(tokens, 1, params)["p"]
    assert pv > 0.98 and pp < 0.5
    d = pd.DataFrame([{"source_span_id": "a", "decision_key": "tool", "p_path": pp, "p_exact": False,
                       "p_value_local": pv, "path_kept": False, "value_kept": True}])
    c = analyze_stage3.calibration(d, "p_value_local", "value_kept")
    assert c["forecast"] == "p_value_local" and c["observed_changes"] == 0


def test_call_seeds_are_distinct_and_reproducible():
    a = resample.call_seed(1, "run/driver:2")
    assert a == resample.call_seed(1, "run/driver:2")
    others = {resample.call_seed(1, "run/driver:3"), resample.call_seed(2, "run/driver:2"),
              resample.call_seed(1, "other/driver:2")}
    assert a not in others and len(others) == 3
    assert all(0 <= x < 2 ** 31 for x in others | {a})


def test_sampled_episodes_leave_seed_derivation_to_hbench():
    # tool_loop states the replicate's seed as a base; LLMClient derives each sampled call's seed (hbench.seeds).
    plan = run_plan.make_stage3([("clark", "Age")], "/stage1", chat_seeds=1, v1_seeds=1, p6_seeds=1, agent_seeds=1)
    sampled = [s for s in plan["steps"] if str(s.get("block", "")).startswith("v1_sampled")]
    assert sampled and all(s["temperature"] > 0 and isinstance(s["seed"], int) for s in sampled)
    assert not any("per_call_seed" in s for s in plan["steps"])
    cond = tool_loop.condition("clark", "Age", replicate=1, block=None, order=None, protocol="cold_start",
                               transport="openai_v1", model="m", temperature=1.0, seed=3, max_steps=5)
    assert cond["decoding"].get("seed_mode") in (None, "per_call") and cond["decoding"]["seed"] == 3


def test_target_answers_ignore_the_call_index():
    spec = tool_loop.DECISION
    src = {"decision_spec": spec, "caller": tool_loop.CALLER}
    rep = {"output": {"text": "", "tool_calls": [
        {"name": "get_column_values", "arguments": {"column": "x", "side": "target"}},
        {"name": "submit_match", "arguments": {"source_column": "A", "target": "bmi"}}]}}
    assert resample.tool_spec(src) and not resample.tool_spec({"decision_spec": None})
    assert resample.submitted(rep, src) == "bmi"
    look = {"output": {"text": "", "tool_calls": [{"name": "list_candidate_columns",
                                                   "arguments": {"source_column": "A"}}]}}
    assert resample.submitted(look, src) is None


def test_long_outputs_are_read_back_from_the_blob_store(tmp_path):
    from hbench.runlog.writer import RunLog, read_events

    out = {"text": "x" * 5000, "thinking": None, "tool_calls": None}
    (tmp_path / "run").mkdir()
    with RunLog(tmp_path / "run") as log:
        log.emit("llm_call", output=out)
        log.emit("llm_call", output={"text": "short"})
    raw = [e for e in read_events(tmp_path / "run") if e["type"] == "llm_call"]
    assert "blob" in raw[0]["output"]  # the writer stored it as a reference
    calls = [e for e in resample.read_run_events(tmp_path / "run") if e["type"] == "llm_call"]
    assert [c["output"] for c in calls] == [out, {"text": "short"}]


SHOW_27B = {"parameters": "num_ctx                        32768\ntemperature                    1\ntop_k                          20\ntop_p                          0.95"}


def test_default_temperatures_follow_the_model():
    assert run_plan.default_temperatures("qwen2.5:14b", show=lambda m: {"parameters": None}) == \
        {"ollama": 0.8, "openai_v1": 1.0}
    # 27b sets temperature 1, so /api/chat runs it at 1.0; /v1 always sends 1.0
    assert run_plan.default_temperatures("qwen3.8-27b-bdikit", show=lambda m: SHOW_27B) == \
        {"ollama": 1.0, "openai_v1": 1.0}
    with pytest.raises(RuntimeError):
        run_plan.default_temperatures("missing", show=lambda m: {"error": "not found"})


def test_stage3_plan_for_a_thinking_model():
    plan = run_plan.make_stage3([("clark", "Age")], "/pilot/single", chat_seeds=2, v1_seeds=2, p6_seeds=2,
                                agent_seeds=2, model="qwen3.8-27b-bdikit", think=True, chat_by="value", n_p6=8,
                                repeat_subset="p6", temperatures={"ollama": 1.0, "openai_v1": 1.0})
    steps = plan["steps"]
    eps = [s for s in steps if s["kind"] == "tool_loop"]
    assert eps and all(s["model"] == "qwen3.8-27b-bdikit" and s["think"] is True for s in eps)
    sel = {s["what"]: s for s in steps if s["kind"] == "select"}
    assert sel["chat"]["options"] == {"by": "value", "n_p6": 8} and sel["v1"]["options"] == {"n_p6": 8}
    assert {s["subset"] for s in steps if s.get("block") in ("chat_repeat", "v1_repeat")} == {"p6"}
    assert {s["overrides"]["temperature"] for s in steps if s.get("block") == "chat_cal"} == {1.0}
    assert plan["temperatures"] == {"ollama": 1.0, "openai_v1": 1.0} and plan["think"] is True
    # the 14b plan is unchanged: no model/think on episodes, no selection options
    plan14 = run_plan.make_stage3([("clark", "Age")], "/stage1", chat_seeds=1, v1_seeds=1, p6_seeds=1, agent_seeds=1)
    assert not any("model" in s or "think" in s for s in plan14["steps"] if s["kind"] == "tool_loop")
    assert not any("options" in s for s in plan14["steps"] if s["kind"] == "select")


def test_model_sampler_comes_from_the_call_or_the_run_header(tmp_path, monkeypatch):
    assert resample.model_sampler_of({"model_sampler": {"top_k": 20.0}}) == {"top_k": 20.0}
    assert resample.model_sampler_of({"model": "m"}) is None  # no call field, no run: unknown
    monkeypatch.setitem(resample._RUNS, str(tmp_path.resolve()),
                        {"samplers": {"qwen3.8-27b-bdikit": {"temp": 1.0, "top_k": 20.0, "top_p": 0.95}}})
    got = resample.model_sampler_of({"model": "qwen3.8-27b-bdikit"}, tmp_path)
    assert got == {"temp": 1.0, "top_k": 20.0, "top_p": 0.95}
    assert resample.model_sampler_of({"model": "qwen2.5:14b"}, tmp_path) is None


def test_stage1_runs_finds_both_layouts(tmp_path):
    for d in ("1a/raw/run_a", "1a/run_b", "1a/_blobs/sha256"):
        (tmp_path / d).mkdir(parents=True)
    for d in ("1a/raw/run_a", "1a/run_b"):
        (tmp_path / d / "events.driver.jsonl").write_text("")
    assert [p.name for p in resample.stage1_runs(tmp_path)] == ["run_a", "run_b"]
