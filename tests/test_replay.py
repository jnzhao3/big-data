import json
import threading
from http.server import HTTPServer
from pathlib import Path

from hbench.decisions import json_field_parser, register_decision_parser
from hbench.llm import LLMClient
from hbench.replay import compare, compare_tokens, load_calls, replay_call
from hbench.runlog import RunLog, read_events
from conftest import MockOllama as _Server

PILOT = json.loads((Path(__file__).parent / "fixtures" / "pilot_case_id_logprobs.json").read_text())["paths"]


def test_compare_tokens():
    a = [["x", -0.1, [["x", -0.1], ["y", -2.0]]], ["z", -0.2, [["z", -0.2]]]]
    b = [["x", -0.11, [["x", -0.11], ["y", -1.9]]], ["z", -0.2, [["z", -0.2]]]]
    c = [["x", -0.1, [["x", -0.1]]], ["q", -0.3, [["q", -0.3]]]]
    r = compare_tokens(a, b)
    assert r["same_tokens"] and not r["bitwise_same_logprobs"]
    assert abs(r["max_abs_dlogprob_chosen"] - 0.01) < 1e-9 and abs(r["max_abs_dlogprob_any"] - 0.1) < 1e-9
    assert compare_tokens(a, c)["first_divergent_token"] == 1 and compare_tokens(a, a)["bitwise_same_logprobs"]
    assert compare_tokens(None, a) == {"comparable": False}


def test_compare_reports_decision_margin_shift(tmp_path):
    register_decision_parser("method:pilot", json_field_parser("target"))
    log = RunLog.create(tmp_path, "e1", "t", "clark", run_id="r")
    events = []
    for name in ("cold", "repeat_1tok_suffix"):
        rec = PILOT[name]
        toks = [[t, next(lp for c, lp in top if c == t), top] for t, top in zip(rec["tokens"], rec["top"])]
        span = log.emit("llm_call", caller="method:pilot", output={"text": rec["content"]},
                        token_logprobs=log.blobs.put(toks), engine={"path_class": "x"})
        events.append(next(e for e in read_events(tmp_path / "r") if e["span_id"] == span))
    (row,) = compare(events[0], events[1], log.blobs)
    assert row["same_text"] and row["same_tokens"] and not row["bitwise_same_logprobs"]
    assert row["decision"] == "bcr_id" and row["decision_flipped"] is False and round(row["margin_source"], 4) == 0.1266
    assert round(row["margin_source"], 4) == 0.1266 and round(row["d_margin"], 4) == -0.0070


def test_decision_spec_in_the_log_needs_no_registered_parser(tmp_path):
    log = RunLog.create(tmp_path, "e1", "t", "clark", run_id="r")
    events = []
    for name in ("cold", "repeat_1tok_suffix"):
        rec = PILOT[name]
        toks = [[t, next(lp for c, lp in top if c == t), top] for t, top in zip(rec["tokens"], rec["top"])]
        span = log.emit("llm_call", caller="method:never_registered", output={"text": rec["content"]},
                        decision_spec={"type": "json_field", "field": "target"},
                        token_logprobs=log.blobs.put(toks), engine={"path_class": "x"})
        events.append(next(e for e in read_events(tmp_path / "r") if e["span_id"] == span))
    (row,) = compare(events[0], events[1], log.blobs)
    assert row["decision"] == "bcr_id" and round(row["d_margin"], 4) == -0.0070


def test_replay_resends_the_exact_logged_request(tmp_path):
    _Server.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        src = RunLog.create(tmp_path / "raw", "e0", "t", "clark", run_id="src")
        LLMClient(src, "qwen2.5:14b", base=base, decoding={"temperature": 0, "seed": 1}).chat(
            [{"role": "user", "content": "q"}], caller="method:test", format="json",
            decision={"type": "json_field", "field": "target"})
        src.close()
        calls, blobs = load_calls(tmp_path / "raw" / "src")
        assert calls[0]["decision_spec"] == {"type": "json_field", "field": "target"}
        sent_original = _Server.seen[-1][1]
        rep = RunLog.create(tmp_path / "raw", "replay", "t", "clark", run_id="rep")
        client = LLMClient(rep, "placeholder", base=base, top_logprobs=None)
        n0 = len(_Server.seen)
        res = replay_call(calls[0], blobs, client, protocol="resend")
        assert len(_Server.seen) - n0 == 2 and _Server.seen[-1][1] == sent_original  # pre-send, then the measured call
        replay_call(calls[0], blobs, client, protocol="after_unrelated")
        assert "rivers" in _Server.seen[-2][1]["messages"][0]["content"]
        rep.close()
        ev = next(e for e in read_events(tmp_path / "raw" / "rep") if e["span_id"] == res.span_id)
        assert ev["replay_of"] == calls[0]["span_id"] and ev["protocol"] == "resend" and ev["caller"] == "replay:resend"
        (row,) = compare(calls[0], ev, blobs)
        assert row["bitwise_same_logprobs"]  # mock server returns the same payload
        assert row["decision"] == "gender" and row["decision_flipped"] is False and row["d_margin"] == 0.0
    finally:
        srv.shutdown()


TOOL = json.loads((Path(__file__).parent / "fixtures" / "probe3_tool_call.json").read_text())


def test_compare_reports_every_decision(tmp_path):
    from hbench.llm.normalize import normalize_native
    n = normalize_native(TOOL["ollama"]["request"], TOOL["ollama"]["response"])
    log = RunLog.create(tmp_path, "e1", "t", "clark", run_id="r")
    spec = [{"type": "tool_call", "field": "name"}, {"type": "tool_call", "field": "arguments.source_column"}]
    spans = [log.emit("llm_call", caller="probe:tool_loop", output=n["output"], decision_spec=spec,
                      token_logprobs=log.blobs.put(n["token_logprobs"]), engine={"path_class": "x"}) for _ in range(2)]
    ev = {e["span_id"]: e for e in read_events(tmp_path / "r")}
    rows = compare(ev[spans[0]], ev[spans[1]], log.blobs)
    assert [r["decision_key"] for r in rows] == ["name", "arguments.source_column"]
    assert [r["decision"] for r in rows] == ["list_target_columns", "Tumor_Size_cm"]
    assert not any(r["decision_flipped"] for r in rows) and all(r["bitwise_same_logprobs"] for r in rows)
    # the prefix margin of both decisions is the first token: <tool_call> (call a tool at all) vs a text token
    assert [round(r["margin_source"], 2) for r in rows] == [5.71, 5.71]


def test_compare_treats_sampled_calls_as_not_greedy(tmp_path):
    log = RunLog.create(tmp_path, "e1", "t", "clark", run_id="r")
    toks = [["{\"target\": \"", -0.01, [["{\"target\": \"", -0.01], ["x", -9.0]]],
            ["b", -0.9, [["a", -0.6], ["b", -0.9], ["c", -3.0]]], ["\"}", -0.01, [["\"}", -0.01], ["x", -9.0]]]]
    spans = [log.emit("llm_call", caller="m", output={"text": '{"target": "b"}'}, decoding={"temperature": t},
                      decision_spec={"type": "json_field", "field": "target"}, token_logprobs=log.blobs.put(toks),
                      engine={"path_class": "x"}) for t in (0.8, 0)]
    ev = {e["span_id"]: e for e in read_events(tmp_path / "r")}
    (hot,) = compare(ev[spans[0]], ev[spans[0]], log.blobs)
    (cold,) = compare(ev[spans[1]], ev[spans[1]], log.blobs)
    assert round(hot["margin_source"], 2) == -0.30   # sampled below the top candidate: a negative gap, no constraint
    assert round(cold["margin_source"], 2) == 2.10   # at T=0 the top candidate must have been ruled out


def test_compare_says_whether_the_lineages_match(tmp_path):
    log = RunLog.create(tmp_path, "e1", "t", "clark", run_id="r")
    span = log.emit("llm_call", caller="m", output={"text": "x"}, token_logprobs=log.blobs.put([["x", -0.1, []]]))
    ev = next(e for e in read_events(tmp_path / "r") if e["span_id"] == span)
    a, b = {"fingerprint": "f1", "canonical": "B10", "uncertain": []}, {"fingerprint": "f2", "canonical": "B9 S1", "uncertain": []}
    assert compare(ev, ev, log.blobs, a, dict(a))[0]["lineage_same"] is True
    (row,) = compare(ev, ev, log.blobs, a, b)
    assert row["lineage_same"] is False and (row["lineage_source"], row["lineage_replay"]) == ("B10", "B9 S1")
    assert compare(ev, ev, log.blobs, a, {**b, "uncertain": ["unknown start"]})[0]["lineage_same"] is None
    assert compare(ev, ev, log.blobs)[0]["lineage_same"] is None  # runs audited before lineage existed


def test_compare_keeps_two_tool_calls_apart(tmp_path):
    log = RunLog.create(tmp_path, "e1", "t", "clark", run_id="r")
    spec = [{"type": "tool_call", "field": "name"}, {"type": "tool_call", "field": "name", "index": 1}]
    outs = [{"text": "", "tool_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]},
            {"text": "", "tool_calls": [{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}]}]
    spans = [log.emit("llm_call", caller="m", output=o, decision_spec=spec, engine={}) for o in outs]
    ev = {e["span_id"]: e for e in read_events(tmp_path / "r")}
    rows = compare(ev[spans[0]], ev[spans[1]], log.blobs)
    assert [(r["decision_key"], r["decision"], r["decision_replay"], r["decision_flipped"]) for r in rows] == [
        ("name", "a", "b", True), ("name@1", "b", "a", True)]


def test_long_outputs_stay_inline_and_old_blob_outputs_are_resolved(tmp_path):
    """A thinking answer passes INLINE_LIMIT. New calls keep output inline; a call logged with its output as a blob
    ref (before) still yields its decision rows and evidence."""
    from hbench.decisions import evidence_from_call
    from hbench.llm.record import load_output, record_llm_call
    from hbench.runlog.writer import INLINE_LIMIT, is_blob_ref

    rec = PILOT["cold"]
    toks = [[t, next(lp for c, lp in top if c == t), top] for t, top in zip(rec["tokens"], rec["top"])]
    long_text = json.dumps({"target": "bcr_id", "reason": "x" * (INLINE_LIMIT + 100)})
    log = RunLog.create(tmp_path, "e1", "t", "clark", run_id="r")
    spec = {"type": "json_field", "field": "target"}
    # new: record_llm_call keeps the long output inline
    span = record_llm_call(log, caller="method:x", transport="ollama", endpoint="/api/chat", api_base=None,
                           model="m", request={"model": "m", "messages": [{"role": "user", "content": "q"}]},
                           response={}, normalized={"output": {"text": long_text}, "decoding": {}},
                           latency_s=0.1, decision_spec=spec)
    new = next(e for e in read_events(tmp_path / "r") if e["span_id"] == span)
    assert not is_blob_ref(new["output"]) and new["output"]["text"] == long_text
    # old: the output stored as a blob ref
    old = []
    for _ in range(2):
        s = log.emit("llm_call", caller="method:x", output={"text": long_text}, decision_spec=spec,
                     token_logprobs=log.blobs.put(toks), engine={"path_class": "x"})
        old.append(next(e for e in read_events(tmp_path / "r") if e["span_id"] == s))
    assert is_blob_ref(old[0]["output"]) and load_output(old[0], log.blobs)["text"] == long_text
    (row,) = compare(old[0], old[1], log.blobs)
    assert row["decision"] == "bcr_id" and row["decision_flipped"] is False and round(row["margin_source"], 4) == 0.1266
    ev = evidence_from_call(old[0], log.blobs)
    assert ev["margin_found"] and round(ev["margin"], 4) == 0.1266
