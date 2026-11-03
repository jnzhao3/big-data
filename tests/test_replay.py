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
    row = compare(events[0], events[1], log.blobs)
    assert row["same_text"] and row["same_tokens"] and not row["bitwise_same_logprobs"]
    assert row["decision"] == "bcr_id" and row["decision_flipped"] is False
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
    row = compare(events[0], events[1], log.blobs)
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
        row = compare(calls[0], ev, blobs)
        assert row["bitwise_same_logprobs"]  # mock server returns the same payload
        assert row["decision"] == "gender" and row["decision_flipped"] is False and row["d_margin"] == 0.0
    finally:
        srv.shutdown()
