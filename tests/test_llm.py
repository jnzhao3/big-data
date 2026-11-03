import json
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest

from hbench.llm import LLMClient, path_class, rederive
from hbench.llm.normalize import normalize_native, normalize_openai
from hbench.runlog import RunLog, read_events
from conftest import NATIVE, V1, MockOllama as _Server



@pytest.mark.parametrize("prompt,cached,load,want", [
    (2741, 0, 3900.0, "cold_full"), (2741, 0, 2.0, "full"), (2741, 2740, 2.0, "single_token"),
    (2741, 2623, None, "partial"), (2741, None, None, "unknown"), (None, 0, None, "unknown")])
def test_path_class(prompt, cached, load, want):
    assert path_class(prompt, cached, load) == want


def test_normalize_native_real_response():
    n = normalize_native(NATIVE["request"], NATIVE["response"])
    assert n["output"]["text"] == '{"target": "gender"}'
    assert n["decoding"]["temperature"] == 0 and n["decoding"]["seed"] == 1 and n["decoding"]["num_ctx"] == 32768
    assert n["engine"]["cached_tokens"] == NATIVE["response"]["prompt_eval_cached_count"]
    assert n["engine"]["path_class"] in ("full", "cold_full")
    assert "".join(t[0] for t in n["token_logprobs"]) == n["output"]["text"]
    assert all(len(t[2]) == 5 for t in n["token_logprobs"]) and "token_logprobs" not in n["unavailable"]


def test_normalize_openai_real_response():
    n = normalize_openai(V1["request"], V1["response"])
    assert n["output"]["text"] == '{"target": "gender"}' and n["decoding"]["format"] == "json"
    assert (n["engine"]["prompt_tokens"], n["engine"]["cached_tokens"]) == (58, 57)
    assert n["engine"]["path_class"] == "single_token"
    assert "".join(t[0] for t in n["token_logprobs"]) == n["output"]["text"]
    assert "num_ctx" in n["unavailable"] and "load_ms" in n["unavailable"]


def test_normalize_reports_what_is_missing():
    req = dict(NATIVE["request"], logprobs=False)
    resp = {k: v for k, v in NATIVE["response"].items() if k not in ("logprobs", "prompt_eval_cached_count")}
    n = normalize_native(req, resp)
    assert n["token_logprobs"] is None and n["unavailable"]["token_logprobs"] == "not requested"
    assert "cached_tokens" in n["unavailable"] and n["engine"]["path_class"] == "unknown"


@pytest.mark.parametrize("transport,path", [("ollama", "/api/chat"), ("openai_v1", "/v1/chat/completions")])
def test_client_builds_requests_and_logs_lossless_calls(tmp_path, server, transport, path):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    c = LLMClient(log, "qwen2.5:14b", transport=transport, base=server, decoding={"temperature": 0, "seed": 1})
    res = c.chat([{"role": "user", "content": "q"}], caller="method:test", step=3, format="json")
    sent_path, sent = _Server.seen[-1]
    assert sent_path == path and sent["logprobs"] is True and sent["top_logprobs"] == 5
    if transport == "ollama":
        assert sent["options"] == {"temperature": 0, "seed": 1, "num_ctx": 32768} and sent["format"] == "json"
    else:
        assert sent["temperature"] == 0 and sent["response_format"] == {"type": "json_object"} and "options" not in sent
    assert res.text == '{"target": "gender"}' and res.token_logprobs and res.span_id == "driver:1"
    log.close()
    ev = next(e for e in read_events(tmp_path / "r") if e["type"] == "llm_call")
    assert ev["llm_schema"] == "hbench.llm_call/3" and ev["transport"] == transport and ev["step"] == 3
    assert log.blobs.get(ev["request_raw"]) == sent                       # exact bytes sent are recoverable
    assert log.blobs.get(ev["response_raw"]) == (NATIVE if transport == "ollama" else V1)["response"]
    assert log.blobs.get(ev["token_logprobs"]) == res.token_logprobs
    assert ev["logprobs_sha256"] and ev["input_sha256"] and ev["output_sha256"]
    rd = rederive(ev, log.blobs)  # normalization can be redone from the raw payloads alone
    assert rd["rederived"] and rd["engine"] == ev["engine"] and rd["output"] == ev["output"]


def test_input_hash_ignores_logprob_settings(tmp_path, server):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    LLMClient(log, "m", base=server, top_logprobs=5).chat([{"role": "user", "content": "q"}], caller="x")
    LLMClient(log, "m", base=server, top_logprobs=None).chat([{"role": "user", "content": "q"}], caller="x")
    log.close()
    a, b = [e for e in read_events(tmp_path / "r") if e["type"] == "llm_call"]
    assert a["input_sha256"] == b["input_sha256"] and a["request_sha256"] != b["request_sha256"]


def test_failed_call_is_logged_then_raised(tmp_path):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    with pytest.raises(Exception):
        LLMClient(log, "m", base="http://127.0.0.1:9", timeout=2).chat([{"role": "user", "content": "q"}], caller="x")
    log.close()
    ev = next(e for e in read_events(tmp_path / "r") if e["type"] == "llm_call")
    assert ev["error"] and ev["response_raw"] is None and ev["unavailable"]["response"] == "request failed"


class _LiteLLMServer(_Server):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        _Server.seen.append((self.path, body))
        if self.path.endswith("/chat/completions"):
            out = V1["response"]
        elif self.path.endswith("/api/chat"):
            out = {k: v for k, v in NATIVE["response"].items() if k != "logprobs"}
        else:
            out = {}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_litellm_callback_writes_the_same_schema(tmp_path):
    litellm = pytest.importorskip("litellm")
    from hbench.runlog import litellm_cb

    srv = HTTPServer(("127.0.0.1", 0), _LiteLLMServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    cb = litellm_cb.install(log, caller="method:test")
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": "calling", "tool_calls": [
                   {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "[]"}]
    try:
        with litellm_cb.parent_span("driver:7", step=3):
            litellm.completion(model="openai/qwen2.5:14b", messages=history, api_base=base + "/v1", api_key="x",
                               temperature=0, logprobs=True, top_logprobs=5)
            litellm.completion(model="ollama_chat/qwen2.5:14b", messages=history, api_base=base, temperature=0)
        assert cb.wait(10)
    finally:
        litellm_cb.uninstall(cb)
        srv.shutdown()
    calls = {e["transport"]: e for e in read_events(tmp_path / "r") if e["type"] == "llm_call"}
    o, oc = calls["litellm/openai"], calls["litellm/ollama_chat"]
    assert o["llm_schema"] == "hbench.llm_call/3" and o["parent_id"] == "driver:7" and o["step"] == 3
    assert o["prompt_sha256"] and o["prompt_sha256"] != o["input_sha256"]
    assert o["output"]["text"] == '{"target": "gender"}' and o["engine"]["cached_tokens"] == 57
    assert log.blobs.get(o["token_logprobs"]) and o["cloud_call"] is False
    sent = log.blobs.get(o["request_raw"])  # what actually went out: the openai route keeps the tool call
    assert "tool_calls" in [m for m in sent["messages"] if m["role"] == "assistant"][0]
    sent_oc = log.blobs.get(oc["request_raw"])  # ollama_chat drops it, and cannot give logprobs or cache counts
    assert "tool_calls" not in [m for m in sent_oc["messages"] if m["role"] == "assistant"][0]
    assert oc["token_logprobs"] is None and "token_logprobs" in oc["unavailable"]


def test_litellm_callback_waits_until_the_event_is_written(tmp_path):
    pytest.importorskip("litellm")
    from hbench.runlog import litellm_cb

    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    cb = litellm_cb.HBenchLiteLLMLogger(log, caller="method:test")
    kwargs = {"litellm_call_id": "c1", "model": "qwen2.5:14b",
              "litellm_params": {"custom_llm_provider": "openai", "api_base": "http://localhost:11434/v1"},
              "additional_args": {"complete_input_dict": V1["request"]}}
    cb.log_pre_api_call("qwen2.5:14b", V1["request"]["messages"], kwargs)
    gate, write = threading.Event(), cb._write

    def slow_write(*a):
        gate.wait(5)
        write(*a)

    cb._write = slow_write
    t = threading.Thread(target=cb.log_success_event, args=(kwargs, V1["response"], None, None))
    t.start()
    assert cb.wait(0.3) is False  # reported but not yet written: wait() must not return
    gate.set()
    t.join(5)
    cb.log_success_event(kwargs, V1["response"], None, None)  # LiteLLM may report one call twice
    assert cb.wait(1)
    calls = [e for e in read_events(tmp_path / "r") if e["type"] == "llm_call"]
    assert len(calls) == 1 and calls[0]["litellm_call_id"] == "c1" and calls[0]["engine"]["cached_tokens"] == 57


def test_is_local():
    pytest.importorskip("litellm")
    from hbench.runlog.litellm_cb import is_local
    assert is_local("http://127.0.0.1:11434/v1", "openai") and is_local(None, "ollama_chat")
    assert not is_local(None, "openai") and not is_local("https://api.openai.com/v1", "openai")


TOOL = json.loads((Path(__file__).parent / "fixtures" / "probe3_tool_call.json").read_text())


def test_tool_call_arguments_are_parsed_on_both_transports():
    a = normalize_native(TOOL["ollama"]["request"], TOOL["ollama"]["response"])["output"]["tool_calls"]
    b = normalize_openai(TOOL["openai_v1"]["request"], TOOL["openai_v1"]["response"])["output"]["tool_calls"]
    assert a == b == [{"name": "list_target_columns", "arguments": {"source_column": "Tumor_Size_cm"}}]
    bad = normalize_openai({}, {"choices": [{"message": {"tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}]}}]})
    assert bad["output"]["tool_calls"] == [{"name": "f", "arguments": "{oops", "arguments_invalid": True}]


def test_same_tool_call_hashes_the_same_on_both_transports(tmp_path):
    from hbench.llm.normalize import NORMALIZERS
    from hbench.llm.record import record_llm_call
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    for t in ("ollama", "openai_v1"):
        fx = TOOL[t]
        record_llm_call(log, caller="x", transport=t, endpoint=None, api_base="http://localhost:11434",
                        model="qwen2.5:14b", request=fx["request"], response=fx["response"],
                        normalized=NORMALIZERS[t](fx["request"], fx["response"]), latency_s=0.1)
    log.close()
    a, b = [e for e in read_events(tmp_path / "r") if e["type"] == "llm_call"]
    assert a["output_sha256"] == b["output_sha256"]
    assert TOOL["ollama"]["logged_output_sha256"] != TOOL["openai_v1"]["logged_output_sha256"]  # v2 hashed them apart


def test_prompt_sha256_ignores_decoding(tmp_path, server):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    c = LLMClient(log, "m", base=server)
    msgs = [{"role": "user", "content": "q"}]
    r0 = c.chat(msgs, caller="x", temperature=0)
    r1 = c.chat(msgs, caller="x", temperature=0.8, seed=3)
    r2 = c.chat(msgs, caller="x", temperature=0, format="json")
    log.close()
    a, b, f = [e for e in read_events(tmp_path / "r") if e["type"] == "llm_call"]
    assert a["llm_schema"] == "hbench.llm_call/3"
    assert a["prompt_sha256"] == b["prompt_sha256"] == r0.prompt_sha256 == r1.prompt_sha256
    assert a["input_sha256"] != b["input_sha256"]          # the generation key still includes decoding
    assert f["prompt_sha256"] != a["prompt_sha256"] == r0.prompt_sha256 != r2.prompt_sha256  # format changes the prompt


def test_prompt_sha256_includes_think(tmp_path, server):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    c = LLMClient(log, "m", base=server)
    a = c.chat([{"role": "user", "content": "q"}], caller="x", temperature=0, think=True)
    b = c.chat([{"role": "user", "content": "q"}], caller="x", temperature=0, think=False)
    log.close()
    assert a.prompt_sha256 != b.prompt_sha256  # thinking changes what the model is shown


def _effective(decoding, endpoint="/api/chat"):
    from hbench.sampler import effective
    return effective(decoding, endpoint)


def test_result_evidence_is_the_explicit_call_in_one_line(tmp_path, server):
    from hbench.decisions import evidence
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    res = LLMClient(log, "m", base=server, decoding={"temperature": 0, "seed": 1}).chat(
        [{"role": "user", "content": "q"}], caller="x", format="json", decision={"type": "json_field", "field": "target"})
    log.close()
    explicit = evidence(res.span_id, res.token_logprobs, res.text, "gender", json_key="target", temperature=0,
                        sampler_params=_effective(res.normalized["decoding"]), grammar=True,
                        prompt_sha256=res.prompt_sha256)
    assert res.evidence() == explicit and res.evidence()["margin_found"] is True
    assert res.evidence(source="test")["source"] == "test"  # extra fields are kept, as with evidence()


def test_evidence_from_a_logged_call_matches_the_result(tmp_path, server):
    from hbench.decisions import evidence_from_call
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    res = LLMClient(log, "m", base=server, decoding={"temperature": 0}).chat(
        [{"role": "user", "content": "q"}], caller="x", format="json", decision={"type": "json_field", "field": "target"})
    log.close()
    event = next(e for e in read_events(tmp_path / "r") if e["type"] == "llm_call")
    assert evidence_from_call(event, log.blobs) == res.evidence()  # e.g. for calls made through LiteLLM


def test_result_evidence_picks_a_decision_by_key_and_reports_a_missing_one():
    from hbench.decisions import call_evidence
    from hbench.llm.normalize import normalize_native
    n = normalize_native(TOOL["ollama"]["request"], TOOL["ollama"]["response"])
    spec = [{"type": "tool_call", "field": "name"}, {"type": "tool_call", "field": "arguments.source_column"}]
    first = call_evidence(n, spec, transport="ollama", span_id="s")
    arg = call_evidence(n, spec, transport="ollama", span_id="s", key="arguments.source_column")
    assert round(first["margin_value"], 2) == 10.53 and round(arg["margin_value"], 2) == 8.96
    none = call_evidence({**n, "output": {"text": "not json", "tool_calls": None}},
                         {"type": "json_field", "field": "target"}, transport="ollama", span_id="s")
    assert none["margin_found"] is False and none["margin_note"] == "no decision found in the output"


def test_client_decoding_defaults_to_the_run_condition(tmp_path, server):
    from hbench.design import make_condition
    from hbench.runlog.session import start_run
    cond = make_condition(method={"name": "m"}, model={"name": "m"}, decoding={"temperature": 0.8, "seed": "per_replicate"},
                          engine={"transport": "ollama"}, data={"study": "clark"}, replicate=4)
    log = start_run(tmp_path / "raw", "e0", cond, task={"kind": "sm"}, verify_data=False)
    LLMClient(log, "m", base=server).chat([{"role": "user", "content": "q"}], caller="x")
    log.close()
    sent = _Server.seen[-1][1]["options"]
    assert (sent["temperature"], sent["seed"]) == (0.8, 4)  # stated once, in the condition; the seed resolved
