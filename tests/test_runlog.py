import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hbench.manifest import verify_manifest
from hbench.runlog import RunLog, make_run_id, read_events
from hbench.runlog.check import check_run
from hbench.runlog.header import collect_run_header
from hbench.runlog.llm import ollama_chat


def test_run_id_format():
    rid = make_run_id("E2", "full qwen2.5:14b", "clark", stamp="20261103T010000Z", rand="ab12")
    assert rid == "E2-full_qwen2.5_14b-clark-20261103T010000Z-ab12"


def test_events_blobs_index_and_never_overwrite(tmp_path):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="run1")
    h = log.emit("run_header", exp="e0", condition={"t": 0})
    big = {"text": "x" * 5000}
    s2 = log.emit("tool_call", parent_id=h, step=1, tool="match_values", result=big, args={"a": 1})
    log.close(status="ok", note="done")
    ev = read_events(tmp_path / "run1")
    assert [e["type"] for e in ev] == ["run_header", "tool_call", "run_end"] and [e["seq"] for e in ev] == [1, 2, 3]
    assert ev[1]["span_id"] == s2 and ev[1]["parent_id"] == h and ev[1]["step"] == 1
    assert ev[1]["args"] == {"a": 1} and ev[1]["result"]["blob"].startswith("sha256:")  # large value went to a blob
    assert log.blobs.get(ev[1]["result"]) == big
    assert ev[2]["event_counts"] == {"run_header": 1, "tool_call": 1}
    index = [json.loads(x) for x in (tmp_path / "runs_index.jsonl").read_text().splitlines()]
    assert [x["event"] for x in index] == ["start", "end"]
    with pytest.raises(FileExistsError):
        RunLog.create(tmp_path, "e0", "t", "clark", run_id="run1")
    with pytest.raises(FileExistsError):
        RunLog(tmp_path / "run1", proc="driver")
    assert check_run(tmp_path / "run1")["problems"] == []


def test_second_process_and_truncated_last_line(tmp_path):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="run2")
    log.emit("run_header", exp="e0")
    mcp = RunLog.attach(tmp_path / "run2", proc="mcp")
    mcp.emit("run_header", proc_role="mcp server")
    mcp.emit("tool_call", tool="preview_domain")
    log.close()
    with open(mcp.path, "a") as f:  # simulate a crash in the middle of a write
        f.write('{"v": "hbench.log/1", "run_id": "ru')
    ev = read_events(tmp_path / "run2")
    assert sorted(e["proc"] for e in ev) == ["driver", "driver", "mcp", "mcp"]
    res = check_run(tmp_path / "run2")
    assert res["processes"] == ["driver", "mcp"] and res["problems"] == []


def test_check_run_flags_missing_header_and_cloud_calls(tmp_path):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="run3")
    log.emit("llm_call", model="openai/gpt-4o-mini", cloud_call=True)
    probs = check_run(tmp_path / "run3")["problems"]
    assert any("not run_header" in p for p in probs) and any("non-local" in p for p in probs)
    assert any("no run_end" in p for p in probs)


def test_threads_share_one_log(tmp_path):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="run4")
    log.emit("run_header")
    ts = [threading.Thread(target=lambda: [log.emit("note", i=i) for i in range(50)]) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    log.close()
    ev = read_events(tmp_path / "run4")
    assert len(ev) == 202 and sorted(e["seq"] for e in ev) == list(range(1, 203))


def test_header_without_ollama(data):
    h = collect_run_header("e0", {"t": 0}, {"study": "clark"}, prompts={"p": "hello"}, data=data,
                           ollama_models=["qwen2.5:14b"], ollama_base="http://127.0.0.1:9", verify_data=False)
    assert h["ollama"]["error"] and h["prompts"]["p"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert h["python"]["flags"]["ignore_environment"] in (0, 1) and h["packages"]["versions"]["pandas"]
    assert all(not any(m in k for m in ("KEY", "TOKEN", "SECRET")) for k in h["env"]["vars"])


def test_manifest_verify_detects_changes(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    manifest = {"files": {"a.csv": {"bytes": 4, "md5": "c4e8e4d2e7b6fa5a74b9ff1ec3fa1eb7"},
                          "missing.csv": {"bytes": 1, "md5": "0"}}}
    probs = verify_manifest(tmp_path, manifest)
    assert "missing: missing.csv" in probs and "changed: a.csv" in probs


class _Ollama(BaseHTTPRequestHandler):
    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = json.dumps({"model": "m", "created_at": "t", "message": {"role": "assistant", "content": '{"target": "bmi"}'},
                          "done": True, "done_reason": "stop", "total_duration": 5, "load_duration": 1,
                          "prompt_eval_count": 7, "eval_count": 3}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def mock_ollama():
    srv = HTTPServer(("127.0.0.1", 0), _Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_ollama_chat_logs_full_call(tmp_path, mock_ollama):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="run5")
    msgs = [{"role": "user", "content": "match BMI"}]
    body = {"model": "qwen2.5:14b", "messages": msgs, "format": "json", "options": {"temperature": 0, "seed": 1}}
    reply = ollama_chat(body, log, "method:test", base=mock_ollama, step=0)
    ollama_chat(body, log, "method:test", base=mock_ollama, step=1)
    assert reply["message"]["content"] == '{"target": "bmi"}'
    a, b = read_events(tmp_path / "run5")
    assert a["params"] == {"format": "json", "options": {"temperature": 0, "seed": 1}}
    assert a["ollama_timing"]["load_duration"] == 1 and a["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}
    assert log.blobs.get(a["messages"][0]) == msgs[0]
    assert a["input_sha256"] == b["input_sha256"] and a["output_sha256"] == b["output_sha256"]
    with pytest.raises(Exception):
        ollama_chat(body, log, "method:test", base="http://127.0.0.1:9", step=2)
    assert read_events(tmp_path / "run5")[-1]["error"]


class _OpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path.endswith("/chat/completions"):
            out = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
                   "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
        elif self.path.endswith("/api/chat"):
            out = {"model": "m", "created_at": "t", "message": {"role": "assistant", "content": "ok"}, "done": True,
                   "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 1}
        else:
            out = {}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_litellm_callback_records_request_body(tmp_path):
    litellm = pytest.importorskip("litellm")
    from hbench.runlog import litellm_cb

    srv = HTTPServer(("127.0.0.1", 0), _OpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="run6")
    cb = litellm_cb.install(log, caller="method:test")
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": "calling", "tool_calls": [
                   {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "[]"}]
    try:
        with litellm_cb.parent_span("driver:7", step=3):
            litellm.completion(model="openai/qwen2.5:14b", messages=history, api_base=base + "/v1", api_key="x",
                               temperature=0)
            litellm.completion(model="ollama_chat/qwen2.5:14b", messages=history, api_base=base, temperature=0)
        assert cb.wait(10)  # success callbacks run in a background thread
    finally:
        litellm_cb.uninstall(cb)
        srv.shutdown()
    calls = [e for e in read_events(tmp_path / "run6") if e["type"] == "llm_call"]
    assert len(calls) == 2
    by = {c["provider"]: c for c in calls}
    assert by["openai"]["response"]["content"] == "ok" and by["openai"]["cloud_call"] is False
    assert by["openai"]["parent_id"] == "driver:7" and by["openai"]["step"] == 3
    req_openai = by["openai"]["request_body"]
    req_ollama = by["ollama_chat"]["request_body"]
    req_openai = log.blobs.get(req_openai) if "blob" in req_openai else req_openai
    req_ollama = log.blobs.get(req_ollama) if "blob" in req_ollama else req_ollama
    # what was actually sent: the openai route keeps the assistant tool call, ollama_chat drops it
    assert "tool_calls" in [m for m in req_openai["messages"] if m["role"] == "assistant"][0]
    assert "tool_calls" not in [m for m in req_ollama["messages"] if m["role"] == "assistant"][0]


def test_is_local():
    pytest.importorskip("litellm")
    from hbench.runlog.litellm_cb import is_local
    assert is_local("http://127.0.0.1:11434/v1", "openai") and is_local(None, "ollama_chat")
    assert not is_local(None, "openai") and not is_local("https://api.openai.com/v1", "openai")
