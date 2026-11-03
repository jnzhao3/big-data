import json
import threading

import pytest

from hbench.manifest import verify_manifest
from hbench.runlog import RunLog, make_run_id, read_events
from hbench.runlog.check import check_run
from hbench.runlog.header import collect_run_header


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
    assert h["engine"]["api"]["error"] and h["prompts"]["p"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert h["condition_id"] is None  # a free-form condition gets no id; make_condition() conditions do"
    assert h["python"]["flags"]["ignore_environment"] in (0, 1) and h["packages"]["versions"]["pandas"]
    assert all(not any(m in k for m in ("KEY", "TOKEN", "SECRET")) for k in h["env"]["vars"])


def test_manifest_verify_detects_changes(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    manifest = {"files": {"a.csv": {"bytes": 4, "md5": "c4e8e4d2e7b6fa5a74b9ff1ec3fa1eb7"},
                          "missing.csv": {"bytes": 1, "md5": "0"}}}
    probs = verify_manifest(tmp_path, manifest)
    assert "missing: missing.csv" in probs and "changed: a.csv" in probs




def test_close_hooks_feed_run_end_and_never_lose_it(tmp_path):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="run7")
    log.emit("run_header")
    log.add_close_hook(lambda l: {"engine_audit": {"foreign_tasks": 0}})
    log.add_close_hook(lambda l: 1 / 0)
    log.close()
    end = read_events(tmp_path / "run7")[-1]
    assert end["type"] == "run_end" and end["engine_audit"] == {"foreign_tasks": 0}
    assert "ZeroDivisionError" in end["close_hook_errors"][0]


def test_check_run_reports_audit_and_drift(tmp_path):
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="run8")
    log.emit("run_header", engine_drift=["ollama 0.41.0 != locked 0.40.0"])
    log.close(engine_audit={"foreign_tasks": 2, "unmatched_calls": [], "truncated": ["driver:3"], "sampler_mismatches": []})
    probs = check_run(tmp_path / "run8")["problems"]
    assert any("engine-lock" in x for x in probs) and any("another client" in x for x in probs)
    assert any("truncated" in x for x in probs)
