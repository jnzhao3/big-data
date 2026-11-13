import json
import shutil
from pathlib import Path

import pytest

from hbench.design import condition_id, condition_slug, make_condition, ollama_lease, validate_condition
from hbench.engine.audit import audit_problems, audit_run, join_calls, parse_requests
from hbench.engine.lock import check, lock_from_snapshot
from hbench.engine.snapshot import parse_env_map, parse_runner_cmd
from hbench.runlog import RunLog

FX = Path(__file__).parent / "fixtures"
LOG = (FX / "ollama_server_window.log").read_text()


def test_parse_server_log_window():
    p = parse_requests(LOG)
    t = p["tasks"]
    assert len(t) == 11 and p["runner_starts"] == 1 and [o["path"] for o in p["other_requests"]] == ["/api/show"]
    assert [x["http"]["path"] for x in t[:3]] == ["/v1/chat/completions"] * 3
    assert (t[0]["n_tokens"], t[0]["cached"], t[0]["computed"]) == (2741, 0, 2741)       # cold, batched
    assert (t[1]["n_tokens"], t[1]["cached"], t[1]["computed"]) == (2741, 2740, 1)       # identical resend
    assert [x["sampler"]["temp"] for x in t] == [0.0] * 3 + [0.8] * 4 + [0.0] * 4
    assert t[0]["sampler"]["top_p"] == 1.0 and t[3]["sampler"]["top_p"] == 0.9         # endpoint-dependent default
    assert [i for i, x in enumerate(t) if x["ram_restore"]] == [7] and not any(x["truncated"] for x in t)


def _call(span, prompt, temp=0.0, cached=None):
    return {"span_id": span, "engine": {"prompt_tokens": prompt, "cached_tokens": cached},
            "decoding": {"temperature": temp}}


def test_join_flags_foreign_requests_and_unlogged_calls():
    tasks = parse_requests(LOG)["tasks"]  # 2741, 2741, 42, 42, 42, 42, 42, 2741, 2741, 2741, 2741 prompt tokens
    ours = [_call("d:1", 2741), _call("d:2", 2741), _call("d:3", 42)] + [_call(f"d:{i}", 2741) for i in range(4, 8)]
    j = join_calls(ours, tasks)
    assert len(j["matched"]) == 7 and j["unmatched_calls"] == []
    assert [t["n_tokens"] for t in j["foreign"]] == [42, 42, 42, 42]  # requests we did not make
    j2 = join_calls(ours + [_call("d:99", 999)], tasks)
    assert [c["span_id"] for c in j2["unmatched_calls"]] == ["d:99"]  # a call the server never saw


def test_audit_run_end_to_end(tmp_path):
    shutil.copy(FX / "ollama_server_window.log", tmp_path / "server.log")
    log = RunLog.create(tmp_path / "raw", "e1", "t", "clark", run_id="r")
    log.emit("run_header", engine={"server_log": {"path": str(tmp_path / "server.log"), "offset": 0}})
    for i, (n, t) in enumerate([(2741, 0.0), (2741, 0.0), (42, 0.0), (42, 0.8), (42, 0.8), (42, 0.8), (42, 0.5),
                                (2741, 0.0), (2741, 0.0), (2741, 0.0), (2741, 0.0)]):
        log.emit("llm_call", api_base="http://localhost:11434", response_raw={"blob": "sha256:00", "bytes": 1},
                 engine={"prompt_tokens": n, "cached_tokens": None}, decoding={"temperature": t})
    summary = audit_run(log.run_dir)
    assert (summary["matched"], summary["foreign_tasks"], summary["unmatched_calls"]) == (11, 0, [])
    assert len(summary["sampler_mismatches"]) == 1          # we asked for 0.5, the server ran 0.8
    assert summary["ram_restores"] == ["driver:9"] and summary["runner_starts"] == 1
    assert (log.run_dir / "engine_audit.json").exists()
    assert audit_problems(summary) == ["1 call(s) ran with sampler settings other than requested (temp)"]
    missing = audit_run(log.run_dir, server_log=str(tmp_path / "missing.log"))
    assert "unavailable" in missing and audit_problems(missing)[0].startswith("engine audit not possible")


def test_runner_cmd_env_map_and_lock_drift():
    flags = parse_runner_cmd("/x/llama-server --model /b --port 1 -c 32768 -np 1 --no-jinja --cache-type-k q8_0 "
                             "--flash-attn on -b 1024 -ub 1024 --context-shift --keep 4")
    assert flags["c"] == "32768" and flags["no-jinja"] is True and flags["context-shift"] is True and flags["keep"] == "4"
    env = parse_env_map("OLLAMA_NUM_PARALLEL:1 OLLAMA_ORIGINS:[http://a https://b] OLLAMA_KV_CACHE_TYPE:q8_0 X:")
    assert env == {"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_ORIGINS": "[http://a https://b]", "OLLAMA_KV_CACHE_TYPE": "q8_0", "X": ""}
    snap = {"api": {"version": "0.40.0", "models": {"m": {"digest": "d1", "template_sha256": "t1"}}},
            "server_config": {"OLLAMA_NUM_PARALLEL": "1"}, "runners": {"m": {"model_blob_sha256": "b", "flags": flags}}}
    lock = lock_from_snapshot(snap)
    assert check(snap, lock) == []
    snap2 = json.loads(json.dumps(snap))
    snap2["api"]["version"] = "0.41.0"
    snap2["runners"]["m"]["flags"]["cache-type-k"] = "f16"
    snap2["api"]["models"]["m"]["digest"] = "d2"
    drift = check(snap2, lock)
    assert len(drift) == 3 and any("0.41.0" in d for d in drift) and any("cache-type-k" in d for d in drift)
    pc = dict(snap, prompt_cache_mib=8192.0)
    assert check(dict(pc, prompt_cache_mib=4096.0), lock_from_snapshot(pc)) == ["prompt cache 4096 MiB != locked 8192 MiB"]
    no_log = {"api": snap["api"], "server_log": {"path": "/x/ollama.log", "offset": None}, "server_config": {},
              "runners": {"m": {"unavailable": "no runner start found in the log"}}}
    assert check(no_log, lock) == ["not checked: server config and runner flags (no readable server log at "
                                   "/x/ollama.log; set HBENCH_OLLAMA_LOG)"]  # a missing log must not pass as "ok"


def _cond(**kw):
    base = dict(method={"name": "clark_match", "mode": "full"}, model={"name": "qwen2.5:14b"},
                decoding={"temperature": 0, "seed": 1}, engine={"transport": "ollama", "protocol": "natural"},
                data={"study": "clark"})
    base.update(kw)
    return make_condition(**base)


def test_condition_schema():
    a, b = _cond(replicate=1), _cond(replicate=2)
    assert condition_id(a) == condition_id(b)                                  # replicates share a condition
    assert condition_id(a) != condition_id(_cond(decoding={"temperature": 0.8}))
    assert condition_slug(a) == "clark_match-full-qwen2.5:14b-T0-s1-r1"
    assert "decoding.temperature must be stated" in "; ".join(validate_condition({**a, "decoding": {}}))
    with pytest.raises(ValueError):
        _cond(data={})
    with pytest.raises(ValueError):
        _cond(engine={"protocol": "lukewarm"})


def test_ollama_lease_is_exclusive(tmp_path):
    path = tmp_path / "ollama.lock"
    with ollama_lease("first", path=path) as rec:
        assert rec["holder"] == "first"
        with pytest.raises(TimeoutError, match="first"):
            with ollama_lease("second", path=path):
                pass
    with ollama_lease("third", path=path):
        pass


def test_audit_compares_every_sampler_setting(tmp_path):
    shutil.copy(FX / "ollama_server_window.log", tmp_path / "server.log")
    log = RunLog.create(tmp_path / "raw", "e1", "t", "clark", run_id="r")
    log.emit("run_header", engine={"server_log": {"path": str(tmp_path / "server.log"), "offset": 0}})
    plan = [(2741, "openai_v1", 0.0)] * 2 + [(42, "openai_v1", 0.0)] + [(42, "ollama", 0.8)] * 4 + [(2741, "ollama", 0.0)] * 4
    for i, (n, transport, t) in enumerate(plan):
        dec = {"temperature": t, **({"top_p": 0.95} if i == 8 else {})}  # stated, but the server ran 0.9
        log.emit("llm_call", api_base="http://localhost:11434", transport=transport, engine={"prompt_tokens": n},
                 response_raw={"blob": "sha256:00", "bytes": 1}, decoding=dec)
    s = audit_run(log.run_dir)
    assert [(m["span_id"], m["key"], m["requested"], m["effective"]) for m in s["sampler_mismatches"]] == [
        ("driver:10", "top_p", 0.95, 0.9)]  # defaults of both endpoints (top_p 1.0 on /v1, 0.9 on /api/chat) match
    assert audit_problems(s) == ["1 call(s) ran with sampler settings other than requested (top_p)"]


def test_effective_decoding_in_header(tmp_path):
    from hbench.runlog.session import start_run
    cond = make_condition(method={"name": "m"}, model={"name": "qwen2.5:14b"}, decoding={"temperature": 0, "seed": 1},
                          engine={"transport": "ollama"}, data={"study": "clark"})
    log = start_run(tmp_path, "e0", cond, task={"kind": "sm"}, verify_data=False)
    log.close()
    from hbench.runlog import read_events
    header = next(e for e in read_events(log.run_dir) if e["type"] == "run_header")
    assert header["effective_decoding"] == {"temp": 0, "top_k": 40, "top_p": 0.9, "min_p": 0.0, "seed": 1}
    assert log.effective_decoding == header["effective_decoding"]


def test_audit_flags_sampler_stages_p_path_cannot_model(tmp_path):
    text = LOG.replace("repeat_penalty = 1.000", "repeat_penalty = 1.100", 1)  # the first task's server params
    (tmp_path / "server.log").write_text(text)
    log = RunLog.create(tmp_path / "raw", "e1", "t", "clark", run_id="r")
    log.emit("run_header", engine={"server_log": {"path": str(tmp_path / "server.log"), "offset": 0}})
    for n in [2741, 2741, 42, 42, 42, 42, 42, 2741, 2741, 2741, 2741]:
        log.emit("llm_call", api_base="http://localhost:11434", response_raw={"blob": "sha256:00", "bytes": 1},
                 engine={"prompt_tokens": n}, decoding={"temperature": None})
    s = audit_run(log.run_dir)
    assert [m["span_id"] for m in s["sampler_unmodelled"]] == ["driver:2"]
    assert any("not modelled" in p for p in audit_problems(s))


def _batch_run(tmp_path, monkeypatch, loaded_batch, offset=0, runner_blob=None, name="r"):
    """The fixture window as a qwen2.5:14b run (its runner starts with -b 1024 -ub 1024) under a lock that pins
    loaded_batch for the model."""
    import hbench.engine.lock as lockmod
    lock = json.loads(lockmod.ENGINE_LOCK.read_text())
    if loaded_batch:
        lock["models"]["qwen2.5:14b"]["loaded_batch"] = loaded_batch
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(lockmod, "ENGINE_LOCK", tmp_path / "lock.json")
    if not (tmp_path / "server.log").exists():
        shutil.copy(FX / "ollama_server_window.log", tmp_path / "server.log")
    log = RunLog.create(tmp_path / "raw", "e1", "t", "clark", run_id=name)
    runners = {"qwen2.5:14b": {"model_blob_sha256": runner_blob}} if runner_blob else {}
    log.emit("run_header", engine={"server_log": {"path": str(tmp_path / "server.log"), "offset": offset},
                                   "runners": runners})
    for n, t in [(2741, 0.0), (2741, 0.0), (42, 0.0), (42, 0.8), (42, 0.8), (42, 0.8), (42, 0.8),
                 (2741, 0.0), (2741, 0.0), (2741, 0.0), (2741, 0.0)]:
        log.emit("llm_call", api_base="http://localhost:11434", model="openai/qwen2.5:14b", transport="openai_v1",
                 response_raw={"blob": "sha256:00", "bytes": 1}, engine={"prompt_tokens": n},
                 decoding={"temperature": t})
    return log


def test_audit_records_the_loaded_batch_and_checks_the_lock(tmp_path, monkeypatch):
    blob = "2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54"
    log = _batch_run(tmp_path, monkeypatch, {"b": 1024, "ub": 1024}, runner_blob=blob)
    s = audit_run(log.run_dir)
    rows = json.loads((log.run_dir / "engine_audit.json").read_text())["calls"]
    assert {(r["batch"]["b"], r["batch"]["ub"], r["batch"]["model_blob"]) for r in rows} == {(1024, 1024, blob)}
    assert s["batch_mismatches"] == [] and not any("batch" in p for p in audit_problems(s))
    log = _batch_run(tmp_path, monkeypatch, {"b": 512, "ub": 512}, name="r512")
    s = audit_run(log.run_dir)
    assert len(s["batch_mismatches"]) == 11 and s["batch_mismatches"][0]["loaded"] == {"b": 1024, "ub": 1024}
    assert any(p.startswith("11 call(s) ran on a runner whose batch sizes are not the locked ones")
               for p in audit_problems(s))
    log = _batch_run(tmp_path, monkeypatch, {"b": 1024, "ub": 1024}, runner_blob="f" * 64, name="other")
    assert len(audit_run(log.run_dir)["batch_mismatches"]) == 11      # served by another model's runner
    log = _batch_run(tmp_path, monkeypatch, None, name="unpinned")
    assert audit_run(log.run_dir)["batch_mismatches"] == []         # a model without loaded_batch is not checked


def test_loaded_batch_of_a_model_started_before_the_run(tmp_path, monkeypatch):
    from hbench.engine.audit import loaded_batches
    raw = (FX / "ollama_server_window.log").read_bytes()
    after_runner = raw.index(b"\n", raw.index(b'msg="starting llama-server"')) + 1
    shutil.copy(FX / "ollama_server_window.log", tmp_path / "server.log")
    got = loaded_batches(str(tmp_path / "server.log"), after_runner, len(raw))
    assert len(got) == 11 and {(g["b"], g["ub"]) for g in got} == {(1024, 1024)}
    log = _batch_run(tmp_path, monkeypatch, {"b": 512, "ub": 512}, offset=after_runner)
    assert len(audit_run(log.run_dir)["batch_mismatches"]) == 11


def test_validate_rechecks_batches_against_the_current_lock(tmp_path, monkeypatch):
    from hbench.engine.audit import recheck_sampler
    from hbench.runlog import read_events
    log = _batch_run(tmp_path, monkeypatch, None)
    s = audit_run(log.run_dir)
    assert s["batch_mismatches"] == []
    _batch_run(tmp_path, monkeypatch, {"b": 512, "ub": 512}, name="lockonly")   # the lock now pins 512
    assert len(recheck_sampler(log.run_dir, read_events(log.run_dir), s)["batch_mismatches"]) == 11
    p = log.run_dir / "engine_audit.json"                                     # an audit from before batches
    old = json.loads(p.read_text())
    for r in old["calls"]:
        r.pop("batch")
    p.write_text(json.dumps(old))
    assert recheck_sampler(log.run_dir, read_events(log.run_dir), s)["batch_mismatches"] == []


def test_tag_entry_finds_latest_tags():
    from hbench.runlog.ollama import tag_entry
    t = {"models": [{"name": "qwen3.8-27b-bdikit:latest", "digest": "f33c"}, {"name": "qwen2.5:14b", "digest": "7cdf"},
                    {"name": "qwen3.8-27b-bdikit-nb1024:latest", "digest": "01ce"}]}
    assert tag_entry(t, "qwen3.8-27b-bdikit")["digest"] == "f33c"
    assert tag_entry(t, "qwen3.8-27b-bdikit:latest")["digest"] == "f33c"
    assert tag_entry(t, "qwen3.8-27b-bdikit-nb1024")["digest"] == "01ce"
    assert tag_entry(t, "qwen2.5:14b")["digest"] == "7cdf" and tag_entry(t, "qwen2.5") is None
