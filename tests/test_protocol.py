"""Engine protocols (start action + per-call action), shared by natural runs and replays (E0 v3, T3)."""
import json
import shutil
from pathlib import Path

import pytest

from conftest import MockOllama
from hbench.design import condition_id, effective_seed, make_condition, validate_condition
from hbench.llm import LLMClient
from hbench.runlog import RunLog, read_events
from hbench.runlog.session import start_run

FX = Path(__file__).parent / "fixtures"


def _cond(protocol="natural", transport="ollama", **kw):
    return make_condition(method={"name": "m"}, model={"name": "m"}, decoding=kw.pop("decoding", {"temperature": 0}),
                          engine={"transport": transport, "protocol": protocol}, data={"study": "clark"}, **kw)


def _start(tmp_path, server, cond):
    return start_run(tmp_path / "raw", "e0", cond, task={"kind": "sm"}, models=["m"], lease=False, audit=False,
                     verify_data=False, ollama_base=server)


def test_protocol_vocabulary():
    assert validate_condition({**_cond(), "engine": {"protocol": "warm"}})  # never implemented: rejected
    a = _cond(decoding={"temperature": 0.8, "seed": "per_replicate"}, replicate=3)
    b = _cond(decoding={"temperature": 0.8, "seed": "per_replicate"}, replicate=4)
    assert condition_id(a) == condition_id(b) and effective_seed(a) == 3 and effective_seed(_cond()) is None
    assert "decoding.seed" in "; ".join(validate_condition({**a, "decoding": {"temperature": 0.8, "seed": "x"}}))


def test_start_run_cold_start_unloads_and_logs(tmp_path, server):
    log = _start(tmp_path, server, _cond("cold_start"))
    unloads = [b for p, b in MockOllama.seen if p == "/api/generate"]
    assert unloads == [{"model": "m", "keep_alive": 0}]
    log.close()
    types = [e["type"] for e in read_events(log.run_dir)]
    assert types[0] == "run_header" and types[1] == "engine_action"
    act = next(e for e in read_events(log.run_dir) if e["type"] == "engine_action")
    assert (act["action"], act["model"], act["confirmed"], act["protocol"]) == ("unload", "m", True, "cold_start")


def test_per_call_protocol_needs_the_hbench_client(tmp_path, server):
    with pytest.raises(ValueError, match="per-call"):
        _start(tmp_path, server, _cond("cold", transport="litellm/openai"))
    log = _start(tmp_path, server, _cond("cold_start", transport="litellm/openai"))  # a start action works anywhere
    log.close()


def test_client_cold_protocol_unloads_before_each_call(tmp_path, server):
    log = _start(tmp_path, server, _cond("cold"))
    client = LLMClient(log, "m", base=server)
    for _ in range(2):
        client.chat([{"role": "user", "content": "q"}], caller="x")
    log.close()
    types = [e["type"] for e in read_events(log.run_dir) if e["type"] in ("engine_action", "llm_call")]
    assert types == ["engine_action", "llm_call", "engine_action", "llm_call"]
    assert LLMClient(log, "m", base=server, per_call="none").per_call == "none"  # replays manage the engine themselves


def test_apply_overrides_by_transport():
    from hbench.replay import apply_overrides
    native = {"model": "m", "messages": [], "options": {"temperature": 0, "num_ctx": 32768}}
    got = apply_overrides(native, "ollama", {"temperature": 0.8, "seed": 3})
    assert got["options"] == {"temperature": 0.8, "seed": 3, "num_ctx": 32768} and native["options"]["temperature"] == 0
    v1 = apply_overrides({"model": "m", "messages": [], "temperature": 0}, "openai_v1", {"temperature": 1.0, "top_p": 0.5})
    assert (v1["temperature"], v1["top_p"]) == (1.0, 0.5)
    with pytest.raises(ValueError):
        apply_overrides(native, "ollama", {"num_ctx": 4096})          # not a sampler setting
    with pytest.raises(ValueError):
        apply_overrides(v1, "openai_v1", {"min_p": 0.1})               # /v1 has no min_p


def _source_run(tmp_path, server, n=3):
    src = RunLog.create(tmp_path / "raw", "e0", "t", "clark", run_id="src")
    src.emit("run_header", condition=_cond(), engine=None)
    c = LLMClient(src, "m", base=server, decoding={"temperature": 0, "seed": 1})
    for _ in range(n):
        c.chat([{"role": "user", "content": "q"}], caller="method:test", format="json",
               decision={"type": "json_field", "field": "target"})
    src.close()
    return src.run_dir


def test_cold_start_replay_requires_a_prefix(tmp_path, server):
    from hbench.cli import main
    run = _source_run(tmp_path, server)
    with pytest.raises(SystemExit, match="prefix"):
        main(["replay", str(run), "--protocol", "cold_start", "--calls", "1,2", "--out", str(tmp_path / "rep")])


def test_replay_seeds_share_one_condition(tmp_path, server):
    from hbench.cli import main
    run = _source_run(tmp_path, server, n=1)
    MockOllama.seen = []
    assert main(["replay", str(run), "--protocol", "cold_start", "--set", "temperature=0.8", "--seeds", "1-2",
                 "--out", str(tmp_path / "rep")]) == 0
    sent = [b for p, b in MockOllama.seen if p == "/api/chat"]
    assert [b["options"]["seed"] for b in sent] == [1, 2] and all(b["options"]["temperature"] == 0.8 for b in sent)
    assert [b for p, b in MockOllama.seen if p == "/api/generate"] == [{"model": "m", "keep_alive": 0}] * 2
    runs = sorted((tmp_path / "rep").glob("replay-*"))
    headers = [next(e for e in read_events(r) if e["type"] == "run_header") for r in runs]
    assert len({h["condition_id"] for h in headers}) == 1 and sorted(h["condition"]["replicate"] for h in headers) == [1, 2]
    calls = [e for r in runs for e in read_events(r) if e["type"] == "llm_call"]
    assert sorted(e["overrides"]["seed"] for e in calls) == [1, 2]
    rows = [row for r in runs for row in __import__("csv").DictReader(open(r / "replay_compare.csv"))]
    assert all(row["overrides"] and row["p_path_source"] for row in rows)


def test_audit_verifies_the_protocol(tmp_path):
    from hbench.engine.audit import audit_problems, audit_run
    shutil.copy(FX / "ollama_server_window.log", tmp_path / "server.log")
    sizes = [2741, 2741, 42, 42, 42, 42, 42, 2741, 2741, 2741, 2741]
    out = {}
    for protocol in ("cold_start", "cold"):
        log = RunLog.create(tmp_path / "raw", "e1", "t", "clark", run_id=protocol)
        log.emit("run_header", condition=_cond(protocol),
                 engine={"server_log": {"path": str(tmp_path / "server.log"), "offset": 0}})
        for n in sizes:
            log.emit("llm_call", api_base="http://localhost:11434", response_raw={"blob": "sha256:00", "bytes": 1},
                     engine={"prompt_tokens": n}, decoding={"temperature": None})
        out[protocol] = audit_run(log.run_dir)
    assert out["cold_start"]["protocol_violations"] == []      # the window starts with a runner start, then cold
    assert len(out["cold"]["protocol_violations"]) == 10       # only the first task follows a runner start
    assert any("protocol cold" in p for p in audit_problems(out["cold"]))


def test_cold_start_without_models_is_refused(tmp_path):
    with pytest.raises(ValueError, match="models"):
        start_run(tmp_path / "raw", "e0", _cond("cold_start"), task={"kind": "sm"}, models=(), verify_data=False)


def test_cold_start_replay_repeats_are_separate_runs(tmp_path, server):
    from hbench.cli import main
    run = _source_run(tmp_path, server, n=1)
    MockOllama.seen = []
    assert main(["replay", str(run), "--protocol", "cold_start", "--repeat", "2", "--out", str(tmp_path / "rep")]) == 0
    assert len([b for p, b in MockOllama.seen if p == "/api/generate"]) == 2   # each repeat starts cold
    assert len(list((tmp_path / "rep").glob("replay-*"))) == 2
