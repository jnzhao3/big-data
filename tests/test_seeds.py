"""Per-call sampling seeds (hbench.seeds): llama-server reseeds its sampler with each request's seed, so calls
that share one seed sample with the same random draws. E0 turns a stated seed into one seed per sampled call."""
import json

import pandas as pd
import pytest

from conftest import MockOllama
from hbench.design import make_condition, validate_condition
from hbench.llm import LLMClient
from hbench.runlog import RunLog, read_events
from hbench.runlog.session import start_run
from hbench.seeds import SEED_RULE, call_seed, check_seeds, is_sampled


def _cond(decoding, replicate=0):
    return make_condition(method={"name": "m"}, model={"name": "m"}, decoding=decoding,
                          engine={"transport": "ollama"}, data={"study": "clark"}, replicate=replicate)


def _start(tmp_path, server, cond, name="raw"):
    return start_run(tmp_path / name, "e0", cond, task={"kind": "sm"}, models=["m"], lease=False, audit=False,
                     verify_data=False, ollama_base=server)


def _sent_seeds():
    return [b["options"].get("seed") for p, b in MockOllama.seen if p == "/api/chat"]


def _three_calls(client):
    for q in ("a", "b", "a"):  # the same prompt twice: it still gets two seeds
        client.chat([{"role": "user", "content": q}], caller="method:test", format="json")


def test_call_seed_rule():
    s = call_seed(1, "method:x", "p" * 64, 0)
    assert s == call_seed(1, "method:x", "p" * 64, 0) and 0 <= s < 2 ** 31
    assert len({s, call_seed(2, "method:x", "p" * 64, 0), call_seed(1, "method:y", "p" * 64, 0),
                call_seed(1, "method:x", "q" * 64, 0), call_seed(1, "method:x", "p" * 64, 1)}) == 5
    assert SEED_RULE == "call/1"
    assert is_sampled(0.8) and is_sampled(None) and not is_sampled(0) and not is_sampled(0.0)


def test_client_sends_one_seed_per_sampled_call(tmp_path, server):
    runs = []
    for name in ("a", "b"):
        MockOllama.seen = []
        log = _start(tmp_path, server, _cond({"temperature": 0.8, "seed": "per_replicate"}, replicate=5), name)
        _three_calls(LLMClient(log, "m", base=server))
        log.close()
        runs.append((_sent_seeds(), log.run_dir))
    seeds, run_dir = runs[0]
    assert len(set(seeds)) == 3 and runs[1][0] == seeds          # distinct per call, reproducible from the seed
    calls = [e for e in read_events(run_dir) if e["type"] == "llm_call"]
    assert [e["seed_base"] for e in calls] == [5, 5, 5] and {e["seed_rule"] for e in calls} == {SEED_RULE}
    ks = [0, 0, 1]
    assert seeds == [call_seed(5, "driver/method:test", e["prompt_sha256"], k) for e, k in zip(calls, ks)]
    assert [e["decoding"]["seed"] for e in calls] == seeds         # the log records the seed actually sent
    assert check_seeds(read_events(run_dir))["independent"] is True


def test_greedy_calls_keep_the_stated_seed(tmp_path, server):
    # at T=0 the seed is unused; keeping it leaves T=0 requests byte-identical to earlier runs
    MockOllama.seen = []
    log = _start(tmp_path, server, _cond({"temperature": 0, "seed": 1}))
    _three_calls(LLMClient(log, "m", base=server))
    log.close()
    assert _sent_seeds() == [1, 1, 1]
    assert not any("seed_rule" in e for e in read_events(log.run_dir) if e["type"] == "llm_call")


def test_shared_seed_mode_is_explicit(tmp_path, server):
    assert "seed_mode" in "; ".join(validate_condition(_cond({"temperature": 0.8, "seed": 1})
                                                       | {"decoding": {"temperature": 0.8, "seed_mode": "x"}}))
    MockOllama.seen = []
    log = _start(tmp_path, server, _cond({"temperature": 0.8, "seed": 7, "seed_mode": "shared"}))
    _three_calls(LLMClient(log, "m", base=server))
    log.close()
    assert _sent_seeds() == [7, 7, 7]
    chk = check_seeds(read_events(log.run_dir))
    assert chk["problems"] == [] and chk["independent"] is False and any("shared" in n for n in chk["notes"])


def test_litellm_route_gets_per_call_seeds(tmp_path, server):
    litellm = pytest.importorskip("litellm")
    from hbench.runlog import litellm_cb
    MockOllama.seen = []
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    cb = litellm_cb.install(log, caller="method:magneto")
    try:
        for q in ("a", "b"):
            litellm.completion(model="openai/m", messages=[{"role": "user", "content": q}], api_base=server + "/v1",
                               api_key="x", temperature=0.8, seed=3)
        litellm.completion(model="openai/m", messages=[{"role": "user", "content": "c"}], api_base=server + "/v1",
                           api_key="x", temperature=0, seed=3)
        assert cb.wait(10)
    finally:
        litellm_cb.uninstall(cb)
    sent = [b.get("seed") for p, b in MockOllama.seen if p == "/v1/chat/completions"]
    calls = [e for e in read_events(tmp_path / "r") if e["type"] == "llm_call"]
    assert sent[2] == 3 and len(set(sent[:2])) == 2 and 3 not in sent[:2]
    by_seed = {e["decoding"]["seed"]: e for e in calls}
    for s in sent[:2]:
        e = by_seed[s]
        assert e["seed_base"] == 3 and e["seed_rule"] == SEED_RULE
        assert s == call_seed(3, "driver/method:magneto", e["prompt_sha256"], 0)


def _ev(seed, prompt, temp=0.8, rule=None, caller="m"):
    e = {"type": "llm_call", "caller": caller, "prompt_sha256": prompt, "decoding": {"temperature": temp, "seed": seed}}
    return {**e, **({"seed_rule": rule, "seed_base": 1} if rule else {})}


def test_check_seeds_flags_shared_draws():
    shared = check_seeds([_ev(1, "p1"), _ev(1, "p2"), _ev(1, "p3", temp=0)])
    assert shared["independent"] is False and "share one seed" in shared["problems"][0]
    repeat = check_seeds([_ev(1, "p1"), _ev(1, "p1")])                            # a seeded repeat of one prompt
    assert repeat["problems"] == [] and repeat["independent"] is False and "seeded repeats" in repeat["notes"][0]
    pre = check_seeds([_ev(1, "p1"), _ev(1, "p1", caller="replay:pre")])           # a replay pre-send is no repeat
    assert pre["independent"] is True
    collide = check_seeds([_ev(9, "p1", rule=SEED_RULE), _ev(9, "p2", rule=SEED_RULE)])  # derived: chance only
    assert collide["problems"] == [] and collide["independent"] is True
    unseeded = check_seeds([_ev(None, "p1"), _ev(None, "p2")])
    assert unseeded["problems"] == [] and any("no seed" in n for n in unseeded["notes"])
    assert check_seeds([_ev(1, "p1", temp=0), _ev(1, "p2", temp=0)]) == {"problems": [], "notes": [],
                                                                          "independent": None, "n_sampled": 0}
    fixed = check_seeds([{"type": "run_header", "proc": "driver",
                          "condition": {"decoding": {"temperature": 0.8, "seed": 4}}}, _ev(5, "p1", rule=SEED_RULE)])
    assert any("every replicate" in n for n in fixed["notes"])


def test_log_validate_reports_shared_seeds(tmp_path):
    from hbench.runlog.check import check_run
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    log.emit("run_header", condition=None)
    for p in ("p1", "p2"):
        log.emit("llm_call", caller="m", prompt_sha256=p, decoding={"temperature": 0.8, "seed": 1})
    log.emit("run_end")
    log.close()
    out = check_run(tmp_path / "r")
    assert any("share one seed" in p for p in out["problems"]) and out["seeds"]["independent"] is False


def test_sampling_check_needs_independent_seeds():
    from hbench.stability import sampling_check
    long = pd.DataFrame({"run_id": ["r1", "r2", "r1", "r2"], "item_id": ["a", "a", "b", "b"],
                         "pred": ["x", "y", "x", "x"], "greedy_path": [True, False, True, True],
                         "p_path": [0.5, None, 0.6, 0.6]})
    ok = sampling_check(long)
    assert ok["alarm"] is False and ok["interval_pairwise"]
    bad = sampling_check(long, seed_independent=False)
    assert bad["bound_expected"] == ok["bound_expected"]          # expectations add up under any dependence
    assert bad["alarm"] is None and bad["interval_pairwise"] is None and "seed" in bad["dependence_note"]


def test_replay_seeds_reproduce_a_natural_run(tmp_path, server):
    from hbench.cli import main
    MockOllama.seen = []
    log = _start(tmp_path, server, _cond({"temperature": 0.8, "seed": "per_replicate"}, replicate=2))
    _three_calls(LLMClient(log, "m", base=server))
    log.close()
    natural = _sent_seeds()
    assert len(set(natural)) == 3
    MockOllama.seen = []
    assert main(["replay", str(log.run_dir), "--protocol", "as_is", "--set", "temperature=0.8", "--seeds", "2",
                 "--out", str(tmp_path / "rep")]) == 0
    assert _sent_seeds() == natural                     # the same rule: replaying seed 2 sends the run's seeds


def test_replay_of_a_greedy_run_at_t_above_zero_derives_seeds(tmp_path, server):
    from hbench.cli import main
    MockOllama.seen = []
    log = _start(tmp_path, server, _cond({"temperature": 0, "seed": 1}))
    _three_calls(LLMClient(log, "m", base=server))
    log.close()
    MockOllama.seen = []
    assert main(["replay", str(log.run_dir), "--protocol", "as_is", "--set", "temperature=0.8",
                 "--out", str(tmp_path / "rep")]) == 0
    seeds = _sent_seeds()
    assert len(set(seeds)) == 3 and 1 not in seeds       # the source's stated seed 1 is the base
    rep = next((tmp_path / "rep").glob("replay-*"))
    assert check_seeds(read_events(rep))["independent"] is True


def test_litellm_uninstall_stops_logging(tmp_path, server):
    # LiteLLM copies installed callbacks into its per-event lists; an uninstalled logger used to stay there and
    # wrote every later call, of any run in the process, into its own run's log
    litellm = pytest.importorskip("litellm")
    from hbench.runlog import litellm_cb
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r1")
    cb = litellm_cb.install(log, caller="first")
    kw = dict(model="openai/m", api_base=server + "/v1", api_key="x", temperature=0)
    litellm.completion(messages=[{"role": "user", "content": "a"}], **kw)
    assert litellm_cb.uninstall(cb)
    litellm.completion(messages=[{"role": "user", "content": "b"}], **kw)
    assert not any(cb in getattr(litellm, n, []) for n in ("callbacks", "input_callback", "success_callback"))
    assert sum(e["type"] == "llm_call" for e in read_events(tmp_path / "r1")) == 1


def test_replay_set_seed_is_a_base(tmp_path, server):
    from hbench.cli import main
    MockOllama.seen = []
    log = _start(tmp_path, server, _cond({"temperature": 0, "seed": 1}))
    _three_calls(LLMClient(log, "m", base=server))
    log.close()
    MockOllama.seen = []
    assert main(["replay", str(log.run_dir), "--protocol", "as_is", "--set", "temperature=0.8", "--set", "seed=5",
                 "--out", str(tmp_path / "rep")]) == 0
    seeds = _sent_seeds()
    assert len(set(seeds)) == 3 and 5 not in seeds
    with pytest.raises(SystemExit, match="not both"):
        main(["replay", str(log.run_dir), "--set", "seed=5", "--seeds", "1", "--out", str(tmp_path / "rep2")])


def test_attached_process_takes_the_run_settings_and_its_own_seeds(tmp_path, server):
    MockOllama.seen = []
    log = _start(tmp_path, server, _cond({"temperature": 0.8, "seed": "per_replicate"}, replicate=3))
    msg = [{"role": "user", "content": "same"}]
    LLMClient(log, "m", base=server).chat(msg, caller="tool:llm")
    mcp = RunLog.attach(log.run_dir, "mcp")
    LLMClient(mcp, "m", base=server).chat(msg, caller="tool:llm")      # same caller and prompt, other process
    mcp.close()
    log.close()
    seeds = _sent_seeds()
    assert len(set(seeds)) == 2
    calls = [e for e in read_events(log.run_dir) if e["type"] == "llm_call"]
    assert {e["seed_base"] for e in calls} == {3} and check_seeds(read_events(log.run_dir))["independent"] is True
    MockOllama.seen = []
    shared = _start(tmp_path, server, _cond({"temperature": 0.8, "seed": 7, "seed_mode": "shared"}), "raw2")
    att = RunLog.attach(shared.run_dir, "mcp")
    LLMClient(att, "m", base=server).chat(msg, caller="tool:llm")
    att.close()
    shared.close()
    assert _sent_seeds() == [7]


def test_replay_seeds_reproduce_a_two_process_run(tmp_path, server):
    from hbench.cli import main
    MockOllama.seen = []
    log = _start(tmp_path, server, _cond({"temperature": 0.8, "seed": "per_replicate"}, replicate=2))
    msg = [{"role": "user", "content": "same"}]
    LLMClient(log, "m", base=server).chat(msg, caller="tool:llm")
    mcp = RunLog.attach(log.run_dir, "mcp")
    LLMClient(mcp, "m", base=server).chat(msg, caller="tool:llm")
    mcp.close()
    log.close()
    natural = sorted(_sent_seeds())
    MockOllama.seen = []
    assert main(["replay", str(log.run_dir), "--protocol", "as_is", "--seeds", "2",
                 "--out", str(tmp_path / "rep")]) == 0
    assert sorted(_sent_seeds()) == natural


def test_check_replicates_finds_copies():
    from hbench.seeds import check_replicates
    a, b = [_ev(11, "p1", rule=SEED_RULE)], [_ev(11, "p1", rule=SEED_RULE)]
    assert check_replicates({"r1": a, "r2": b}) == {"independent": False, "shared": [["r1", "r2", 1]]}
    assert check_replicates({"r1": a, "r2": [_ev(12, "p1", rule=SEED_RULE)]})["independent"] is True
    assert check_replicates({"r1": [_ev(1, "p", temp=0)], "r2": [_ev(1, "p", temp=0)]})["independent"] is None


def test_stability_drops_the_alarm_for_seeded_copies(tmp_path, server):
    # a fixed integer seed at T>0: each replicate sends the same per-call seeds, so the runs are copies
    from hbench.cli import main
    items = []
    for r in (1, 2):
        log = _start(tmp_path, server, _cond({"temperature": 0.8, "seed": 4}, replicate=r), f"raw{r}")
        LLMClient(log, "m", base=server).chat([{"role": "user", "content": "q"}], caller="method:test")
        log.close()
        p = log.run_dir / "items.gdc-sm.csv"
        pd.DataFrame({"run_id": [log.run_dir.name], "item_id": ["a"], "pred": ["x"], "correct": ["True"],
                      "margin": ["1.0"], "p_path": ["0.5"], "greedy_path": ["True"], "kind": ["sm"],
                      "study": ["clark"]}).to_csv(p, index=False)
        items.append(str(p))
    out = tmp_path / "stab.json"
    assert main(["stability", *items, "--out", str(out)]) == 0
    (rep,) = json.loads(out.read_text()).values()
    assert rep["sampling_check"]["seed_independent"] is False and rep["sampling_check"]["alarm"] is None


def test_ollama_chat_route_logs_its_sampler_settings(tmp_path, server):
    litellm = pytest.importorskip("litellm")
    from hbench.runlog import litellm_cb
    MockOllama.seen = []
    log = RunLog.create(tmp_path, "e0", "t", "clark", run_id="r")
    cb = litellm_cb.install(log, caller="m")
    try:
        for q in ("a", "b"):
            litellm.completion(model="ollama_chat/m", messages=[{"role": "user", "content": q}], api_base=server,
                               temperature=0.8, seed=3)
        assert cb.wait(10)
    finally:
        litellm_cb.uninstall(cb)
    calls = [e for e in read_events(tmp_path / "r") if e["type"] == "llm_call"]
    sent = [b["options"]["seed"] for p, b in MockOllama.seen if p == "/api/chat"]
    assert [e["decoding"]["temperature"] for e in calls] == [0.8, 0.8] and len(set(sent)) == 2
    assert sorted(e["decoding"]["seed"] for e in calls) == sorted(sent) and {e["seed_base"] for e in calls} == {3}


def test_explicit_per_call_mode_is_the_default_condition():
    from hbench.design import condition_id
    assert condition_id(_cond({"temperature": 0.8, "seed": 1, "seed_mode": "per_call"})) == \
        condition_id(_cond({"temperature": 0.8, "seed": 1}))


def _episode(ids, endpoint="/v1/chat/completions"):
    """The second request of an agent episode: the history carries the server's tool-call ids."""
    a, b = ids
    calls = [{"id": a, "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
             {"id": b, "type": "function", "function": {"name": "g", "arguments": '{"y": 2}'}}]
    if endpoint == "/api/chat":
        for c in calls:
            c["function"]["arguments"] = json.loads(c["function"]["arguments"])
    return [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": a, "content": "r1"}, {"role": "tool", "tool_call_id": b, "content": "r2"}]


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/api/chat"])
def test_prompt_hash_ignores_random_tool_call_ids(endpoint):
    from hbench.llm.normalize import canonical_tool_ids, generation_key, prompt_key
    from hbench.llm.record import prompt_sha256
    from hbench.runlog.writer import sha256_json
    m1, m2 = _episode(("call_x1vkyext", "call_4q34kdk2"), endpoint), _episode(("call_aaaa", "call_bbbb"), endpoint)
    req = lambda m: {"model": "m", "messages": m}
    assert prompt_sha256(req(m1), {}) == prompt_sha256(req(m2), {})
    assert sha256_json(generation_key("m", m1, {})) != sha256_json(generation_key("m", m2, {}))  # exact input
    canon = canonical_tool_ids(m1)
    assert [c["id"] for c in canon[1]["tool_calls"]] == ["call_0", "call_1"]
    assert [canon[2]["tool_call_id"], canon[3]["tool_call_id"]] == ["call_0", "call_1"]
    assert m1[1]["tool_calls"][0]["id"] == "call_x1vkyext"                       # the request is not modified
    swapped = _episode(("call_x1vkyext", "call_4q34kdk2"), endpoint)
    swapped[2]["tool_call_id"], swapped[3]["tool_call_id"] = swapped[3]["tool_call_id"], swapped[2]["tool_call_id"]
    assert prompt_sha256(req(swapped), {}) != prompt_sha256(req(m1), {})           # which result answers which call
    plain = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert canonical_tool_ids(plain) == plain                                     # no ids: earlier hashes unchanged
    assert prompt_sha256(req(plain), {}) == sha256_json(
        {"model": "m", "messages": plain, "tools": None, "format": None, "think": None})
    assert prompt_key("m", None, None, None)["messages"] == []


def test_seeded_agent_episode_repeats_with_new_tool_call_ids(tmp_path, server):
    """Two runs of one seeded T>0 episode send the same seeds although the server gave other tool-call ids."""
    runs = []
    for name, ids in (("a", ("call_x1vkyext", "call_4q34kdk2")), ("b", ("call_9zz", "call_8yy"))):
        MockOllama.seen = []
        log = _start(tmp_path, server, _cond({"temperature": 0.8, "seed": 1}), name)
        c = LLMClient(log, "m", base=server)
        c.chat([{"role": "user", "content": "q"}], caller="agent")
        c.chat(_episode(ids, "/api/chat"), caller="agent")
        log.close()
        runs.append(_sent_seeds())
    assert runs[0] == runs[1] and len(set(runs[0])) == 2


def test_litellm_episode_repeats_with_new_tool_call_ids(tmp_path, server):
    litellm = pytest.importorskip("litellm")
    from hbench.runlog import litellm_cb
    runs = []
    for name, ids in (("a", ("call_x1vkyext", "call_4q34kdk2")), ("b", ("call_9zz", "call_8yy"))):
        MockOllama.seen = []
        log = RunLog.create(tmp_path, "e0", "t", "clark", run_id=name)
        cb = litellm_cb.install(log, caller="agent")
        try:
            litellm.completion(model="openai/m", messages=_episode(ids), api_base=server + "/v1", api_key="x",
                               temperature=0.8, seed=1)
            assert cb.wait(10)
        finally:
            litellm_cb.uninstall(cb)
        runs.append([b.get("seed") for p, b in MockOllama.seen if p == "/v1/chat/completions"])
    assert runs[0] == runs[1] and runs[0][0] != 1
