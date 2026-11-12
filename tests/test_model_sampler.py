"""A model's own sampler settings (its Modelfile PARAMETER lines) fill in what a request leaves out.

qwen3.8-27b-bdikit sets temperature 1, top_k 20 and top_p 0.95. The server's defaults (top_k 40, top_p 0.9 on
/api/chat) apply only to settings neither the request nor the model states, and on /v1 Ollama's OpenAI layer
sends temperature and top_p itself (1.0)."""

import json
import math

from hbench import sampler
from hbench.design import make_condition
from hbench.runlog import read_events
from hbench.runlog.session import start_run
from hbench.runlog.writer import RunLog

from conftest import MockOllama

PARAMS_27B = ("temperature                    1\ntop_k                          20\ntop_p                          0.95\n"
              "min_p                          0\nnum_ctx                        32768\npresence_penalty               0\n"
              "repeat_penalty                 1")
MODEL_27B = {"temp": 1.0, "top_k": 20.0, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0}


def _entry(chosen, probs):
    return [chosen, math.log(probs[chosen]), [[t, math.log(p)] for t, p in probs.items()]]


def test_modelfile_parameters_parse_to_server_names():
    assert sampler.model_sampler(PARAMS_27B) == MODEL_27B          # num_ctx is not a sampler setting
    assert sampler.model_sampler(None) == {} and sampler.model_sampler("") == {}
    assert sampler.model_sampler({"temperature": 0.7, "stop": "<|im_end|>"}) == {"temp": 0.7}


def test_defaults_put_the_model_over_the_server_except_what_v1_sends():
    assert sampler.defaults("/api/chat", model=MODEL_27B) == {**MODEL_27B}
    v1 = sampler.defaults("/v1/chat/completions", model=MODEL_27B)
    assert v1["temp"] == 1.0 and v1["top_p"] == 1.0 and v1["top_k"] == 20.0   # observed: top_k 20 on /v1 for 27b
    assert sampler.defaults("/api/chat", model={}) == sampler.defaults("/api/chat")  # a model without settings
    eff = sampler.effective({"temperature": 0, "top_p": 1.0}, "/v1/chat/completions", model=MODEL_27B)
    assert eff["temp"] == 0 and eff["top_p"] == 1.0 and eff["top_k"] == 20.0      # stated settings win


def test_lookup_ignores_route_prefix_and_latest_tag():
    by = {"qwen3.8-27b-bdikit": MODEL_27B}
    assert sampler.lookup_model_sampler(by, "openai/qwen3.8-27b-bdikit:latest") == MODEL_27B
    assert sampler.lookup_model_sampler({"m:latest": {}}, "m") == {}
    assert sampler.lookup_model_sampler(by, "qwen2.5:14b") is None                 # unknown, not "none"


def test_default_temperature_prediction_uses_the_models_sampler():
    from hbench.decisions import evidence
    toks = [_entry("a", {"a": 0.92, "b": 0.05, "c": 0.02})]
    d = {"key": "t", "value": "a", "json_key": None}
    server = evidence("s", toks, "a", "a", decision=d, temperature=0)["p_path_default"]["/api/chat"]
    model = evidence("s", toks, "a", "a", decision=d, temperature=0,
                     model_sampler=MODEL_27B)["p_path_default"]["/api/chat"]
    assert server["p"] == 1.0          # top_p 0.9 keeps only "a"
    assert model["p"] < 0.99           # top_p 0.95 keeps "b" as well, at temperature 1


def _cond(transport="ollama"):
    return make_condition(method={"name": "m"}, model={"name": "m"}, decoding={"temperature": 0, "seed": 1},
                          engine={"transport": transport}, data={"study": "clark"})


def test_run_records_the_models_sampler_on_every_call(tmp_path, server):
    from hbench.llm import LLMClient
    MockOllama.show = {"parameters": PARAMS_27B}
    log = start_run(tmp_path / "raw", "e0", _cond(), task={"kind": "sm"}, models=["m"], lease=False, audit=False,
                    verify_data=False, ollama_base=server)
    res = LLMClient(log, "m", base=server).chat([{"role": "user", "content": "q"}], caller="method:t")
    log.close()
    ev = read_events(log.run_dir)
    header = next(e for e in ev if e["type"] == "run_header")
    assert header["effective_decoding"]["top_k"] == 20.0 and header["effective_decoding"]["temp"] == 0
    call = next(e for e in ev if e["type"] == "llm_call")
    assert call["model_sampler"] == MODEL_27B and res.model_sampler == MODEL_27B
    att = RunLog.attach(log.run_dir, "mcp.1")                       # another process of the run knows it too
    assert sampler.lookup_model_sampler(att.model_samplers, "m") == MODEL_27B


def _audited_run(tmp_path, server_top_k, recorded=True):
    """A run whose engine_audit.json says the server ran top_k=server_top_k, with an audit summary written
    before the model's own settings were taken into account (it flagged top_k)."""
    log = RunLog.create(tmp_path / "raw", "e0", "t", "clark", run_id=f"r{server_top_k}{recorded}")
    models = {"m": {"modelfile_parameters": PARAMS_27B}}
    log.emit("run_header", condition=_cond("litellm/openai"), engine={"api": {"models": models}})
    call = {"span_id": "driver:1", "model": "openai/m", "transport": "litellm/openai",
            "decoding": {"temperature": 0, "top_p": 1.0}}
    if recorded:
        call["model_sampler"] = MODEL_27B
    log.emit("llm_call", **{k: v for k, v in call.items() if k != "span_id"})
    span = next(e for e in read_events(log.run_dir) if e["type"] == "llm_call")["span_id"]
    ran = {"temp": 0.0, "top_k": float(server_top_k), "top_p": 1.0, "min_p": 0.0}
    (log.run_dir / "engine_audit.json").write_text(json.dumps({"calls": [{"span_id": span, "sampler": ran}]}))
    old = {"matched": 1, "sampler_mismatches": [{"span_id": span, "key": "top_k", "requested": 40, "effective": 20.0}]}
    log.add_close_hook(lambda _log: {"engine_audit": old})
    log.close()
    return log.run_dir


def test_validate_rechecks_old_audits_with_the_models_sampler(tmp_path):
    from hbench.runlog.check import check_run
    for recorded in (True, False):                                  # from the call, else from the run header
        probs = check_run(_audited_run(tmp_path, 20, recorded))["problems"]
        assert not any("sampler settings" in p for p in probs), probs
    probs = check_run(_audited_run(tmp_path, 40))["problems"]      # the server really ignored the model
    assert any("sampler settings other than requested (top_k)" in p for p in probs)
