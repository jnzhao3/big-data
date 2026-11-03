import math

import pytest

from hbench import sampler


API_CHAT = {"temp": 0.8, "top_k": 40, "top_p": 0.9, "min_p": 0.0}
V1 = {"temp": 1.0, "top_k": 40, "top_p": 1.0, "min_p": 0.0}


def _entry(chosen, probs):
    """[token, logprob, top-k] with the chosen token first among `probs` ({token: probability})."""
    return [chosen, math.log(probs[chosen]), [[t, math.log(p)] for t, p in probs.items()]]


def test_defaults_and_effective_come_from_the_lock():
    assert sampler.defaults("/api/chat") == API_CHAT and sampler.defaults("/v1/chat/completions") == V1
    eff = sampler.effective({"temperature": 0, "seed": 1}, "/api/chat")
    assert eff == {"temp": 0, "top_k": 40, "top_p": 0.9, "min_p": 0.0, "seed": 1}
    assert sampler.effective({"temperature": None, "top_p": 0.5}, "/v1/chat/completions")["top_p"] == 0.5
    assert sampler.effective({"temperature": 0.3}, None) == {"temp": 0.3}  # unknown endpoint: only what was stated


def test_p_keep_top_p_protects_confident_tokens():
    e = _entry("a", {"a": 0.95, "b": 0.03, "c": 0.01})
    assert sampler.p_keep(e, API_CHAT) == (1.0, True)        # top-p 0.9 keeps only "a"
    p, exact = sampler.p_keep(e, V1)                         # /v1 keeps everything
    assert 0.94 < p < 0.951 and exact is False               # the rest of the vocabulary is not visible


def test_p_keep_v1_defaults_use_temperature_one():
    e = _entry("b", {"a": 0.6, "b": 0.3, "c": 0.05, "d": 0.03, "e": 0.01})
    p, exact = sampler.p_keep(e, V1)
    assert p == pytest.approx(0.3, abs=1e-12) and not exact  # T=1 and no truncation: the model's own probability


def test_p_keep_temperature_zero_is_greedy():
    assert sampler.p_keep(_entry("a", {"a": 0.5, "b": 0.4}), {**API_CHAT, "temp": 0}) == (1.0, True)
    assert sampler.p_keep(_entry("b", {"a": 0.5, "b": 0.4}), {**API_CHAT, "temp": 0}) == (0.0, True)


def test_min_p_drops_candidates_below_its_share_of_the_top():
    e = _entry("a", {"a": 0.5, "b": 0.3, "c": 0.04, "d": 0.03, "e": 0.02})
    without = sampler.p_keep(e, {**V1, "top_p": 0.95})[0]
    with_min_p = sampler.p_keep(e, {**V1, "top_p": 0.95, "min_p": 0.1})[0]  # keeps p >= 0.05: only a and b
    assert with_min_p == pytest.approx(0.5 / 0.8) and with_min_p > without


def test_unsupported_stage_gives_no_number():
    tokens = [_entry("a", {"a": 0.9, "b": 0.05})]
    got = sampler.p_path(tokens, 0, {**API_CHAT, "repeat_penalty": 1.1})
    assert got["p"] is None and "repeat_penalty" in got["reason"]
    assert sampler.unsupported({**API_CHAT, **sampler.NEUTRAL}) is None


def test_p_path_multiplies_positions_up_to_the_end():
    tokens = [_entry("x", {"x": 0.7, "y": 0.2}), _entry("z", {"z": 0.5, "w": 0.4}), _entry("q", {"q": 0.1, "r": 0.8})]
    one, two = sampler.p_path(tokens, 0, V1)["p"], sampler.p_path(tokens, 1, V1)["p"]
    assert one == pytest.approx(0.7) and two == pytest.approx(0.7 * 0.5)  # position 2 is after the decision


def test_a_requested_penalty_reaches_p_path():
    from hbench.decisions import evidence
    params = sampler.effective({"temperature": 0.8, "repeat_penalty": 1.1}, "/api/chat")
    assert params["repeat_penalty"] == 1.1                      # the stated stage survives into the sampler
    ev = evidence("s", [_entry("a", {"a": 0.6, "b": 0.3})], "a", "a",
                  decision={"key": "t", "value": "a", "json_key": None}, temperature=0.8, sampler_params=params)
    assert ev["p_path"] is None and "repeat_penalty" in ev["p_path_note"]


def test_v1_penalties_are_recorded_without_changing_old_input_hashes():
    from hbench.llm.normalize import GENERATION_KEYS, decoding_openai, generation_key
    dec = decoding_openai({"model": "m", "messages": [], "temperature": 0, "frequency_penalty": 0.5})
    assert dec["frequency_penalty"] == 0.5 and dec["presence_penalty"] is None
    plain = decoding_openai({"model": "m", "messages": [], "temperature": 0})
    assert set(generation_key("m", [], plain)["decoding"]) == set(GENERATION_KEYS)   # v2 hashes unchanged
    assert generation_key("m", [], dec)["decoding"]["frequency_penalty"] == 0.5      # but a penalty is part of it


def test_p_path_under_a_grammar_is_a_tighter_lower_bound():
    from hbench.decisions import evidence, evidence_columns
    # format=json ruled out "}\n\n" at the last position; the T=0 call chose "}"
    toks = [["{\"", -0.1, [["{\"", -0.1], ["x", -9.0]]], ["}", -0.99, [["}\n\n", -0.46], ["}", -0.99], ["\"", -3.0]]]]
    d = {"key": "t", "value": "}", "json_key": None}
    ev = evidence("s", toks, '{"}', "}", decision=d, temperature=0, grammar=True)
    plain = sampler.p_path(toks, 1, sampler.defaults("/api/chat"))["p"]
    got = ev["p_path_default"]["/api/chat"]
    assert got["p"] > plain and got["lower_bound"] is True and got["exact"] is False
    cols = evidence_columns(ev)
    assert {"p_path_api_chat_exact", "p_path_v1_exact", "p_path_exact"} <= set(cols)
