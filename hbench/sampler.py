"""The server's sampler chain, so the probability of reproducing a decision can be computed from logprobs.

llama-server (Ollama 0.40.0) samples each token through, in this order (its log prints the chain):

    logits -> penalties -> dry -> top-n-sigma -> top-k -> typical -> top-p -> min-p -> xtc -> temperature -> dist

top-p works on the T=1 probabilities, before the temperature is applied. With the defaults of /api/chat
(T=0.8, top_p=0.9) a token whose probability is at least 0.9 is therefore always kept: such positions are
deterministic even at T>0. /v1 defaults (T=1.0, top_p=1.0) give no such protection. The other stages are off
unless set; p_path() refuses (returns None and the reason) rather than ignore one that is on.

Parameters use the server's names (temp, top_k, top_p, min_p). What a call ran with is its stated decoding over
the endpoint's defaults, which env/engine-lock.json records under sampler_defaults and the engine audit checks
call by call.

Logprobs are the model's distribution before any constraint (a format=json grammar can rule out the top token),
and only the top few candidates are visible; the rest of the vocabulary is treated as one candidate. A result is
exact when the top-p nucleus lies within the visible candidates; otherwise it is an approximation and says so.
"""

from __future__ import annotations

import json
import math

NEUTRAL = {"repeat_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "dry_multiplier": 0.0,
           "typical_p": 1.0, "xtc_probability": 0.0, "top_n_sigma": -1.0, "mirostat": 0.0, "adaptive_target": -1.0}
ENDPOINT_DEFAULTS_KEY = {"ollama": "/api/chat", "openai_v1": "/v1/chat/completions",
                         "litellm/openai": "/v1/chat/completions", "litellm/ollama_chat": "/api/chat",
                         "litellm/ollama": "/api/chat"}
_REQUEST_KEYS = {"temperature": "temp", "top_k": "top_k", "top_p": "top_p", "min_p": "min_p",
                 # stages p_path does not model: carried along so unsupported() sees them
                 **{k: k for k in ("repeat_penalty", "frequency_penalty", "presence_penalty", "typical_p", "mirostat")}}
_CHAIN_KEYS = ("temp", "top_k", "top_p", "min_p")


def _lock(lock: dict | None) -> dict:
    if lock is not None:
        return lock
    from .engine.lock import ENGINE_LOCK
    return json.loads(ENGINE_LOCK.read_text()) if ENGINE_LOCK.exists() else {}


def defaults(endpoint: str | None, lock: dict | None = None) -> dict:
    """What the server uses when a request leaves a setting out, for one endpoint ("/api/chat" or
    "/v1/chat/completions"); {} when the lock does not record it."""
    return dict((_lock(lock).get("sampler_defaults") or {}).get(endpoint) or {})


def endpoint_of(transport: str | None, endpoint: str | None = None) -> str | None:
    return endpoint or ENDPOINT_DEFAULTS_KEY.get(transport or "")


def effective(requested: dict | None, endpoint: str | None, lock: dict | None = None) -> dict:
    """The sampler a call ran with: its stated decoding (request names) over the endpoint's defaults. With an
    unknown endpoint only the stated settings are known."""
    out = defaults(endpoint, lock) if endpoint else {}
    for k, v in (requested or {}).items():
        if k in _REQUEST_KEYS and v is not None:
            out[_REQUEST_KEYS[k]] = v
    if (requested or {}).get("seed") is not None:
        out["seed"] = requested["seed"]
    return out


def unsupported(params: dict) -> str | None:
    """Why p_path cannot model these settings (a stage that is not off), or None."""
    bad = [f"{k}={params[k]}" for k, v in NEUTRAL.items() if params.get(k) is not None and params[k] != v]
    return f"sampler stage(s) not modelled: {', '.join(bad)}" if bad else None


def p_keep(entry, params: dict, ruled_out_above: bool = False) -> tuple[float | None, bool]:
    """(probability that the chain picks this position's observed token, exact). At temp 0 the chain is greedy.
    ruled_out_above: a constraint (a format=json grammar) ruled out the candidates scoring above the observed
    token here, as a T=0 source call shows when it did not pick its top candidate; they are left out."""
    tok, lp, top = entry[0], entry[1], entry[2] or []
    cands = [(c, math.exp(clp)) for c, clp in top
             if clp is not None and not (ruled_out_above and lp is not None and clp > lp)]
    if lp is not None and tok not in [c for c, _ in cands]:
        cands.append((tok, math.exp(lp)))
    if not cands:
        return None, False
    cands.sort(key=lambda x: -x[1])
    temp = params.get("temp")
    if temp is not None and temp <= 0:
        return (1.0 if tok == cands[0][0] else 0.0), True
    rest = max(0.0, 1.0 - sum(p for _, p in cands))
    top_k = params.get("top_k")
    if top_k and top_k > 0 and top_k <= len(cands):
        cands, rest = cands[: int(top_k)], 0.0
    top_p = params.get("top_p", 1.0)
    nucleus, cum, exact = [], 0.0, True
    for c, p in cands + [(None, rest)]:  # None stands for the invisible rest of the vocabulary
        if c is None:
            if p <= 0:
                break
            exact = False
        nucleus.append((c, p))
        cum += p
        if cum >= top_p:
            break
    min_p = params.get("min_p") or 0.0
    if min_p > 0:  # an invisible token is at most as likely as the last visible one
        floor = min_p * nucleus[0][1]
        nucleus = [(c, p) for c, p in nucleus if (p if c is not None else cands[-1][1]) >= floor]
    t = 1.0 if temp is None else temp
    weights = [(c, p ** (1.0 / t)) for c, p in nucleus if p > 0]
    total = sum(w for _, w in weights)
    if total <= 0:
        return None, False
    return sum(w for c, w in weights if c == tok) / total, exact


def p_path(tokens: list | None, end: int | None, params: dict, constrained=()) -> dict:
    """{"p", "exact", "reason"}: probability that sampling with params reproduces the observed tokens at
    positions 0..end (e.g. through the end of a decision value). constrained: positions where a constraint ruled
    out the candidates above the observed token (decision_margin's constrained_positions)."""
    reason = unsupported(params)
    if reason:
        return {"p": None, "exact": False, "reason": reason}
    if not tokens or end is None:
        return {"p": None, "exact": False, "reason": "no token logprobs"}
    p, exact = 1.0, True
    for i, entry in enumerate(tokens[: end + 1]):
        k, e = p_keep(entry, params, ruled_out_above=i in constrained)
        if k is None:
            return {"p": None, "exact": False, "reason": "a position has no candidates"}
        p *= k
        exact = exact and e
    return {"p": p, "exact": exact, "reason": None}
