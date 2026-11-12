"""One logged client for Ollama's native /api/chat and its OpenAI-compatible /v1.

    client = LLMClient(log, "qwen2.5:14b", transport="ollama", decoding={"temperature": 0, "seed": 1})
    res = client.chat(messages, caller="method:clark_match", step=i, format="json",
                      decision={"type": "json_field", "field": "target"})
    res.text, res.token_logprobs, res.path_class, res.span_id
    res.evidence()     # the prediction record's evidence for that decision, from the call itself

Logprobs are requested by default (top_logprobs=5): they cost little and give every decision a margin.
num_ctx defaults to 32768, the context /v1 gets for qwen2.5:14b, so both transports share one loaded runner
instead of reloading the model whenever a caller asks for a different context.

With a log opened by start_run, the client takes its decoding from the run's condition unless it is given
one (a "per_replicate" seed resolved), so the settings are stated once. A stated seed is a base: every sampled
call (T>0) is sent with its own seed derived from it (hbench.seeds), unless the condition states
decoding.seed_mode "shared". send() sends a body exactly as given. It performs the per-call action of the
run's engine protocol (hbench.design.PROTOCOLS): with a log
opened by start_run under protocol "cold" it unloads the model before every call. Unloads are logged as
engine_action events, so a run's log shows every change of engine state it caused.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field

from ..runlog.ollama import DEFAULT_BASE
from ..runlog.writer import RunLog
from .normalize import NORMALIZERS
from .record import prompt_sha256, record_llm_call

ENDPOINTS = {"ollama": "/api/chat", "openai_v1": "/v1/chat/completions"}
DECODING_KEYS = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "stop", "repeat_penalty",
                 "frequency_penalty", "presence_penalty", "think")
DEFAULT_NUM_CTX = 32768
DEFAULT_TOP_LOGPROBS = 5


@dataclass
class LLMResult:
    text: str | None
    tool_calls: list | None
    finish_reason: str | None
    token_logprobs: list | None
    prompt_tokens: int | None
    cached_tokens: int | None
    path_class: str
    span_id: str | None
    latency_s: float
    request: dict
    response: dict
    normalized: dict = field(repr=False)
    prompt_sha256: str | None = None
    transport: str | None = None
    caller: str | None = None
    decision_spec: dict | list | None = None

    def evidence(self, key: str | None = None, **extra) -> dict:
        """The `evidence` dict for this call's decision (the first its decision spec names, or the one named
        `key`): margin, reproduction probability and the rest, everything taken from the call itself."""
        from ..decisions import call_evidence
        return call_evidence(self.normalized, self.decision_spec, transport=self.transport, span_id=self.span_id,
                             prompt_sha256=self.prompt_sha256, key=key, caller=self.caller, **extra)


def _seed_slot(body: dict) -> dict:
    """The part of a request body that holds the sampler settings: options on /api/chat, the body on /v1."""
    return body["options"] if isinstance(body.get("options"), dict) else body


def derive_call_seed(body: dict, normalized_decoding: dict, deriver, caller: str | None) -> int | None:
    """Replace a sampled request's stated seed with its per-call seed, in place. Returns the base, or None
    when nothing was replaced (no seed, or a greedy call)."""
    from ..seeds import is_sampled
    slot = _seed_slot(body)
    base = slot.get("seed")
    if base is None or isinstance(base, bool) or not isinstance(base, int) or not is_sampled(slot.get("temperature")):
        return None
    slot["seed"] = deriver.derive(base, caller, prompt_sha256(body, normalized_decoding))
    return base


def _condition_decoding(log) -> dict:
    """The decoding a run's condition states, as client settings ({} without one)."""
    cond = getattr(log, "condition", None) or {}
    dec = {k: v for k, v in (cond.get("decoding") or {}).items() if k in DECODING_KEYS and v is not None}
    if dec.get("seed") == "per_replicate":
        dec["seed"] = (getattr(log, "effective_decoding", None) or {}).get("seed")
    return dec


class LLMClient:
    def __init__(self, log: RunLog | None, model: str, transport: str = "ollama", base: str = DEFAULT_BASE,
                 decoding: dict | None = None, num_ctx: int | None = DEFAULT_NUM_CTX,
                 top_logprobs: int | None = DEFAULT_TOP_LOGPROBS, timeout: float = 600.0, per_call: str | None = None):
        """decoding: by default the run's condition's (the condition start_run stored on the log).
        per_call: the per-call engine action ("none" or "unload"); by default the one of the run's protocol.
        Replays prepare the engine themselves and pass "none"."""
        from ..design import protocol_actions

        if transport not in ENDPOINTS:
            raise ValueError(f"transport must be one of {sorted(ENDPOINTS)}")
        self.log, self.model, self.transport, self.base = log, model, transport, base.rstrip("/")
        from ..seeds import deriver_for, seed_mode
        self.decoding = dict(decoding) if decoding is not None else _condition_decoding(log)
        self.seed_mode = self.decoding.pop("seed_mode", None) or seed_mode(getattr(log, "condition", None))
        self._seeds = deriver_for(log)
        self.num_ctx, self.top_logprobs, self.timeout = num_ctx, top_logprobs, timeout
        self.per_call = per_call or protocol_actions(getattr(log, "condition", None))[1]

    # ------------------------------------------------------------------ requests

    def build_request(self, messages, *, tools=None, format=None, **decoding) -> dict:
        """The exact JSON body for this transport. decoding keys: temperature, top_p, top_k, min_p, seed,
        max_tokens, stop, repeat_penalty, frequency_penalty, presence_penalty, think."""
        d = {**self.decoding, **decoding}
        if self.transport == "ollama":
            options = {k: d[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed", "stop", "repeat_penalty",
                                         "frequency_penalty", "presence_penalty") if d.get(k) is not None}
            if d.get("max_tokens") is not None:
                options["num_predict"] = d["max_tokens"]
            if self.num_ctx is not None:
                options["num_ctx"] = self.num_ctx
            body = {"model": self.model, "messages": messages, "stream": False, "options": options}
            if format is not None:
                body["format"] = format
            if d.get("think") is not None:
                body["think"] = d["think"]
        else:
            body = {"model": self.model, "messages": messages}
            body.update({k: d[k] for k in ("temperature", "top_p", "top_k", "seed", "max_tokens", "stop",
                                           "frequency_penalty", "presence_penalty") if d.get(k) is not None})
            if format == "json":
                body["response_format"] = {"type": "json_object"}
            elif format is not None:
                body["response_format"] = format
        if tools:
            body["tools"] = tools
        if self.top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        return body

    def chat(self, messages, *, caller: str, step: int | None = None, parent_id: str | None = None, tools=None,
             format=None, decision: dict | None = None, **decoding) -> LLMResult:
        """decision: how to find the method's decision in the output (see hbench.decisions.parser_from_spec),
        stored with the call so replay and analysis tools can extract it from the log alone."""
        from ..seeds import SEED_RULE
        body = self.build_request(messages, tools=tools, format=format, **decoding)
        extra = {"decision_spec": decision} if decision else {}
        if self.seed_mode != "shared":
            base = derive_call_seed(body, NORMALIZERS[self.transport](body, None)["decoding"], self._seeds, caller)
            if base is not None:
                extra.update(seed_base=base, seed_rule=SEED_RULE)
        return self.send(body, caller=caller, step=step, parent_id=parent_id, **extra)

    def send(self, body: dict, *, caller: str, step: int | None = None, parent_id: str | None = None,
             **extra) -> LLMResult:
        """POST an exact body (also used by replay), log it as one llm_call, return the normalized result.
        Failed requests are logged with the error and re-raised."""
        if self.per_call == "unload":
            self.unload()
        endpoint = ENDPOINTS[self.transport]
        req = urllib.request.Request(self.base + endpoint, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer ollama"})
        t0 = time.time()
        response, error = None, None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                response = json.load(r)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            latency = round(time.time() - t0, 3)
            normalized = NORMALIZERS[self.transport](body, response)
            span_id = None
            if self.log is not None:
                span_id = record_llm_call(self.log, caller=caller, transport=self.transport, endpoint=endpoint,
                                          api_base=self.base, model=body.get("model"), request=body,
                                          response=response, normalized=normalized, latency_s=latency, error=error,
                                          parent_id=parent_id, step=step, **extra)
        out, eng = normalized["output"], normalized["engine"]
        return LLMResult(text=out["text"], tool_calls=out["tool_calls"], finish_reason=out["finish_reason"],
                         token_logprobs=normalized["token_logprobs"], prompt_tokens=eng["prompt_tokens"],
                         cached_tokens=eng["cached_tokens"], path_class=eng["path_class"], span_id=span_id,
                         latency_s=latency, request=body, response=response, normalized=normalized,
                         prompt_sha256=prompt_sha256(body, normalized["decoding"]), transport=self.transport,
                         caller=caller, decision_spec=extra.get("decision_spec"))

    # ------------------------------------------------------------------ engine state helpers

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def loaded(self) -> list[str]:
        with urllib.request.urlopen(self.base + "/api/ps", timeout=10) as r:
            return [m.get("name") for m in json.load(r).get("models", [])]

    def unload(self, wait_s: float = 30.0, protocol: str | None = None) -> bool:
        """Unload the model (keep_alive=0) and wait until /api/ps no longer lists it: the next call starts cold,
        with an empty prompt cache. Logged as an engine_action event (not an llm_call: it generates nothing)."""
        t0 = time.time()
        self._post("/api/generate", {"model": self.model, "keep_alive": 0})
        end, confirmed = t0 + wait_s, False
        while time.time() < end:
            if self.model not in self.loaded():
                confirmed = True
                break
            time.sleep(0.25)
        if self.log is not None:
            proto = protocol or ((getattr(self.log, "condition", None) or {}).get("engine") or {}).get("protocol")
            self.log.emit("engine_action", action="unload", model=self.model, confirmed=confirmed,
                          waited_s=round(time.time() - t0, 3), protocol=proto)
        return confirmed
