"""One logged client for Ollama's native /api/chat and its OpenAI-compatible /v1.

    client = LLMClient(log, "qwen2.5:14b", transport="ollama", decoding={"temperature": 0, "seed": 1})
    res = client.chat(messages, caller="method:clark_match", step=i, format="json")
    res.text, res.token_logprobs, res.path_class, res.span_id

Logprobs are requested by default (top_logprobs=5): they cost little and give every decision a margin.
num_ctx defaults to 32768, the context /v1 gets for qwen2.5:14b, so both transports share one loaded runner
instead of reloading the model whenever a caller asks for a different context.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field

from ..runlog.ollama import DEFAULT_BASE
from ..runlog.writer import RunLog
from .normalize import NORMALIZERS
from .record import record_llm_call

ENDPOINTS = {"ollama": "/api/chat", "openai_v1": "/v1/chat/completions"}
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


class LLMClient:
    def __init__(self, log: RunLog | None, model: str, transport: str = "ollama", base: str = DEFAULT_BASE,
                 decoding: dict | None = None, num_ctx: int | None = DEFAULT_NUM_CTX,
                 top_logprobs: int | None = DEFAULT_TOP_LOGPROBS, timeout: float = 600.0):
        if transport not in ENDPOINTS:
            raise ValueError(f"transport must be one of {sorted(ENDPOINTS)}")
        self.log, self.model, self.transport, self.base = log, model, transport, base.rstrip("/")
        self.decoding = dict(decoding or {})
        self.num_ctx, self.top_logprobs, self.timeout = num_ctx, top_logprobs, timeout

    # ------------------------------------------------------------------ requests

    def build_request(self, messages, *, tools=None, format=None, **decoding) -> dict:
        """The exact JSON body for this transport. decoding keys: temperature, top_p, top_k, min_p, seed,
        max_tokens, stop, repeat_penalty, think."""
        d = {**self.decoding, **decoding}
        if self.transport == "ollama":
            options = {k: d[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed", "stop", "repeat_penalty")
                       if d.get(k) is not None}
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
            body.update({k: d[k] for k in ("temperature", "top_p", "top_k", "seed", "max_tokens", "stop")
                         if d.get(k) is not None})
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
        body = self.build_request(messages, tools=tools, format=format, **decoding)
        extra = {"decision_spec": decision} if decision else {}
        return self.send(body, caller=caller, step=step, parent_id=parent_id, **extra)

    def send(self, body: dict, *, caller: str, step: int | None = None, parent_id: str | None = None,
             **extra) -> LLMResult:
        """POST an exact body (also used by replay), log it as one llm_call, return the normalized result.
        Failed requests are logged with the error and re-raised."""
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
                         latency_s=latency, request=body, response=response, normalized=normalized)

    # ------------------------------------------------------------------ engine state helpers

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def loaded(self) -> list[str]:
        with urllib.request.urlopen(self.base + "/api/ps", timeout=10) as r:
            return [m.get("name") for m in json.load(r).get("models", [])]

    def unload(self, wait_s: float = 30.0) -> bool:
        """Unload the model (keep_alive=0) and wait until /api/ps no longer lists it: the next call starts cold,
        with an empty prompt cache. Not logged as an llm_call (it generates nothing)."""
        self._post("/api/generate", {"model": self.model, "keep_alive": 0})
        end = time.time() + wait_s
        while time.time() < end:
            if self.model not in self.loaded():
                return True
            time.sleep(0.25)
        return False
