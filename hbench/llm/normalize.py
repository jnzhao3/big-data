"""Translate raw request/response payloads of each transport into the normalized llm_call fields.

Pure functions, so normalization can be re-run on stored raw payloads (`hbench llm rederive`) whenever the
schema grows. Every transport fills the same keys; what a transport cannot provide is listed in `unavailable`
with the reason, so analyses can tell "missing" from "zero".

Token logprobs are stored as [token, logprob, [[candidate, logprob], ...]] per generated token; the candidate
list is the server's top-k and normally includes the chosen token itself.
"""

from __future__ import annotations

import json

# Parameters that change what the model generates. Logprob settings are excluded: they change the
# response payload, not the generation, so they stay out of input_sha256.
GENERATION_KEYS = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "stop", "format", "num_ctx",
                   "think", "repeat_penalty", "tools")
# Also part of what generates, but only when set: added after v2, so leaving them out when absent keeps every
# earlier input_sha256 unchanged.
EXTRA_GENERATION_KEYS = ("frequency_penalty", "presence_penalty", "typical_p", "mirostat")
COLD_LOAD_MS = 500.0  # a native call whose load_duration exceeds this loaded the model itself


def path_class(prompt_tokens, cached_tokens, load_ms=None) -> str:
    """How the engine computed the prompt: cold_full (model loaded during the call), full (no reuse),
    partial (prefix reused, several tokens recomputed), single_token (only the last token recomputed,
    e.g. an identical resend), or unknown. The E1 pilot found single_token paths shift logprobs relative
    to batched paths, so this is a covariate for every analysis of run-to-run differences."""
    if prompt_tokens is None or cached_tokens is None:
        return "unknown"
    if cached_tokens <= 0:
        return "cold_full" if load_ms is not None and load_ms >= COLD_LOAD_MS else "full"
    return "single_token" if prompt_tokens - cached_tokens <= 1 else "partial"


def _tokens(entries) -> list | None:
    if entries is None:
        return None
    out = []
    for e in entries:
        top = [[t.get("token"), t.get("logprob")] for t in (e.get("top_logprobs") or [])]
        out.append([e.get("token"), e.get("logprob"), top])
    return out


def _tool_calls(calls):
    """[{"name", "arguments"}]. /v1 sends arguments as a JSON string (re-serialized, without the spaces the model
    generated), /api/chat as an object; both become an object, so the same call compares and hashes the same.
    A string that is not valid JSON is kept, flagged arguments_invalid."""
    if not calls:
        return None
    out = []
    for c in calls:
        fn = c.get("function", c)
        item = {"name": fn.get("name"), "arguments": fn.get("arguments")}
        if isinstance(item["arguments"], str):
            try:
                item["arguments"] = json.loads(item["arguments"])
            except json.JSONDecodeError:
                item["arguments_invalid"] = True
        out.append(item)
    return out


def decoding_native(request: dict) -> dict:
    o = request.get("options") or {}
    return {"temperature": o.get("temperature"), "top_p": o.get("top_p"), "top_k": o.get("top_k"),
            "min_p": o.get("min_p"), "seed": o.get("seed"), "max_tokens": o.get("num_predict"), "stop": o.get("stop"),
            "format": request.get("format"), "num_ctx": o.get("num_ctx"), "think": request.get("think"),
            "repeat_penalty": o.get("repeat_penalty"), "frequency_penalty": o.get("frequency_penalty"),
            "presence_penalty": o.get("presence_penalty"), "typical_p": o.get("typical_p"),
            "mirostat": o.get("mirostat"), "tools": bool(request.get("tools")) or None,
            "logprobs": request.get("logprobs"), "top_logprobs": request.get("top_logprobs")}


def decoding_openai(request: dict) -> dict:
    rf = request.get("response_format") or {}
    return {"temperature": request.get("temperature"), "top_p": request.get("top_p"), "top_k": request.get("top_k"),
            "min_p": None, "seed": request.get("seed"),
            "max_tokens": request.get("max_tokens", request.get("max_completion_tokens")), "stop": request.get("stop"),
            "format": "json" if rf.get("type") == "json_object" else (rf or None), "num_ctx": None, "think": None,
            "repeat_penalty": None, "frequency_penalty": request.get("frequency_penalty"),
            "presence_penalty": request.get("presence_penalty"), "typical_p": None, "mirostat": None,
            "tools": bool(request.get("tools")) or None,
            "logprobs": request.get("logprobs"), "top_logprobs": request.get("top_logprobs")}


def normalize_native(request: dict, response: dict | None) -> dict:
    """Ollama /api/chat."""
    unavailable = {}
    if response is None:
        return {"decoding": decoding_native(request), "engine": None, "output": None, "token_logprobs": None,
                "unavailable": {"response": "request failed"}}
    msg = response.get("message") or {}
    prompt, cached = response.get("prompt_eval_count"), response.get("prompt_eval_cached_count")
    if cached is None:
        unavailable["cached_tokens"] = "server did not report prompt_eval_cached_count"
    load_ms = response["load_duration"] / 1e6 if response.get("load_duration") is not None else None
    ms = lambda k: None if response.get(k) is None else response[k] / 1e6  # noqa: E731
    tokens = _tokens(response.get("logprobs"))
    if tokens is None:
        unavailable["token_logprobs"] = "not requested" if not request.get("logprobs") else "not returned"
    return {"decoding": decoding_native(request),
            "engine": {"prompt_tokens": prompt, "cached_tokens": cached,
                       "computed_tokens": None if prompt is None or cached is None else prompt - cached,
                       "load_ms": load_ms, "path_class": path_class(prompt, cached, load_ms),
                       "prompt_eval_ms": ms("prompt_eval_duration"), "eval_ms": ms("eval_duration"),
                       "total_ms": ms("total_duration")},
            "output": {"text": msg.get("content"), "tool_calls": _tool_calls(msg.get("tool_calls")),
                       "thinking": msg.get("thinking"), "finish_reason": response.get("done_reason"),
                       "n_tokens": response.get("eval_count")},
            "token_logprobs": tokens, "unavailable": unavailable}


def normalize_openai(request: dict, response: dict | None) -> dict:
    """OpenAI-compatible chat completions (Ollama /v1, or LiteLLM's openai/ route)."""
    unavailable = {"num_ctx": "not settable on /v1; see the engine snapshot", "load_ms": "not reported on /v1"}
    if response is None:
        return {"decoding": decoding_openai(request), "engine": None, "output": None, "token_logprobs": None,
                "unavailable": {"response": "request failed"}}
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = response.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        unavailable["cached_tokens"] = "usage.prompt_tokens_details.cached_tokens missing"
    lp = choice.get("logprobs")
    tokens = _tokens(lp.get("content")) if isinstance(lp, dict) else None
    if tokens is None:
        unavailable["token_logprobs"] = "not requested" if not request.get("logprobs") else "not returned"
    return {"decoding": decoding_openai(request),
            "engine": {"prompt_tokens": prompt, "cached_tokens": cached,
                       "computed_tokens": None if prompt is None or cached is None else prompt - cached,
                       "load_ms": None, "path_class": path_class(prompt, cached, None),
                       "prompt_eval_ms": None, "eval_ms": None, "total_ms": None},
            "output": {"text": msg.get("content"), "tool_calls": _tool_calls(msg.get("tool_calls")),
                       "thinking": msg.get("reasoning") or msg.get("reasoning_content"),
                       "finish_reason": choice.get("finish_reason"), "n_tokens": usage.get("completion_tokens")},
            "token_logprobs": tokens, "unavailable": unavailable}


def normalize_usage_only(request: dict, response: dict | None, reason: str) -> dict:
    """Routes that expose only text and token counts (e.g. LiteLLM's ollama_chat/)."""
    out = normalize_openai(request or {}, response)
    if isinstance((request or {}).get("options"), dict):  # a native body (LiteLLM's ollama_chat/ route)
        out["decoding"] = decoding_native(request)
    out["unavailable"].update({"token_logprobs": reason, "cached_tokens": reason})
    if out["engine"]:
        out["engine"].update(cached_tokens=None, computed_tokens=None, path_class="unknown")
    out["token_logprobs"] = None
    return out


NORMALIZERS = {"ollama": normalize_native, "openai_v1": normalize_openai}


def generation_key(model: str, messages, decoding: dict) -> dict:
    """The part of a request that determines generation (input_sha256 is its hash)."""
    dec = {k: decoding.get(k) for k in GENERATION_KEYS}
    dec.update({k: decoding[k] for k in EXTRA_GENERATION_KEYS if decoding.get(k) is not None})
    return {"model": model, "messages": messages or [], "decoding": dec}


def canonical_tool_ids(messages):
    """messages with tool-call ids renamed call_0, call_1, ... in order of first appearance (an assistant
    message's tool_calls[].id, a tool message's tool_call_id), so links between calls and results are kept.
    Ollama gives every tool call a random id that the next request sends back, so without this two runs of
    one agent episode never share a prompt from the second call on, and get other per-call seeds (a seeded T>0
    episode could not be repeated). Messages without ids are returned unchanged."""
    if not isinstance(messages, list):
        return messages
    ids: dict = {}

    def canon(i):
        return ids.setdefault(i, f"call_{len(ids)}")

    out = []
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("tool_calls"), list) and any(
                isinstance(c, dict) and c.get("id") is not None for c in m["tool_calls"]):
            m = {**m, "tool_calls": [{**c, "id": canon(c["id"])} if isinstance(c, dict) and c.get("id") is not None
                                     else c for c in m["tool_calls"]]}
        if isinstance(m, dict) and m.get("tool_call_id") is not None:
            m = {**m, "tool_call_id": canon(m["tool_call_id"])}
        out.append(m)
    return out


def prompt_key(model: str, messages, tools, format, think=None) -> dict:
    """What the model is shown, without the sampling settings (prompt_sha256 is its hash): the same prompt at
    T=0 and T=0.8, or with another seed, shares it, which input_sha256 does not. think changes the rendered
    prompt, so it is part of it. Tool-call ids are canonical (canonical_tool_ids): the server's random ids are not
    prompt content. input_sha256 keeps the exact messages."""
    return {"model": model, "messages": canonical_tool_ids(messages) or [], "tools": tools or None, "format": format,
            "think": think}
