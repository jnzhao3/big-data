"""llm_call events, and a logged client for direct Ollama /api/chat calls.

Each message is stored once in the blob store and referenced by hash, so the history an agent resends
at every step costs disk space only once. input_sha256 / output_sha256 let E5 find the first step where
two runs saw the same input but sampled different outputs.
"""

from __future__ import annotations

import json
import time
import urllib.request

from .ollama import DEFAULT_BASE
from .writer import RunLog, sha256_json

OLLAMA_TIMING = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count",
                 "eval_duration")


def message_refs(log: RunLog, messages) -> list[str]:
    return [log.blobs.put(m)["blob"] for m in messages or []]


def output_key(content, tool_calls) -> dict:
    """What counts as "the same output": text plus tool names and arguments (tool-call ids are random)."""
    calls = []
    for c in tool_calls or []:
        fn = c.get("function", c) if isinstance(c, dict) else {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append({"name": fn.get("name"), "arguments": args})
    return {"content": content or "", "tool_calls": calls}


def log_llm_call(log: RunLog, *, caller: str, provider: str, model: str, api_base: str | None, endpoint: str | None,
                 messages, params: dict, response: dict | None, usage: dict | None = None, latency_s: float | None = None,
                 request_body=None, error: str | None = None, parent_id: str | None = None, step: int | None = None,
                 **extra) -> str:
    content = (response or {}).get("content")
    tool_calls = (response or {}).get("tool_calls")
    return log.emit("llm_call", parent_id=parent_id, step=step, caller=caller, provider=provider, model=model,
                    api_base=api_base, endpoint=endpoint, params=params, messages=message_refs(log, messages),
                    input_sha256=sha256_json({"model": model, "messages": messages or [], "params": params}),
                    response=response, output_sha256=None if response is None else sha256_json(output_key(content, tool_calls)),
                    usage=usage, latency_s=latency_s, request_body=request_body, error=error, **extra)


def ollama_chat(body: dict, log: RunLog | None, caller: str, base: str = DEFAULT_BASE, step: int | None = None,
                parent_id: str | None = None, timeout: float = 600) -> dict:
    """POST {base}/api/chat (non-streaming), log the call, return Ollama's JSON reply. Errors are logged and
    re-raised."""
    body = dict(body, stream=False)
    req = urllib.request.Request(base.rstrip("/") + "/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    reply, error = None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.load(r)
        return reply
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        if log is not None:
            msg = (reply or {}).get("message") or {}
            params = {k: v for k, v in body.items() if k not in ("messages", "model", "stream")}
            response = None if reply is None else {
                "content": msg.get("content"), "tool_calls": msg.get("tool_calls"), "thinking": msg.get("thinking"),
                "done_reason": reply.get("done_reason"), "model": reply.get("model"),
                "created_at": reply.get("created_at")}
            usage = None if reply is None else {
                "prompt_tokens": reply.get("prompt_eval_count"), "completion_tokens": reply.get("eval_count")}
            log_llm_call(log, caller=caller, provider="ollama", model=body.get("model"), api_base=base,
                         endpoint="/api/chat", messages=body.get("messages"), params=params, response=response,
                         usage=usage, latency_s=round(time.time() - t0, 3), error=error, parent_id=parent_id,
                         step=step, ollama_timing=None if reply is None else {k: reply.get(k) for k in OLLAMA_TIMING})
