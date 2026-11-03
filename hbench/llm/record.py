"""The llm_call event (schema hbench.llm_call/3): raw payloads kept losslessly, normalized fields derived.

Raw request and response are always stored as blobs, so any field a later analysis needs can be derived
again without rerunning (`hbench llm rederive`). Messages are also stored one blob per message, so the
history an agent resends at every step costs disk space once.

/3 adds prompt_sha256 (the prompt without sampling settings) and parses tool-call arguments into objects on
both transports. Readers accept /2 events, which lack prompt_sha256.
"""

from __future__ import annotations

from ..runlog.writer import RunLog, sha256_json
from .normalize import NORMALIZERS, generation_key, prompt_key

LLM_CALL_SCHEMA = "hbench.llm_call/3"


def prompt_sha256(request: dict | None, decoding: dict | None) -> str | None:
    request = request or {}
    if not request.get("messages"):
        return None
    return sha256_json(prompt_key(request.get("model"), request.get("messages"), request.get("tools"),
                                  (decoding or {}).get("format"), (decoding or {}).get("think")))


def record_llm_call(log: RunLog, *, caller: str, transport: str, endpoint: str | None, api_base: str | None,
                    model: str | None, request: dict | None, response: dict | None, normalized: dict,
                    latency_s: float | None, error: str | None = None, parent_id: str | None = None,
                    step: int | None = None, **extra) -> str:
    request = request or {}
    messages = request.get("messages")
    decoding = normalized.get("decoding") or {}
    output = normalized.get("output")
    tokens = normalized.get("token_logprobs")
    return log.emit(
        "llm_call", parent_id=parent_id, step=step, llm_schema=LLM_CALL_SCHEMA, caller=caller, transport=transport,
        endpoint=endpoint, api_base=api_base, model=model,
        request_raw=log.blobs.put(request) if request else None,
        response_raw=log.blobs.put(response) if response is not None else None,
        messages=[log.blobs.put(m)["blob"] for m in messages or []],
        decoding=decoding, engine=normalized.get("engine"), output=output,
        token_logprobs=log.blobs.put(tokens) if tokens is not None else None,
        input_sha256=sha256_json(generation_key(model, messages, decoding)),
        prompt_sha256=prompt_sha256(request, decoding),
        request_sha256=sha256_json(request) if request else None,
        output_sha256=None if output is None else sha256_json(
            {"text": output.get("text") or "", "tool_calls": output.get("tool_calls") or []}),
        logprobs_sha256=None if tokens is None else sha256_json(tokens),
        unavailable=normalized.get("unavailable") or {}, latency_s=latency_s, error=error, **extra)


def rederive(event: dict, blobs) -> dict:
    """Normalized fields recomputed from the raw payloads of a stored llm_call (for schema upgrades)."""
    norm = NORMALIZERS.get(event.get("transport"))
    if norm is None or not event.get("request_raw"):
        return {"span_id": event["span_id"], "rederived": False, "reason": f"no normalizer for {event.get('transport')}"}
    request = blobs.get(event["request_raw"])
    response = blobs.get(event["response_raw"]) if event.get("response_raw") else None
    return {"span_id": event["span_id"], "rederived": True, **norm(request, response)}


def load_tokens(event: dict, blobs) -> list | None:
    ref = event.get("token_logprobs")
    return None if ref is None else (blobs.get(ref) if isinstance(ref, dict) else ref)
