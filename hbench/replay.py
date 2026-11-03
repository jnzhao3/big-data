"""Replay logged LLM calls under a chosen engine protocol and compare the outputs.

This is the measuring instrument behind E1 (how much does the computation path move outputs), E5 (how faithful
is a prefix replay) and E6 (does an intervention change that). A replay re-sends the exact logged request body;
the protocol only prepares the engine state first:

    as_is            whatever state the server is in
    cold             unload the model first: full recompute from an empty prompt cache
    resend           send the same request once before: the measured call recomputes only its last token
    after_unrelated  send an unrelated short request first: the slot holds another prompt (RAM-cache restore)

compare() reports text identity, the first diverging token, logprob shifts at chosen tokens and over shared
candidates, and, when the source call carries a decision_spec (or a parser is registered for its caller), the
decision margins and whether the decision flipped.
"""

from __future__ import annotations

from pathlib import Path

from .decisions import decision_margin, decision_parser, parser_from_spec
from .llm.client import LLMClient
from .llm.record import load_tokens
from .runlog.writer import BlobStore, read_events

PROTOCOLS = {"as_is": (), "cold": ("unload",), "resend": ("send_source",), "after_unrelated": ("send_unrelated",)}
UNRELATED = [{"role": "user", "content": "Write one sentence about rivers."}]
_TRANSPORT = {"ollama": "ollama", "openai_v1": "openai_v1", "litellm/openai": "openai_v1",
              "litellm/ollama_chat": "ollama", "litellm/ollama": "ollama"}


def load_calls(run_dir) -> tuple[list[dict], BlobStore]:
    run_dir = Path(run_dir)
    calls = [e for e in read_events(run_dir) if e["type"] == "llm_call" and e.get("request_raw")]
    return calls, BlobStore(run_dir.parent / "_blobs")


def client_for(event: dict, log, base: str | None = None) -> LLMClient:
    transport = _TRANSPORT.get(event.get("transport"))
    if transport is None:
        raise ValueError(f"cannot replay transport {event.get('transport')!r}")
    api = base or (event.get("api_base") or "http://localhost:11434").removesuffix("/v1")
    return LLMClient(log, event.get("model"), transport=transport, base=api, top_logprobs=None)


def replay_call(event: dict, blobs: BlobStore, client: LLMClient, protocol: str = "as_is", step: int | None = None):
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}")
    request = blobs.get(event["request_raw"])
    client.model = request.get("model", client.model)  # the bare model name, also used to unload it
    for action in PROTOCOLS[protocol]:
        if action == "unload":
            client.unload()
        elif action == "send_source":
            client.send(request, caller="replay:pre", parent_id=event["span_id"])
        elif action == "send_unrelated":
            body = client.build_request(UNRELATED, temperature=0, max_tokens=8)
            if client.transport == "ollama" and "options" in request:
                body["options"]["num_ctx"] = request["options"].get("num_ctx", body["options"].get("num_ctx"))
            client.send(body, caller="replay:pre")
    return client.send(request, caller=f"replay:{protocol}", parent_id=event["span_id"], step=step,
                       replay_of=event["span_id"], protocol=protocol)


def compare_tokens(a: list | None, b: list | None) -> dict:
    """Token-level comparison of two generations of the same request."""
    if a is None or b is None:
        return {"comparable": False}
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x[0] != y[0]), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    upto = first if first is not None else len(a)
    d_chosen = [abs(x[1] - y[1]) for x, y in zip(a[:upto], b[:upto]) if x[1] is not None and y[1] is not None]
    d_any = []
    for x, y in zip(a[:upto], b[:upto]):
        tb = dict((c, lp) for c, lp in y[2] or [])
        d_any += [abs(lp - tb[c]) for c, lp in x[2] or [] if c in tb and lp is not None and tb[c] is not None]
    return {"comparable": True, "same_tokens": first is None, "first_divergent_token": first,
            "bitwise_same_logprobs": a == b, "max_abs_dlogprob_chosen": max(d_chosen, default=0.0),
            "max_abs_dlogprob_any": max(d_any, default=0.0)}


def compare(src: dict, rep: dict, blobs: BlobStore) -> dict:
    """Compare a replayed llm_call event with its source event."""
    ta, tb = load_tokens(src, blobs), load_tokens(rep, blobs)
    text_a = (src.get("output") or {}).get("text")
    text_b = (rep.get("output") or {}).get("text")
    row = {"source_span": src["span_id"], "replay_span": rep["span_id"], "protocol": rep.get("protocol"),
           "same_text": text_a == text_b,
           "path_source": (src.get("engine") or {}).get("path_class"),
           "path_replay": (rep.get("engine") or {}).get("path_class"), **compare_tokens(ta, tb)}
    parser = parser_from_spec(src.get("decision_spec")) or decision_parser(src.get("caller"))
    if parser and text_a:
        for d in parser(text_a):
            ma = decision_margin(ta, text_a, d["value"], d.get("json_key"))
            db = parser(text_b or "")
            vb = next((x["value"] for x in db if x["key"] == d["key"]), None)
            mb = decision_margin(tb, text_b, vb, d.get("json_key")) if vb is not None else {"margin": None}
            row.update({"decision": d["value"], "decision_replay": vb, "decision_flipped": vb != d["value"],
                        "margin_source": ma["margin"], "margin_replay": mb["margin"],
                        "d_margin": None if ma["margin"] is None or mb["margin"] is None else mb["margin"] - ma["margin"]})
            break  # one decision per call for now (the first the parser names)
    return row
