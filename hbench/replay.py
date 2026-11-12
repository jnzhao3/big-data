"""Replay logged LLM calls under a chosen engine protocol and compare the outputs.

This is the measuring instrument behind E1 (how much does the computation path move outputs), E5 (how faithful
is a prefix replay) and E6 (does an intervention change that). A replay re-sends the exact logged request body,
optionally with sampler settings overridden (apply_overrides: T>0 resampling of logged requests); the protocol
only prepares the engine state:

    as_is            whatever state the server is in
    cold_start       unload the model once before the run (start_run does it), then replay the calls in order:
                     a run that itself started cold gets its whole lineage back (only a prefix 0..k of the
                     calls may be replayed)
    cold             unload the model before each call: full recompute from an empty prompt cache
    resend           send the same request once before: the measured call recomputes only its last token
    after_unrelated  send an unrelated short request first: the slot holds another prompt (RAM-cache restore)

compare() reports text identity, the first diverging token, logprob shifts at chosen tokens and over shared
candidates, and, when the source call carries a decision_spec (or a parser is registered for its caller), the
decision margins and whether the decision flipped: one row per decision the spec names (a tool call's name and
its arguments are separate decisions), or one row without decision fields when there is none.

Seeds (hbench.seeds): a sampled replay call is sent with its own per-call seed, by the rule natural runs use, so
calls are independent and `--seeds s` sends exactly the seeds a natural run with base seed s sent. Without
--seeds, a sampled source call is resent with its logged seed, and a greedy source call resampled at T>0 takes its
logged seed as the base.
"""

from __future__ import annotations

from pathlib import Path

import copy

from . import sampler
from .decisions import decision_margin, decision_parser, parser_from_spec
from .llm.client import LLMClient
from .llm.record import load_output, load_tokens
from .runlog.writer import BlobStore, read_events

PROTOCOLS = {"as_is": (), "cold_start": (), "cold": ("unload",), "resend": ("send_source",),
             "after_unrelated": ("send_unrelated",)}
RUN_PROTOCOL = {"cold_start": "cold_start", "cold": "cold"}  # the engine.protocol of the replay run (else natural)
OVERRIDABLE = {"ollama": ("temperature", "seed", "top_p", "top_k", "min_p"),
               "openai_v1": ("temperature", "seed", "top_p", "top_k")}
UNRELATED = [{"role": "user", "content": "Write one sentence about rivers."}]
_TRANSPORT = {"ollama": "ollama", "openai_v1": "openai_v1", "litellm/openai": "openai_v1",
              "litellm/ollama_chat": "ollama", "litellm/ollama": "ollama"}


def replay_seeds(calls: list[dict], blobs: BlobStore, seed: int | None, overrides: dict | None) -> list:
    """(seed to send or None to keep the logged one, its base or None) per source call, in order.

    seed: the replay's base seed (--seeds, or --set seed). A sampled call whose source seed was derived from
    the same base is sent its logged seed: exactly what the natural run sent. Other sampled calls get per-call
    seeds by the rule natural runs use, counting repeated prompts per process over the whole source run.
    Without a base, a sampled source call keeps its logged seed and a greedy one resampled at T>0 takes its
    logged seed as the base."""
    from .llm.record import prompt_sha256
    from .seeds import SeedDeriver, is_sampled
    derivers, out = {}, []
    for ev in calls:
        dec = ev.get("decoding") or {}
        temp = (overrides or {}).get("temperature", dec.get("temperature"))
        if not is_sampled(temp):
            out.append((seed, None))
            continue
        base = seed if seed is not None else None if is_sampled(dec.get("temperature")) else dec.get("seed")
        if base is None:
            out.append((None, None))
            continue
        d = derivers.setdefault(ev.get("proc"), SeedDeriver(ev.get("proc")))
        prompt = ev.get("prompt_sha256") or prompt_sha256(blobs.get(ev["request_raw"]), dec)
        derived = d.derive(base, ev.get("caller"), prompt)  # always counted, so k stays aligned with the source
        logged = dec.get("seed") if ev.get("seed_rule") and ev.get("seed_base") == base else None
        out.append((logged if logged is not None else derived, base))
    return out


def load_calls(run_dir) -> tuple[list[dict], BlobStore]:
    run_dir = Path(run_dir)
    calls = [e for e in read_events(run_dir) if e["type"] == "llm_call" and e.get("request_raw")]
    return calls, BlobStore(run_dir.parent / "_blobs")


def client_for(event: dict, log, base: str | None = None) -> LLMClient:
    transport = _TRANSPORT.get(event.get("transport"))
    if transport is None:
        raise ValueError(f"cannot replay transport {event.get('transport')!r}")
    return LLMClient(log, event.get("model"), transport=transport, base=base or api_base(event), top_logprobs=None,
                     decoding={}, per_call="none")  # sends logged bodies as they are


def api_base(event: dict) -> str:
    return (event.get("api_base") or "http://localhost:11434").removesuffix("/v1")


def apply_overrides(request: dict, transport: str, overrides: dict | None) -> dict:
    """The logged request with sampler settings replaced (a copy): in options on /api/chat, at the top level on
    /v1. Only sampler settings can be overridden; anything else would change what is being replayed."""
    body = copy.deepcopy(request)
    for k, v in (overrides or {}).items():
        allowed = OVERRIDABLE[_TRANSPORT[transport]]
        if k not in allowed:
            raise ValueError(f"cannot override {k!r} on {transport} (allowed: {', '.join(allowed)})")
        if _TRANSPORT[transport] == "ollama":
            body.setdefault("options", {})[k] = v
        else:
            body[k] = v
    return body


def replay_call(event: dict, blobs: BlobStore, client: LLMClient, protocol: str = "as_is", step: int | None = None,
                overrides: dict | None = None, seed_base: int | None = None):
    """seed_base: the base the per-call seed in overrides was derived from (replay_seeds), recorded with the
    call."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}")
    request = apply_overrides(blobs.get(event["request_raw"]), event.get("transport"), overrides)
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
    from .seeds import SEED_RULE
    extra = {"overrides": overrides} if overrides else {}
    if seed_base is not None:
        extra.update(seed_base=seed_base, seed_rule=SEED_RULE)
    return client.send(request, caller=f"replay:{protocol}", parent_id=event["span_id"], step=step,
                       replay_of=event["span_id"], protocol=protocol, **extra)


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


def compare(src: dict, rep: dict, blobs: BlobStore, lineage_source: dict | None = None,
            lineage_replay: dict | None = None) -> list[dict]:
    """Compare a replayed llm_call event with its source event: one row per decision of the source call. With the
    two calls' lineage records (engine_audit.json), lineage_same says whether bitwise identity is predicted."""
    ta, tb = load_tokens(src, blobs), load_tokens(rep, blobs)
    out_a, out_b = load_output(src, blobs) or {}, load_output(rep, blobs) or {}
    text_a, text_b = out_a.get("text"), out_b.get("text")
    known = all(x is not None and not x.get("uncertain") for x in (lineage_source, lineage_replay))
    row = {"source_span": src["span_id"], "replay_span": rep["span_id"], "protocol": rep.get("protocol"),
           "same_text": text_a == text_b,
           "path_source": (src.get("engine") or {}).get("path_class"),
           "path_replay": (rep.get("engine") or {}).get("path_class"),
           "lineage_source": (lineage_source or {}).get("canonical"),
           "lineage_replay": (lineage_replay or {}).get("canonical"),
           "lineage_same": lineage_source["fingerprint"] == lineage_replay["fingerprint"] if known else None,
           **compare_tokens(ta, tb)}
    parser = parser_from_spec(src.get("decision_spec")) or decision_parser(src.get("caller"))
    rows = []
    if parser and (text_a or out_a.get("tool_calls")):
        replayed = parser(out_b)
        greedy_a = (src.get("decoding") or {}).get("temperature") == 0  # T=0: a top candidate not chosen was ruled out
        greedy_b = (rep.get("decoding") or {}).get("temperature") == 0
        for d in parser(out_a):
            ma = decision_margin(ta, text_a, d["value"], decision=d, greedy=greedy_a)
            db = next((x for x in replayed if x["key"] == d["key"]), None)
            vb = None if db is None else db["value"]
            mb = decision_margin(tb, text_b, vb, decision=db, greedy=greedy_b) if db is not None else {"margin": None}
            out = {**row, "decision_key": d["key"], "decision": d["value"], "decision_replay": vb,
                   "decision_flipped": vb != d["value"], "margin_source": ma["margin"],
                   "margin_replay": mb["margin"],
                   "d_margin": None if ma["margin"] is None or mb["margin"] is None else mb["margin"] - ma["margin"]}
            if rep.get("overrides"):  # resampling: how likely the replay's sampler was to reproduce the source
                params = sampler.effective(rep.get("decoding"), sampler.endpoint_of(rep.get("transport"),
                                                                                    rep.get("endpoint")),
                                           model=rep.get("model_sampler", src.get("model_sampler")))
                out.update(overrides=rep["overrides"], sampling_change=out["decision_flipped"],
                           p_path_source=sampler.p_path(ta, ma.get("end"), params)["p"])
            rows.append(out)
    return rows or [row]
