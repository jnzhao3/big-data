"""Decisions as first-class records: which tokens of an LLM output carry a method's decision, and how close
that decision was to flipping.

At T=0 an output changes only if some perturbation (a different computation path, a different prompt) moves
logits by more than the margin at a decision token. The margin of a decision is therefore

    margin = min over the decision's token span of  (logprob(chosen token) - logprob(best other candidate))

computed on the span of the decided value only. The minimum over the whole output is the wrong quantity: it
lands on punctuation or explanation tokens unrelated to the decision (E1 pilot: 0.1145 on a reasoning token
versus 0.1196 on the decision). p_flip(T) estimates, from the same token distribution, how often sampling at
temperature T would pick another token at the margin position.

A method states where its decision is when it calls the model (LLMClient.chat(..., decision=spec)); the spec is
stored on the llm_call as `decision_spec`, so parser_from_spec() rebuilds the parser from the log alone. Callers
that cannot attach a spec (LiteLLM) register a parser instead (register_decision_parser). evidence() turns an
LLM call into the `evidence` dict that prediction records carry (span_id, margin, ...), which the scorers pass
into item tables and stability_report stratifies flips by.
"""

from __future__ import annotations

import json
import math
from typing import Callable

_PARSERS: dict[str, Callable] = {}


def register_decision_parser(caller: str, fn: Callable[[str], list[dict]]) -> None:
    """fn(text) -> [{"key": ..., "value": ..., "json_key": ... or None}, ...] for outputs of this caller."""
    _PARSERS[caller] = fn


def decision_parser(caller: str):
    return _PARSERS.get(caller)


def json_field_parser(field: str, key: str | None = None) -> Callable[[str], list[dict]]:
    """Parser for outputs that are a JSON object with the decision in `field` (e.g. {"target": "bmi"})."""
    def parse(text: str) -> list[dict]:
        try:
            obj = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(obj, dict) or not isinstance(obj.get(field), str):
            return []
        return [{"key": key or field, "value": obj[field], "json_key": field}]
    return parse


def parser_from_spec(spec: dict | None):
    """A parser from the decision spec stored in an llm_call (decision_spec), so any later tool can extract the
    decision from the log alone. Types: {"type": "json_field", "field": "target"}; {"type": "text"} (the whole
    stripped output is the decision)."""
    if not spec:
        return None
    if spec.get("type") == "json_field":
        return json_field_parser(spec["field"], spec.get("key"))
    if spec.get("type") == "text":
        return lambda text: [{"key": spec.get("key", "text"), "value": (text or "").strip(), "json_key": None}] if text else []
    raise ValueError(f"unknown decision spec type {spec.get('type')!r}")


def value_char_span(text: str, value: str, json_key: str | None = None) -> tuple[int, int] | None:
    """Character span of the decided value in the output. With json_key, the string value right after
    "json_key": is located (so a value that also appears in a reason text is not confused with it)."""
    if not text or value is None:
        return None
    if json_key is not None:
        k = text.find(json.dumps(json_key, ensure_ascii=False))
        if k >= 0:
            v = text.find('"' + value, k + len(json_key) + 2)  # the value's opening quote after the key
            if v >= 0:
                return v + 1, v + 1 + len(value)
    i = text.find(value)
    return (i, i + len(value)) if i >= 0 else None


def token_span(tokens: list, start: int, end: int) -> list[int]:
    """Indices of the tokens overlapping characters [start, end) of the concatenated token text."""
    out, pos = [], 0
    for i, t in enumerate(tokens):
        piece = t[0] if isinstance(t, (list, tuple)) else t
        a, b = pos, pos + len(piece or "")
        pos = b
        if b > start and a < end:
            out.append(i)
    return out


def position_margin(entry) -> tuple[float | None, str | None, float | None]:
    """(chosen logprob - best other candidate's logprob, that candidate, its logprob) at one token position.
    None when the top-k list holds no other candidate (top_logprobs too small to tell)."""
    tok, lp, top = entry[0], entry[1], entry[2] or []
    others = [(c, clp) for c, clp in top if c != tok and clp is not None]
    if lp is None or not others:
        return None, None, None
    c, clp = max(others, key=lambda x: x[1])
    return lp - clp, c, clp


def p_flip(entry, temperature: float) -> float | None:
    """Probability that sampling at this position picks another token, from the visible top-k only
    (renormalized). 0 at T=0 unless there is an exact tie."""
    tok, lp, top = entry[0], entry[1], entry[2] or []
    cands = {c: clp for c, clp in top if clp is not None}
    if lp is None or not cands:
        return None
    cands.setdefault(tok, lp)
    if temperature <= 0:
        best = max(cands.values())
        return 0.0 if cands[tok] >= best else 1.0
    z = max(cands.values())
    w = {c: math.exp((v - z) / temperature) for c, v in cands.items()}
    return 1.0 - w[tok] / sum(w.values())


def decision_margin(token_logprobs: list | None, text: str | None, value: str, json_key: str | None = None) -> dict:
    """Margin of one decided value in an output. Keys: found, span, margin, margin_pos, chosen_token,
    runner_up, runner_up_logprob, aligned (concatenated tokens == text)."""
    out = {"value": value, "found": False, "span": [], "margin": None, "margin_pos": None, "chosen_token": None,
           "runner_up": None, "runner_up_logprob": None, "aligned": None}
    if not token_logprobs or text is None:
        out["reason"] = "no token logprobs"
        return out
    joined = "".join(t[0] or "" for t in token_logprobs)
    out["aligned"] = joined == text
    span_chars = value_char_span(joined, value, json_key)  # offsets in the token text, which equals text if aligned
    if span_chars is None:
        out["reason"] = "value not found in the output tokens"
        return out
    span = token_span(token_logprobs, *span_chars)
    out.update(found=True, span=span)
    best = None
    for i in span:
        m, c, clp = position_margin(token_logprobs[i])
        if m is not None and (best is None or m < best[0]):
            best = (m, i, c, clp)
    if best is None:
        out["reason"] = "no alternative candidates in top_logprobs"
        return out
    m, i, c, clp = best
    out.update(margin=m, margin_pos=i, chosen_token=token_logprobs[i][0], runner_up=c, runner_up_logprob=clp)
    return out


def evidence(span_id: str | None, token_logprobs: list | None, text: str | None, value: str | None,
             json_key: str | None = None, temperature: float | None = None, **extra) -> dict:
    """The `evidence` dict for a prediction record: which call decided it and how close the decision was."""
    ev = {"span_id": span_id, **extra}
    if value is None:
        return ev
    d = decision_margin(token_logprobs, text, value, json_key)
    ev.update(margin=d["margin"], runner_up=d["runner_up"], margin_pos=d["margin_pos"],
              margin_found=d["found"], margin_note=d.get("reason"))
    if d["margin_pos"] is not None and temperature is not None:
        ev["p_flip"] = p_flip(token_logprobs[d["margin_pos"]], temperature)
    return ev
