"""Decisions as first-class records: which tokens of an LLM output carry a method's decision, and how close
that decision was to flipping.

At T=0 an output changes only if some perturbation (a different computation lineage, a different prompt) moves
the logits by more than the gap at some position up to the end of the decision value: a token that flips before
the value changes everything after it. The margin of a decision is therefore

    margin = min over positions 0..end of the value of  (logprob(chosen token) - logprob(best other candidate))

a guarantee: perturbations smaller than it everywhere cannot change the decision (margin_def "prefix/1").
margin_value is the same minimum over the value's own tokens, the definition before E0 v3, kept as a diagnostic;
positions after the value (a reason text) never count. At T=0 the chosen token is the best one the server
allowed, so when a constraint such as a format=json grammar ruled out a better-scoring candidate, the gap is
taken to the best visible candidate below the chosen one (0 if there is none): never more than the real gap.

p_path_default gives, for each endpoint's default sampler, the probability that sampling reproduces the tokens
through the end of the value (hbench.sampler): what a T=0 run says about the same decision at default
temperature. Runs at T>0 also get p_path under their own sampler and greedy_path (every token up to the end
was the top candidate), which stability.sampling_check uses.

A decision is a value taken from the generated token stream, whichever channel carries it: the message text
(a JSON field, or the whole text) or a tool call (its name, an argument, or the whole call). A method states
where its decision is when it calls the model (LLMClient.chat(..., decision=spec)); the spec is stored on the
llm_call as `decision_spec`, so parser_from_spec() rebuilds the parser from the log alone. Parsers read the
normalized output ({"text", "tool_calls"}); decision_span() finds the value in the generated token text, where
a tool call appears as raw `<tool_call>{...}</tool_call>` text even though the message content is empty.
Callers that cannot attach a spec (LiteLLM) register a parser instead (register_decision_parser). evidence()
turns an LLM call into the `evidence` dict that prediction records carry (span_id, margin, ...), which the
scorers pass into item tables and stability_report stratifies flips by.
"""

from __future__ import annotations

import json
import re
from typing import Callable

_PARSERS: dict[str, Callable] = {}
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"


def _as_output(x) -> dict:
    """The normalized output of a call ({"text", "tool_calls"}); a bare string is a text-only output."""
    return x if isinstance(x, dict) else {"text": x, "tool_calls": None}


def _reads_outputs(fn):
    fn.takes_output = True  # parsers built here read the whole output; others get the text
    return fn


def register_decision_parser(caller: str, fn: Callable) -> None:
    """A parser for outputs of this caller: one built here (json_field_parser, tool_call_parser,
    parser_from_spec), which reads the whole output, or fn(text) -> [{"key", "value", "json_key"}, ...]."""
    _PARSERS[caller] = fn if getattr(fn, "takes_output", False) else (lambda out: fn(_as_output(out).get("text")))


def decision_parser(caller: str):
    return _PARSERS.get(caller)


def json_field_parser(field: str, key: str | None = None) -> Callable[[dict | str], list[dict]]:
    """Parser for outputs that are a JSON object with the decision in `field` (e.g. {"target": "bmi"})."""
    def parse(out) -> list[dict]:
        try:
            obj = json.loads(_as_output(out).get("text"))
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(obj, dict) or not isinstance(obj.get(field), str):
            return []
        return [{"key": key or field, "value": obj[field], "json_key": field}]
    return _reads_outputs(parse)


def tool_call_parser(field: str, index: int = 0, key: str | None = None) -> Callable[[dict | str], list[dict]]:
    """Parser for a tool call's name ("name"), one argument ("arguments.<path>", dots for nesting) or the whole
    call ("*", valued {"name", "arguments"}). Nothing is returned when the call or the argument is absent."""
    def parse(out) -> list[dict]:
        calls = _as_output(out).get("tool_calls") or []
        if index >= len(calls):
            return []
        call = calls[index]
        if field == "*":
            value, json_key = {"name": call.get("name"), "arguments": call.get("arguments")}, None
        elif field == "name":
            value, json_key = call.get("name"), "name"
        elif field.startswith("arguments."):
            value, path = call.get("arguments"), field.split(".")[1:]
            for k in path:
                if not isinstance(value, dict) or k not in value:
                    return []
                value = value[k]
            json_key = path[-1]
        else:
            raise ValueError(f"tool_call field must be name, arguments.<path> or *, not {field!r}")
        if value is None:
            return []
        return [{"key": key or (field if index == 0 else f"{field}@{index}"), "value": value, "json_key": json_key,
                 "channel": "tool_call", "index": index, "field": field}]
    return _reads_outputs(parse)


def parser_from_spec(spec: dict | list | None):
    """A parser from the decision spec stored in an llm_call (decision_spec), so any later tool can extract the
    decision from the log alone. Types: {"type": "json_field", "field": "target"}; {"type": "text"} (the whole
    stripped output is the decision); {"type": "tool_call", "field": "name" | "arguments.<path>" | "*",
    "index": 0}. A list of specs names several decisions of one call."""
    if not spec:
        return None
    if isinstance(spec, list):
        parts = [parser_from_spec(s) for s in spec]
        return _reads_outputs(lambda out: [d for p in parts for d in p(out)])
    if spec.get("type") == "json_field":
        return json_field_parser(spec["field"], spec.get("key"))
    if spec.get("type") == "text":
        def parse_text(out):
            text = _as_output(out).get("text")
            return [{"key": spec.get("key", "text"), "value": (text or "").strip(), "json_key": None}] if text else []
        return _reads_outputs(parse_text)
    if spec.get("type") == "tool_call":
        return tool_call_parser(spec["field"], spec.get("index", 0), spec.get("key"))
    raise ValueError(f"unknown decision spec type {spec.get('type')!r}")


def _value_after(text: str, value, pos: int) -> tuple[int, int] | None:
    """Characters of a JSON value written at or after pos: a string by its content, anything else by its JSON
    text (with or without spaces after separators, as the model may have written it)."""
    if isinstance(value, str):
        v = text.find('"' + value, pos)
        return (v + 1, v + 1 + len(value)) if v >= 0 else None
    for seps in ((", ", ": "), (",", ":")):
        lit = json.dumps(value, ensure_ascii=False, separators=seps)
        v = text.find(lit, pos)
        if v >= 0:
            return v, v + len(lit)
    return None


def _key_end(text: str, key: str, pos: int = 0) -> int | None:
    """Where the value of JSON key `key` starts searching: just after `"key":`. Requiring the colon keeps a
    string value that happens to equal the key from being taken for it."""
    m = re.compile(re.escape(json.dumps(key, ensure_ascii=False)) + r"\s*:").search(text, pos)
    return m.end() if m else None


def value_char_span(text: str, value: str, json_key: str | None = None) -> tuple[int, int] | None:
    """Character span of the decided value in the output. With json_key, the string value right after
    "json_key": is located (so a value that also appears in a reason text is not confused with it)."""
    if not text or value is None:
        return None
    if json_key is not None:
        k = _key_end(text, json_key)
        if k is not None:
            v = text.find('"' + value, k)  # the value's opening quote after the key
            if v >= 0:
                return v + 1, v + 1 + len(value)
    i = text.find(value)
    return (i, i + len(value)) if i >= 0 else None


def decision_span(token_logprobs: list, decision: dict) -> tuple[int, int] | None:
    """Character span of a decision (an item a parser returned) in the concatenated generated token text. A tool
    call is found in the index-th <tool_call> block: its name after "name", an argument after "arguments" and each
    key of its path, the whole call from the block's start to its closing tag (or the end of the text)."""
    joined = "".join(t[0] or "" for t in token_logprobs or [])
    if decision.get("channel") != "tool_call":
        value = decision.get("value")
        return value_char_span(joined, value, decision.get("json_key")) if isinstance(value, str) else None
    start = -1
    for _ in range(decision.get("index", 0) + 1):
        start = joined.find(TOOL_CALL_OPEN, start + 1)
        if start < 0:
            return None
    close = joined.find(TOOL_CALL_CLOSE, start)
    end = len(joined) if close < 0 else close + len(TOOL_CALL_CLOSE)
    block = joined[:end]
    field = decision.get("field")
    if field == "*":
        return start, end
    keys = ["name"] if field == "name" else ["arguments"] + field.split(".")[1:]
    pos = start
    for k in keys:
        pos = _key_end(block, k, pos)
        if pos is None:
            return None
    return _value_after(block, decision.get("value"), pos)


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


MARGIN_DEF = "prefix/1"
P_PATH_ENDPOINTS = ("/api/chat", "/v1/chat/completions")


def position_margin(entry) -> tuple[float | None, str | None, float | None]:
    """(chosen logprob - best other candidate's logprob, that candidate, its logprob) at one token position.
    None when the top-k list holds no other candidate (top_logprobs too small to tell)."""
    tok, lp, top = entry[0], entry[1], entry[2] or []
    others = [(c, clp) for c, clp in top if c != tok and clp is not None]
    if lp is None or not others:
        return None, None, None
    c, clp = max(others, key=lambda x: x[1])
    return lp - clp, c, clp


def _gap(entry, greedy: bool) -> tuple[float | None, bool, str | None, float | None]:
    """(gap, constrained, competitor, its logprob). With greedy (a T=0 call), a visible candidate scoring above
    the chosen token means a constraint ruled it out; the gap is then taken to the best one below (0 if none)."""
    tok, lp, top = entry[0], entry[1], entry[2] or []
    others = [(c, clp) for c, clp in top if c != tok and clp is not None]
    if lp is None or not others:
        return None, False, None, None
    c, clp = max(others, key=lambda x: x[1])
    if greedy and clp > lp:
        below = [(x, xlp) for x, xlp in others if xlp <= lp]
        if not below:
            return 0.0, True, None, None
        c, clp = max(below, key=lambda x: x[1])
        return lp - clp, True, c, clp
    return lp - clp, False, c, clp


def position_gap(entry, greedy: bool = True) -> tuple[float | None, bool]:
    """(gap, constrained) at one position, as the margin counts it."""
    return _gap(entry, greedy)[:2]


def _is_top(entry) -> bool:
    tok, lp, top = entry[0], entry[1], entry[2] or []
    return lp is not None and all(clp is None or clp <= lp for c, clp in top if c != tok)


def decision_margin(token_logprobs: list | None, text: str | None, value, json_key: str | None = None, *,
                    decision: dict | None = None, greedy: bool = True) -> dict:
    """Margin of one decided value in an output. Keys: found, span (the value's token indices), end, margin
    (prefix), margin_value, margin_pos, chosen_token, runner_up, runner_up_logprob, n_constrained, margin_def,
    aligned (concatenated tokens == text). With `decision` (an item a parser returned) the value is located
    through its channel, so tool-call decisions work too. greedy says the call ran at T=0."""
    if decision is not None:
        value, json_key = decision.get("value"), decision.get("json_key")
    out = {"value": value, "found": False, "span": [], "end": None, "margin": None, "margin_value": None,
           "margin_pos": None, "chosen_token": None, "runner_up": None, "runner_up_logprob": None,
           "n_constrained": None, "constrained_positions": [], "margin_def": MARGIN_DEF, "aligned": None}
    if not token_logprobs:
        out["reason"] = "no token logprobs"
        return out
    joined = "".join(t[0] or "" for t in token_logprobs)
    out["aligned"] = None if text is None else joined == text  # a tool-call reply may carry no text at all
    if decision is not None:
        span_chars = decision_span(token_logprobs, decision)
    else:  # offsets in the token text, which equals text if aligned
        span_chars = value_char_span(joined, value, json_key) if isinstance(value, str) else None
    span = token_span(token_logprobs, *span_chars) if span_chars is not None else []
    if not span:
        out["reason"] = "value not found in the output tokens"
        return out
    end = max(span)
    gaps = [_gap(token_logprobs[i], greedy) for i in range(end + 1)]
    constrained = [i for i, g in enumerate(gaps) if g[1]]
    out.update(found=True, span=span, end=end, n_constrained=len(constrained), constrained_positions=constrained)
    prefix = [(g[0], i) for i, g in enumerate(gaps) if g[0] is not None]
    if not prefix:
        out["reason"] = "no alternative candidates in top_logprobs"
        return out
    m, i = min(prefix)
    in_value = [gaps[j][0] for j in span if gaps[j][0] is not None]
    out.update(margin=m, margin_value=min(in_value) if in_value else None, margin_pos=i,
               chosen_token=token_logprobs[i][0], runner_up=gaps[i][2], runner_up_logprob=gaps[i][3])
    return out


def evidence(span_id: str | None, token_logprobs: list | None, text: str | None, value, json_key: str | None = None,
             *, decision: dict | None = None, temperature: float | None = None, sampler_params: dict | None = None,
             lock: dict | None = None, grammar: bool = False, **extra) -> dict:
    """The `evidence` dict for a prediction record: which call decided it and how close the decision was.
    temperature is what the call asked for (0 means greedy, so constrained positions can be recognised);
    sampler_params is the sampler it ran with (hbench.sampler.effective), used when it is above temperature 0.
    grammar: the call was constrained (format or a JSON schema). Candidates scoring above a T=0 call's choice are
    then left out where it shows they were ruled out, and every p_path is marked a lower bound: a constraint only
    removes competitors, and those it removed elsewhere are not visible."""
    from . import sampler

    ev = {"span_id": span_id, **extra}
    if value is None and decision is None:
        return ev
    d = decision_margin(token_logprobs, text, value, json_key, decision=decision, greedy=temperature == 0)
    ev.update(margin=d["margin"], margin_value=d["margin_value"], margin_def=MARGIN_DEF, margin_pos=d["margin_pos"],
              runner_up=d["runner_up"], margin_found=d["found"], margin_note=d.get("reason"),
              n_constrained=d["n_constrained"])
    end = d["end"]
    if end is None:
        return ev
    constrained = set(d["constrained_positions"])

    def bound(r: dict) -> dict:
        out = {"p": r["p"], "exact": r["exact"] and not grammar, **({"reason": r["reason"]} if r["reason"] else {})}
        return {**out, "lower_bound": True} if grammar else out

    ev["p_path_default"] = {}
    for ep in P_PATH_ENDPOINTS:
        params = sampler.defaults(ep, lock)
        r = (sampler.p_path(token_logprobs, end, params, constrained) if params else
             {"p": None, "exact": False, "reason": f"no sampler_defaults for {ep} in env/engine-lock.json"})
        ev["p_path_default"][ep] = bound(r)
    if sampler_params and (sampler_params.get("temp") or 0) > 0:
        r = bound(sampler.p_path(token_logprobs, end, sampler_params, constrained))
        ev.update(p_path=r["p"], p_path_exact=r["exact"], greedy_path=all(_is_top(t) for t in token_logprobs[: end + 1]),
                  **({"p_path_note": r["reason"]} if r.get("reason") else {}),
                  **({"p_path_lower_bound": True} if grammar else {}))
    return ev


def call_evidence(normalized: dict, spec=None, *, transport: str | None = None, endpoint: str | None = None,
                  span_id: str | None = None, prompt_sha256: str | None = None, key: str | None = None,
                  caller: str | None = None, lock: dict | None = None, **extra) -> dict:
    """evidence() for one decision of a call, taken entirely from what the call recorded: the decision from its
    spec (or the parser registered for its caller; the first one, or the one named `key`), the greedy flag from
    its temperature, the sampler from its decoding over its endpoint's defaults, grammar from its format. This
    is what LLMResult.evidence() and evidence_from_call() return, so a method never restates any of it."""
    from . import sampler

    out = normalized.get("output") or {}
    parser = parser_from_spec(spec) or (decision_parser(caller) if caller else None)
    d = next((x for x in (parser(out) if parser else []) if key is None or x["key"] == key), None)
    if prompt_sha256:
        extra = {"prompt_sha256": prompt_sha256, **extra}
    if d is None:
        return {"span_id": span_id, **extra, "margin": None, "margin_found": False,
                "margin_note": "no decision found in the output" if parser else "no decision spec for this call"}
    dec = normalized.get("decoding") or {}
    return evidence(span_id, normalized.get("token_logprobs"), out.get("text"), d["value"], decision=d,
                    temperature=dec.get("temperature"),
                    sampler_params=sampler.effective(dec, sampler.endpoint_of(transport, endpoint), lock), lock=lock,
                    grammar=dec.get("format") is not None, **extra)


def evidence_from_call(event: dict, blobs, key: str | None = None, lock: dict | None = None, **extra) -> dict:
    """The same evidence from a logged llm_call, e.g. one LiteLLM made (runlog.litellm_cb logs it)."""
    from .llm.record import load_tokens

    normalized = {"output": event.get("output"), "decoding": event.get("decoding"),
                  "token_logprobs": load_tokens(event, blobs)}
    return call_evidence(normalized, event.get("decision_spec"), transport=event.get("transport"),
                         endpoint=event.get("endpoint"), span_id=event.get("span_id"),
                         prompt_sha256=event.get("prompt_sha256"), key=key, caller=event.get("caller"), lock=lock,
                         **extra)


def evidence_columns(ev: dict | None) -> dict:
    """Decision evidence carried into item tables: the deciding call, how close the decision was, and how likely
    sampling is to reproduce it. A v2 record (no margin_def) keeps its margin; it has no margin_value."""
    ev = ev or {}
    ppd = ev.get("p_path_default") or {}
    api, v1 = ppd.get("/api/chat") or {}, ppd.get("/v1/chat/completions") or {}
    return {"evidence_span": ev.get("span_id"), "margin": ev.get("margin"), "margin_value": ev.get("margin_value"),
            "margin_def": ev.get("margin_def"), "n_constrained": ev.get("n_constrained"),
            "p_path_api_chat": api.get("p"), "p_path_api_chat_exact": api.get("exact"),
            "p_path_v1": v1.get("p"), "p_path_v1_exact": v1.get("exact"),
            "p_path": ev.get("p_path"), "p_path_exact": ev.get("p_path_exact"),
            "p_path_lower_bound": bool(api.get("lower_bound") or ev.get("p_path_lower_bound")) if ppd else None,
            "greedy_path": ev.get("greedy_path"), "prompt_sha256": ev.get("prompt_sha256")}
