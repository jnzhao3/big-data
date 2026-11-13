"""E1 Stage 3 driver: resample logged T=0 calls at T>0 and compare each sample with its greedy source.

One resampling pass = one hbench run. It sends a fixed list of logged requests (from any runs) with sampler
settings overridden and one seed, under one protocol:

    cold_start  unload once, then send the calls in order. Each prompt differs from the ones before it, so every
                call is computed in batches: the lineage of a T=0 source that ran batched (P2: batched computation
                does not depend on how the batches split).
    resend      send each request once before the measured one: the measured call recomputes only its last token
                (the single-token lineage), with the same seed.

The sources' greedy outputs and logprobs give each decision's predicted probability of being reproduced by the
sampler (hbench.sampler.p_path through the end of the decision value). A sample "keeps the path" when its tokens
equal the source's through that position, and "keeps the value" when the decision's value is the same. The path
is what p_path predicts; the value can also be reached by another path, so it is kept at least as often.
For agent calls p_path is near 0 at T=1 (it includes the free reasoning text before the tool call), so each
decision also gets p_value_local, the probability of the value's own tokens given the greedy prefix: a forecast
of a value change, not a bound.

Each call gets its own seed, derived from the pass's seed and the source (call_seed below; replay_call does not derive seeds itself). llama-server
reseeds its sampler per request, so with one seed for every call all calls would use the same random draws:
decisions at the same token position (clark's value starts at token 4) would all be sampled with one number.

A source can lend its decision spec to another call (spec_from): a per-call replay of an episode carries no spec
of its own, so its calls are compared with the spec of the episode call they replayed.

Usage (normally from run_plan.py):
    python -I resample.py <selection.json> --seed 1 --protocol cold_start --set temperature=0.8 --out <raw dir>
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from hbench import sampler  # noqa: E402
from hbench.sampler import header_model_samplers, lookup_model_sampler  # noqa: E402
from hbench.decisions import decision_margin, decision_parser, parser_from_spec  # noqa: E402
from hbench.design import make_condition  # noqa: E402
from hbench.engine.lineage import _audit_rows  # noqa: E402
from hbench.llm.record import load_tokens  # noqa: E402
from hbench.replay import _TRANSPORT, RUN_PROTOCOL, api_base, client_for, compare, replay_call  # noqa: E402
from hbench.runlog.session import start_run  # noqa: E402
from hbench.runlog.writer import BlobStore, is_blob_ref, read_events  # noqa: E402

PROTOCOLS = ("cold_start", "resend")


def call_seed(seed: int, *parts) -> int:
    """The seed one resampled call is sent with, derived from the pass's seed and the source. replay_call sends
    seeds exactly as given, so a pass with one seed for all calls would sample every call with the same draws."""
    h = hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


class Blobs:
    """Read blobs from whichever store holds them (sources and samples live under different roots)."""

    def __init__(self, *stores):
        self.stores = stores

    def get(self, ref):
        for s in self.stores:
            try:
                return s.get(ref)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(ref)


_RUNS: dict = {}


def read_run_events(run_dir) -> list[dict]:
    """read_events with llm_call outputs inlined: the writer stores any field over INLINE_LIMIT as a blob ref, and a
    long thinking answer makes the output that large."""
    run_dir = Path(run_dir)
    events, blobs = read_events(run_dir), BlobStore(run_dir.parent / "_blobs")
    for e in events:
        if e.get("type") == "llm_call" and is_blob_ref(e.get("output")):
            e["output"] = blobs.get(e["output"])
    return events


def run_events(run_dir) -> dict:
    """span_id -> event of one run (cached), plus the run's blob store and lineage records."""
    run_dir = str(Path(run_dir).resolve())
    if run_dir not in _RUNS:
        events = read_run_events(run_dir)
        header = next((e for e in events if e.get("type") == "run_header"), None)
        _RUNS[run_dir] = {"events": {e["span_id"]: e for e in events if e.get("type") == "llm_call"},
                          "blobs": BlobStore(Path(run_dir).parent / "_blobs"), "lineage": _audit_rows(Path(run_dir)),
                          "samplers": header_model_samplers(header)}
    return _RUNS[run_dir]


def model_sampler_of(ev: dict, run_dir=None) -> dict | None:
    """The model's own sampler settings for a logged call: its model_sampler field (calls logged from hbench
    49eae7b on), else the run header's engine snapshot (earlier runs). Without them hbench predicts the
    endpoint's server defaults, which a model with its own Modelfile settings (qwen3.8-27b-bdikit: temperature
    1, top_k 20, top_p 0.95) does not run with. None when unknown; qwen2.5:14b sets none ({})."""
    if "model_sampler" in ev:
        return ev["model_sampler"]
    if run_dir is None:
        return None
    return lookup_model_sampler(run_events(run_dir)["samplers"], ev.get("model"))


def source_event(src: dict) -> tuple[dict, BlobStore, dict | None]:
    """A selection entry {"run", "span_id", optional "spec_from": {"run", "span_id"}} -> (event, blobs, lineage).
    With spec_from the event is compared with that call's decision spec and caller."""
    r = run_events(src["run"])
    ev = dict(r["events"][src["span_id"]])
    if src.get("spec_from"):
        orig = run_events(src["spec_from"]["run"])["events"][src["spec_from"]["span_id"]]
        ev.update(decision_spec=orig.get("decision_spec"), caller=orig.get("caller"))
    return ev, r["blobs"], r["lineage"].get(src["span_id"])


def selection_sha(sources: list[dict]) -> str:
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:12]


def decision_ends(ev: dict, blobs) -> dict:
    """decision key -> (index of the value's last token, p_path inputs) for a greedy source call."""
    parser = parser_from_spec(ev.get("decision_spec")) or decision_parser(ev.get("caller"))
    out_a = ev.get("output") or {}
    if parser is None or not (out_a.get("text") or out_a.get("tool_calls")):
        return {}
    tokens = load_tokens(ev, blobs)
    return {d["key"]: decision_margin(tokens, out_a.get("text"), d["value"], decision=d, greedy=True)
            for d in parser(out_a)}


def p_value_local(tokens, m: dict, params: dict) -> float | None:
    """Probability that the sampler picks the greedy tokens of the value span itself, given the greedy prefix
    before it. In an agent's reply the prefix is free reasoning text, so p_path (which includes it) is near 0 at
    T=1 and says nothing about the decision; this local probability is the natural forecast of a value change.
    It is a heuristic, not a bound: a sample that wrote another prefix decides in another context."""
    span = m.get("span") or []
    if not tokens or not span or sampler.unsupported(params):
        return None
    out = 1.0
    for i in span:
        k, _ = sampler.p_keep(tokens[i], params, ruled_out_above=i in (m.get("constrained_positions") or ()))
        if k is None:
            return None
        out *= k
    return out


TARGET_KEYS = ("target", "text_target")


def tool_spec(ev: dict) -> bool:
    """Whether a call's decisions come from tool calls (a tool loop), not from a single JSON answer."""
    spec = ev.get("decision_spec")
    return isinstance(spec, list) and any(isinstance(d, dict) and d.get("type") == "tool_call" for d in spec)


def submitted(rep_ev: dict, src_ev: dict):
    """The target a sample submitted: from its submit_match call at any index, else from JSON text. Parallel
    calls get index-suffixed keys (target@1), so matching keys would count a reordering as a changed answer."""
    parser = parser_from_spec(src_ev.get("decision_spec")) or decision_parser(src_ev.get("caller"))
    found = {d["key"]: d["value"] for d in (parser(rep_ev.get("output") or {}) if parser else [])}
    for k in sorted(found):
        if k.split("@")[0] in TARGET_KEYS:
            i = k.partition("@")[2]
            if k.startswith("text_target") or found.get("tool" + (f"@{i}" if i else "")) == "submit_match":
                return found[k]
    return None


def path_kept(same_tokens, first_divergent, end) -> bool | None:
    """Did a sample reproduce the source's tokens through position end (the decision value's last token)?"""
    if end is None or same_tokens is None:
        return None
    return bool(same_tokens or (first_divergent is not None and first_divergent > end))


def path_rows(src: dict, src_ev: dict, rep_ev: dict, blobs, lin_src, lin_rep) -> list[dict]:
    """compare() rows for one sample, with the path-level outcome and the exactness of p_path added."""
    rows = compare(src_ev, rep_ev, blobs, lin_src, lin_rep)
    ends = decision_ends(src_ev, blobs)
    tokens = load_tokens(src_ev, blobs)
    ms = model_sampler_of(rep_ev) if "model_sampler" in rep_ev else model_sampler_of(src_ev, src["run"])
    params = sampler.effective(rep_ev.get("decoding"), sampler.endpoint_of(rep_ev.get("transport"),
                                                                           rep_ev.get("endpoint")), model=ms)
    first = rows[0].get("first_divergent_token") if rows else None
    out = []
    for row in rows:
        m = ends.get(row.get("decision_key")) or {}
        end = m.get("end")
        pp = sampler.p_path(tokens, end, params, constrained=m.get("constrained_positions") or ())
        same = row.get("same_tokens")
        vkept = None if row.get("decision_flipped") is None else not row["decision_flipped"]
        if tool_spec(src_ev) and str(row.get("decision_key")).split("@")[0] in TARGET_KEYS:
            got = submitted(rep_ev, src_ev)  # the answer, wherever the sample put its submit call
            row = {**row, "decision_replay": got}
            vkept = got == row.get("decision")
        rep_out = rep_ev.get("output") or {}
        out.append({**row, "source_run": Path(src["run"]).name, "seed": params.get("seed"),
                    "temp": params.get("temp"), "top_p": params.get("top_p"), "top_k": params.get("top_k"),
                    "end": end, "p_path": pp["p"], "p_exact": pp["exact"],
                    "p_value_local": p_value_local(tokens, m, params),
                    "path_kept": path_kept(same, first, end),
                    "value_kept": vkept,
                    # a value change with no value at all: a thinking model can stop after its reasoning
                    "no_answer": row.get("decision_replay") is None,
                    "empty_output": not (rep_out.get("text") or rep_out.get("tool_calls")),
                    "model_sampler": json.dumps(ms, sort_keys=True) if ms is not None else None,
                    "n_tokens_source": len(tokens or []), "constrained": bool(m.get("n_constrained"))})
    return out


def run_pass(sources: list[dict], *, protocol: str, overrides: dict, seed: int, raw_dir, label: str,
             block: str | None = None, order: int | None = None) -> dict:
    """Send every source once with the overrides and this seed; one hbench run. Returns a summary."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    first, _, _ = source_event(sources[0])
    blobs0 = run_events(sources[0]["run"])["blobs"]
    model = blobs0.get(first["request_raw"]).get("model")
    transport = _TRANSPORT.get(first.get("transport"), first.get("transport"))
    if any(_TRANSPORT.get(source_event(s)[0].get("transport")) != transport for s in sources):
        raise ValueError("a pass sends through one transport: split the selection")
    ov = {**overrides, "seed": seed}
    cond = make_condition(
        method={"name": "e1_resample", "protocol": protocol, "selection": selection_sha(sources), "pass": label,
                "overrides": overrides, "seed_per_call": "sha256(seed,source)"},
        model={"name": model},
        decoding={"temperature": overrides.get("temperature"),
                  **{k: v for k, v in overrides.items() if k != "temperature"}, "seed": "per_replicate"},
        engine={"transport": transport, "protocol": RUN_PROTOCOL.get(protocol, "natural")},
        data={"study": "mixed"}, replicate=seed, block=block, order=order)
    log = start_run(Path(raw_dir), "e1_resample", cond, models=[model], ollama_base=api_base(first),
                    task={"kind": "resample", "label": label, "sources": sources})
    sent = []
    try:
        for i, src in enumerate(sources):
            ev, blobs, _ = source_event(src)
            res = replay_call(ev, blobs, client_for(ev, log), "cold_start" if protocol == "cold_start" else "resend",
                              step=i, overrides={**ov, "seed": call_seed(seed, source_key(src))})
            sent.append((src, res.span_id))
    finally:
        log.close()
    rows = compare_run(log.run_dir)
    import pandas as pd
    pd.DataFrame(rows).to_csv(log.run_dir / "resample_compare.csv", index=False)
    return {"run_dir": str(log.run_dir), "n_calls": len(sent), "n_rows": len(rows),
            "path_changed": sum(1 for r in rows if r.get("path_kept") is False),
            "value_changed": sum(1 for r in rows if r.get("value_kept") is False)}


def run_sources(header: dict, blobs) -> list[dict]:
    """The selection a resampling run sent (the run header stores a large task as a blob reference)."""
    task = header.get("task") or {}
    if is_blob_ref(task):
        task = blobs.get(task)
    return task.get("sources") or []


def measured_calls(events: list[dict]) -> list[dict]:
    """The measured calls of a resampling run, in selection order (a resend's first send is not measured)."""
    return [e for e in events if e.get("type") == "llm_call" and str(e.get("caller", "")).startswith("replay:")
            and e.get("caller") != "replay:pre"]


def source_key(src: dict) -> str:
    """Span ids repeat across runs (driver:2 in every Stage 1 run): a source is its run and its span."""
    return f"{Path(src['run']).name}/{src['span_id']}"


def compare_run(run_dir) -> list[dict]:
    """Recompute the comparison rows of one resampling run from the logs (sources are named in its task)."""
    run_dir = Path(run_dir)
    events = read_run_events(run_dir)
    header = next(e for e in events if e["type"] == "run_header")
    rep_blobs = BlobStore(run_dir.parent / "_blobs")
    sources = run_sources(header, rep_blobs)
    measured = measured_calls(events)
    if len(measured) != len(sources):
        raise ValueError(f"{run_dir.name}: {len(measured)} measured calls for {len(sources)} sources")
    lin_rep = _audit_rows(run_dir)
    cond = header.get("condition") or {}
    rows = []
    for src, rep in zip(sources, measured):
        src_ev, src_blobs, lin_src = source_event(src)
        if rep.get("replay_of") != src_ev["span_id"]:
            raise ValueError(f"{run_dir.name}: call order does not match the selection")
        for row in path_rows(src, src_ev, rep, Blobs(src_blobs, rep_blobs), lin_src, lin_rep.get(rep["span_id"])):
            rows.append({"run": run_dir.name, "pass": (cond.get("method") or {}).get("pass"),
                         "block": cond.get("block"), "source_index": sources.index(src),
                         "source_span_id": src["span_id"], "group": src.get("group"), "stratum": src.get("stratum"),
                         **{k: v for k, v in row.items() if k != "overrides"}})
    return rows


# ------------------------------------------------------------------ selections

CHAT_BINS = ((0.01, 0.05), (0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.0))  # 1 - p_path at /api/chat defaults
CONTROL_MIN_P = 0.92  # control calls: greedy token at least this likely (T=1) at every position up to the value


def hash_order(items, key=lambda x: x["span_id"]):
    """A fixed pseudo-random order (sha256 of the key): selections do not depend on file order."""
    return sorted(items, key=lambda x: hashlib.sha256(str(key(x)).encode()).hexdigest())


def call_predictions(run_dir, endpoint: str, spec_from: dict | None = None) -> list[dict]:
    """One entry per decision-bearing call of a greedy run: the minimum over its decisions of p_path at the
    endpoint's default sampler, whether that is exact, and the smallest greedy probability (T=1) up to the last
    decision value. spec_from: span_id -> {"run", "span_id"} for calls that borrow a spec."""
    r = run_events(run_dir)
    out = []
    for i, (span, ev0) in enumerate(r["events"].items()):
        src = {"run": str(Path(run_dir).resolve()), "span_id": span,
               **({"spec_from": spec_from[span]} if spec_from and span in spec_from else {})}
        ev, blobs, _ = source_event(src)
        params = sampler.defaults(endpoint, model=model_sampler_of(ev, run_dir))
        ends = decision_ends(ev, blobs)
        tokens = load_tokens(ev, blobs)
        found = [m for m in ends.values() if m.get("end") is not None]
        if not found:
            continue
        ps = [sampler.p_path(tokens, m["end"], params, constrained=m.get("constrained_positions") or ())
              for m in found]
        if any(p["p"] is None for p in ps):
            continue
        pv = [p_value_local(tokens, m, params) for m in found]
        last = max(m["end"] for m in found)
        top = [math.exp(t[1]) for t in tokens[: last + 1] if t[1] is not None]
        out.append({**src, "call_index": i, "n_decisions": len(found), "p_min": min(p["p"] for p in ps),
                    "pv_min": min((x for x in pv if x is not None), default=None),
                    "exact": all(p["exact"] for p in ps), "min_greedy_prob": min(top, default=None),
                    "constrained": any(m.get("n_constrained") for m in found)})
    return out


def stage1_runs(stage1_root) -> list[Path]:
    """Stage 1's 1a runs: under 1a/raw (run_plan stage1) or directly under 1a (stage1.py --raw-dir, the 27b pilot)."""
    root = Path(stage1_root, "1a")
    return sorted({p.parent for pat in ("raw/*/events.driver.jsonl", "*/events.driver.jsonl") for p in root.glob(pat)})


def select_chat(stage1_root, per_bin: int = 8, n_control: int = 10, n_p6: int = 20, by: str = "path") -> dict:
    """Stage 3 /api/chat selection from Stage 1's 1a runs (clark_match at T=0, batched lineage). One decision per
    call. by="path" (qwen2.5:14b): per_bin calls in each bin of 1 - p_path (CHAT_BINS), and n_control calls with
    p_path = 1 exactly and every greedy token at least CONTROL_MIN_P likely (a lineage shift cannot open the
    nucleus); the P6 subset takes the first calls of each bin in hash order. by="value" (a thinking model): p_path
    is near 0 for every call, because it runs through the reasoning, so the bins and controls are empty; every call
    with a decision is taken instead (one group, "value"), and the P6 subset is the first n_p6 in hash order."""
    cands = [c for rd in stage1_runs(stage1_root) for c in call_predictions(rd, "/api/chat")]
    if by == "value":
        chosen = [{**c, "group": "value", "stratum": "all"} for c in cands]
        return finish_selection(chosen, hash_order(chosen)[:n_p6], endpoint="/api/chat", by=by,
                                n_candidates=len(cands), population=[c["p_min"] for c in cands])
    chosen, p6 = [], []
    for lo, hi in CHAT_BINS:
        pool = hash_order([c for c in cands if lo < 1 - c["p_min"] <= hi])[:per_bin]
        chosen += [{**c, "group": "calibration", "stratum": f"{lo}-{hi}"} for c in pool]
        p6 += pool[: max(1, n_p6 // len(CHAT_BINS))]
    ctrl = hash_order([c for c in cands if c["p_min"] == 1.0 and c["exact"]
                       and (c["min_greedy_prob"] or 0) >= CONTROL_MIN_P])[:n_control]
    chosen += [{**c, "group": "control", "stratum": "p=1"} for c in ctrl]
    return finish_selection(chosen, p6, endpoint="/api/chat", by=by, n_candidates=len(cands),
                            population=[c["p_min"] for c in cands])


def select_v1(replay_runs: list, n_other: int = 10, n_p6: int = 10) -> dict:
    """Stage 3 /v1 selection from per-call cold replays of T=0 tool-loop episodes (every call batched, so the
    samples share the sources' lineage): the last call of every episode (its final decision), plus n_other
    earlier calls with the lowest value-local probability (p_path is near 0 for nearly every agent call at T=1,
    because it includes the reasoning text written before the tool call). Each replayed call borrows the decision spec of the episode
    call it replayed. The P6 subset is n_p6 final calls in hash order."""
    finals, others = [], []
    for rd in replay_runs:
        rd = Path(rd)
        header = next(e for e in read_events(rd) if e["type"] == "run_header")
        orig = Path((header.get("task") or {})["source_run"])
        spec_from = {span: {"run": str(orig.resolve()), "span_id": ev["replay_of"]}
                     for span, ev in run_events(rd)["events"].items() if ev.get("replay_of")}
        calls = call_predictions(rd, "/v1/chat/completions", spec_from)
        if calls:
            finals.append({**calls[-1], "group": "final", "stratum": "final"})
            others += [{**c, "group": "earlier", "stratum": "earlier"} for c in calls[:-1]]
    # the least certain earlier calls: most value-local probabilities are near 1, which a sample cannot test
    others = sorted(hash_order(others), key=lambda c: c["pv_min"] if c["pv_min"] is not None else 2.0)[:n_other]
    return finish_selection(finals + others, hash_order(finals)[:n_p6], endpoint="/v1/chat/completions",
                            n_candidates=len(finals) + len(others))


def finish_selection(chosen: list[dict], p6: list[dict], **meta) -> dict:
    """Order the calls so neighbours share prompt prefixes (same run together), and mark the P6 subset."""
    key = lambda c: (c["run"], c["call_index"])  # noqa: E731
    p6_keys = {key(c) for c in p6}
    chosen = sorted(chosen, key=key)
    return {**meta, "sources": chosen, "p6": [c for c in chosen if key(c) in p6_keys],
            "sha": selection_sha(chosen)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("selection", help="JSON list of sources ({run, span_id, ...}) or {\"sources\": [...]}")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--protocol", choices=PROTOCOLS, default="cold_start")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--label", default="pass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sel = json.loads(Path(a.selection).read_text())
    sources = sel["sources"] if isinstance(sel, dict) else sel
    overrides = {k: json.loads(v) for k, _, v in (s.partition("=") for s in a.set)}
    print(json.dumps(run_pass(sources, protocol=a.protocol, overrides=overrides, seed=a.seed, raw_dir=a.out,
                              label=a.label)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
