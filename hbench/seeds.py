"""Sampling seeds: one per call.

llama-server (Ollama 0.40.0) reseeds its sampler with each request's seed. Calls sent with the same seed therefore
draw the same random number at the same token position. At T>0 their decisions are sampled with correlated draws:
when the draw at a position is small, every call picks its top candidate there. Each decision's own probability
of changing is unaffected, and so is the expected number of changes over many decisions, but anything that
assumes independent samples (variances, intervals, the sampling_check alarm) is wrong.

So the seed a condition or a caller states is a base, and every sampled call (temperature above 0, or unset,
which means the endpoint's default above 0) is sent with

    call_seed(base, scope, prompt_sha256, k)      rule "call/1": sha256 of the four, first 31 bits

where prompt_sha256 ignores tool-call ids (Ollama's are random), scope is the process and the caller
("driver/method:clark_match": two processes of one run that send the same prompt get different seeds) and k counts earlier calls of that process with the same base, scope and prompt. Calls are
independent, a run is reproducible from its base seed, the same prompt in two conditions gets the same seed
(common random numbers), and `hbench replay --seeds s` sends exactly the seeds a natural run with base s sent.
Greedy calls (temperature 0) keep the stated seed: the sampler does not use it, and T=0 requests stay byte-identical
to earlier runs.

Every route applies the rule: LLMClient.chat, the LiteLLM callback (hbench.runlog.litellm_cb) and replay. A sampled
call whose seed was derived carries seed_base and seed_rule in its llm_call event. A condition that studies shared
seeds on purpose states decoding.seed_mode "shared". check_seeds() verifies a run from its log alone, whatever
route sent the calls; `log validate` runs it.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter

SEED_RULE = "call/1"
SEED_MODES = ("per_call", "shared")


def call_seed(base: int, scope: str | None, prompt_sha256: str | None, k: int) -> int:
    """The seed one sampled call is sent with: a non-negative 31-bit integer."""
    h = hashlib.sha256(json.dumps([base, scope, prompt_sha256, k]).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def seed_scope(proc: str | None, caller: str | None) -> str:
    """The process and caller part of the key: "driver/method:clark_match"."""
    return f"{proc or 'driver'}/{caller}"


def is_sampled(temperature) -> bool:
    """Does a call with this temperature sample? None is the endpoint's default, which is above 0 on both."""
    return temperature is None or float(temperature) > 0


def seed_mode(condition: dict | None) -> str:
    return ((condition or {}).get("decoding") or {}).get("seed_mode") or "per_call"


class SeedDeriver:
    """Counts calls per (base, caller, prompt), so repeated prompts get their own seeds."""

    def __init__(self, proc: str | None = None):
        self._seen: Counter = Counter()
        self.proc = proc
        self._lock = threading.Lock()

    def derive(self, base: int, caller: str | None, prompt_sha256: str | None) -> int:
        scope = seed_scope(self.proc, caller)
        key = (base, scope, prompt_sha256)
        with self._lock:
            k = self._seen[key]
            self._seen[key] += 1
        return call_seed(base, scope, prompt_sha256, k)


def deriver_for(log) -> SeedDeriver:
    """The run's deriver (one per RunLog in this process); a fresh one without a log."""
    if log is None:
        return SeedDeriver()
    d = getattr(log, "_seed_deriver", None)
    if d is None:
        d = SeedDeriver(getattr(log, "proc", None))
        log._seed_deriver = d
    return d


def _sampled_calls(events: list[dict]) -> list[dict]:
    """Sampled calls that reached the model, without replay pre-sends (a pre-send repeats its call on purpose)."""
    return [e for e in events if e.get("type") == "llm_call" and not e.get("error") and e.get("caller") != "replay:pre"
            and is_sampled((e.get("decoding") or {}).get("temperature"))]


def check_seeds(events: list[dict]) -> dict:
    """Are the run's sampled calls independent? From the llm_call events alone, whatever route sent them.

    problems: one seed sent with different prompts by calls whose seed was not derived (derived seeds that
    collide do so by chance). notes: the run shares seeds on purpose (seed_mode "shared"); the same prompt was
    sampled more than once with the same seed (seeded repeats: they test whether a seed reproduces, and are not
    independent samples); sampled calls sent no seed (independent, but cannot be reproduced); the condition
    states a fixed integer seed at T>0 (every replicate of it samples with the same draws, see
    check_replicates). independent: None without sampled calls, False after any shared seed or seeded repeat."""
    header = next((e for e in events if e.get("type") == "run_header" and e.get("proc", "driver") == "driver"), {})
    cond = header.get("condition") or {}
    shared_mode = seed_mode(cond) == "shared"
    sampled = _sampled_calls(events)
    out = {"problems": [], "notes": [], "independent": None, "n_sampled": len(sampled)}
    if not sampled:
        return out
    groups: dict = {}
    for e in sampled:
        seed = (e.get("decoding") or {}).get("seed")
        if seed is not None:
            groups.setdefault(seed, []).append(e)
    bad = {s: g for s, g in groups.items()
           if len({e.get("prompt_sha256") for e in g}) > 1 and any(not e.get("seed_rule") for e in g)}
    if bad:
        n = sum(len(g) for g in bad.values())
        msg = (f"{n} sampled calls with different prompts share one seed ({sorted(bad)[:5]}): llama-server draws "
               f"the same random numbers for them, so their samples are correlated")
        (out["notes"] if shared_mode else out["problems"]).append(
            msg + (" (decoding.seed_mode is 'shared')" if shared_mode else ""))
    pairs = Counter(((e.get("decoding") or {}).get("seed"), e.get("prompt_sha256")) for e in sampled
                    if (e.get("decoding") or {}).get("seed") is not None)
    repeats = sum(n for n in pairs.values() if n > 1)
    if repeats:
        out["notes"].append(f"{repeats} sampled calls repeat a prompt with the same seed: seeded repeats, not "
                            f"independent samples")
    unseeded = sum(1 for e in sampled if (e.get("decoding") or {}).get("seed") is None)
    if unseeded:
        out["notes"].append(f"{unseeded} sampled calls sent no seed: independent, but they cannot be reproduced")
    dec = cond.get("decoding") or {}
    if isinstance(dec.get("seed"), int) and not isinstance(dec.get("seed"), bool) and is_sampled(dec.get("temperature")):
        out["notes"].append(f"decoding.seed is the fixed integer {dec['seed']} at T>0: every replicate of this "
                            f"condition samples with the same draws (use 'per_replicate' for independent replicates)")
    out["independent"] = not bad and not repeats
    return out


def check_replicates(runs: dict[str, list[dict]]) -> dict:
    """Are several runs (replicates of one condition) independent of each other? Two runs that sampled the same
    prompt with the same seed drew the same random numbers: copies, not independent replicates (a fixed integer
    seed, or replay repeats without --seeds). {"independent": None | bool, "shared": [[run, run, n], ...]}."""
    keys = {}
    for name, events in runs.items():
        keys[name] = {((e.get("decoding") or {}).get("seed"), e.get("prompt_sha256")) for e in _sampled_calls(events)
                      if (e.get("decoding") or {}).get("seed") is not None}
    names = sorted(keys)
    shared = [[a, b, len(keys[a] & keys[b])] for i, a in enumerate(names) for b in names[i + 1:] if keys[a] & keys[b]]
    any_sampled = any(_sampled_calls(ev) for ev in runs.values())
    return {"independent": (not shared) if any_sampled else None, "shared": shared}
