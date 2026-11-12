"""Experimental-design metadata shared by every experiment.

A condition is a nested dict with fixed namespaces, so runs from different scripts can be grouped and compared
without per-script special cases:

    make_condition(method={"name": "clark_match", "mode": "full"},
                   model={"name": "qwen2.5:14b"},
                   decoding={"temperature": 0, "seed": 1},
                   engine={"transport": "ollama", "num_ctx": 32768, "protocol": "natural"},
                   data={"study": "clark"}, replicate=2)

condition_id hashes everything except replicate/block/order, so the replicates of one condition share it and
stability_report can group them automatically. decoding.seed may be "per_replicate": each replicate samples with
its own seed (its replicate number), and the condition stays the same.

engine.protocol says how the engine state is prepared, as (start action, per-call action):

    natural      (as_is, none)     whatever state the server is in
    cold_start   (unload, none)    the model is unloaded once before the run, which also empties the host-RAM
                                   prompt cache: every replicate starts from the same state (E1 Stage 1e)
    cold         (as_is, unload)   unloaded before every call (a full recompute each time; slow)

start_run performs the start action for every route; hbench.llm.LLMClient performs the per-call action, so a
per-call protocol is refused for routes it does not drive (LiteLLM).

ollama_lease() is an advisory lock our own runners take, so two experiment processes on this machine do not
interleave requests (the engine audit then confirms no foreign requests reached the server).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

NAMESPACES = ("method", "model", "decoding", "engine", "data")
REQUIRED = {("method", "name"), ("model", "name"), ("data", "study")}
REPLICATION_KEYS = ("replicate", "block", "order")
PROTOCOLS = {"natural": ("as_is", "none"), "cold_start": ("unload", "none"), "cold": ("as_is", "unload")}
SEED_PER_REPLICATE = "per_replicate"
LEASE_PATH = Path(os.environ.get("HBENCH_LEASE", Path.home() / ".hbench" / "ollama.lock"))


def make_condition(method: dict, model: dict, decoding: dict, engine: dict, data: dict, replicate: int = 0,
                   block=None, order=None, **extra) -> dict:
    decoding = {k: v for k, v in decoding.items() if not (k == "seed_mode" and v == "per_call")}  # the default
    cond = {"method": dict(method), "model": dict(model), "decoding": dict(decoding), "engine": dict(engine),
            "data": dict(data), "replicate": replicate, "block": block, "order": order}
    if extra:
        cond["extra"] = extra
    problems = validate_condition(cond)
    if problems:
        raise ValueError("; ".join(problems))
    return cond


def validate_condition(cond: dict) -> list[str]:
    probs = [f"missing {ns}.{k}" for ns, k in REQUIRED if not (cond.get(ns) or {}).get(k)]
    probs += [f"{ns} must be a dict" for ns in NAMESPACES if not isinstance(cond.get(ns), dict)]
    dec = cond.get("decoding") or {}
    if "temperature" not in dec:
        probs.append("decoding.temperature must be stated (None means 'server default', which differs by endpoint)")
    seed = dec.get("seed")
    if seed is not None and seed != SEED_PER_REPLICATE and (not isinstance(seed, int) or isinstance(seed, bool)):
        probs.append(f"decoding.seed must be an integer, None or {SEED_PER_REPLICATE!r}")
    if dec.get("seed_mode") is not None and dec["seed_mode"] not in ("per_call", "shared"):
        probs.append("decoding.seed_mode must be 'per_call' (the default) or 'shared' (see hbench.seeds)")
    proto = (cond.get("engine") or {}).get("protocol", "natural")
    if proto not in PROTOCOLS:
        probs.append(f"engine.protocol must be one of {tuple(PROTOCOLS)}")
    return probs


def protocol_actions(cond: dict | None) -> tuple[str, str]:
    """(start action, per-call action) of a condition's engine protocol."""
    return PROTOCOLS.get(((cond or {}).get("engine") or {}).get("protocol", "natural"), PROTOCOLS["natural"])


def effective_seed(cond: dict) -> int | None:
    """The seed a run samples with: the stated one, or its replicate number under "per_replicate"."""
    seed = (cond.get("decoding") or {}).get("seed")
    return cond.get("replicate") if seed == SEED_PER_REPLICATE else seed


def condition_id(cond: dict) -> str:
    """Short hash of the condition without replication keys; replicates of one condition share it."""
    core = {k: v for k, v in cond.items() if k not in REPLICATION_KEYS}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:12]


def condition_slug(cond: dict) -> str:
    """Readable part of a run id: method, mode, model, temperature, replicate."""
    m, d = cond.get("method", {}), cond.get("decoding", {})
    parts = [m.get("name"), m.get("mode"), cond.get("model", {}).get("name"),
             None if d.get("temperature") is None else f"T{d['temperature']}",
             None if d.get("seed") in (None, SEED_PER_REPLICATE) else f"s{d['seed']}", f"r{cond.get('replicate', 0)}"]
    return "-".join(str(p) for p in parts if p not in (None, ""))


def flatten(cond: dict) -> dict:
    """{"decoding.temperature": 0, ...} for tables."""
    out = {}
    for ns in NAMESPACES:
        for k, v in (cond.get(ns) or {}).items():
            out[f"{ns}.{k}"] = v
    for k in REPLICATION_KEYS:
        out[k] = cond.get(k)
    return out


@contextmanager
def ollama_lease(holder: str, wait: float = 0.0, path: Path = LEASE_PATH):
    """Exclusive advisory lease on the local Ollama server. Raises TimeoutError if another holder keeps it
    longer than `wait` seconds; yields the lease record (also written into the lock file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    end = time.time() + wait
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() >= end:
                f.seek(0)
                other = f.read().strip()
                f.close()
                raise TimeoutError(f"Ollama lease held by: {other or 'unknown'}")
            time.sleep(0.5)
    record = {"holder": holder, "pid": os.getpid(), "host": socket.gethostname(), "since": time.time()}
    f.seek(0)
    f.truncate()
    f.write(json.dumps(record))
    f.flush()
    try:
        yield record
    finally:
        f.seek(0)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
