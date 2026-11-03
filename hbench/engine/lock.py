"""env/engine-lock.json: the inference engine the project is pinned to, checked against every run header.

The Python lock (env/requirements-lock.txt) does not cover Ollama, the model weights, or the runner flags, yet
any of them changes outputs. A Homebrew upgrade of Ollama would do so silently; this file makes it visible.
"""

from __future__ import annotations

import json
from pathlib import Path

ENGINE_LOCK = Path(__file__).resolve().parents[2] / "env" / "engine-lock.json"
# Server-level invariants only. The context size (-c) and batch sizes (-b/-ub) follow whatever num_ctx the last
# caller requested (e.g. qwen3.8:27b was last started with -c 16384 -b 512), so they are per-run covariates,
# recorded in every run header's engine snapshot and in each call's decoding, not lockable configuration.
LOCKED_RUNNER_FLAGS = ("np", "cache-type-k", "cache-type-v", "flash-attn", "context-shift", "keep", "chat-template")


def lock_from_snapshot(snap: dict) -> dict:
    api = snap.get("api") or {}
    return {"ollama_version": api.get("version"),
            "models": {m: {k: info.get(k) for k in ("digest", "template_sha256", "system_sha256", "quantization")}
                       for m, info in (api.get("models") or {}).items()},
            "server_config": snap.get("server_config") or {},
            "prompt_cache_mib": snap.get("prompt_cache_mib"),
            "runners": {m: {"model_blob_sha256": r.get("model_blob_sha256"),
                            "flags": {k: v for k, v in (r.get("flags") or {}).items() if k in LOCKED_RUNNER_FLAGS}}
                        for m, r in (snap.get("runners") or {}).items() if r.get("flags")}}


def check(snap: dict, lock: dict | None = None) -> list[str]:
    """Differences between the current engine and the lock, plus "not checked: ..." for locked settings that
    could not be read (no server log). An empty list means the engine was checked and matches. Models or runners
    the snapshot did not look at are not compared."""
    if lock is None:
        if not ENGINE_LOCK.exists():
            return ["no env/engine-lock.json"]
        lock = json.loads(ENGINE_LOCK.read_text())
    cur = lock_from_snapshot(snap)
    drift = []
    if lock.get("ollama_version") and cur["ollama_version"] and lock["ollama_version"] != cur["ollama_version"]:
        drift.append(f"ollama {cur['ollama_version']} != locked {lock['ollama_version']}")
    for m, want in (lock.get("models") or {}).items():
        have = cur["models"].get(m)
        if have is None:
            continue
        for k, v in want.items():
            if v is not None and have.get(k) is not None and have[k] != v:
                drift.append(f"{m}: {k} changed")
    if lock.get("server_config") and not cur["server_config"]:
        # the server log is the only source of these settings; without it a mismatch would pass silently
        path = (snap.get("server_log") or {}).get("path")
        drift.append(f"not checked: server config and runner flags (no readable server log at {path}; "
                     "set HBENCH_OLLAMA_LOG)")
    pc, want_pc = cur["prompt_cache_mib"], lock.get("prompt_cache_mib")
    if want_pc and pc and pc != want_pc:  # the host-RAM prompt cache decides which calls take the single-token path
        drift.append(f"prompt cache {pc:g} MiB != locked {want_pc:g} MiB")
    for k, v in (lock.get("server_config") or {}).items():
        if k in cur["server_config"] and cur["server_config"][k] != v:
            drift.append(f"server {k}={cur['server_config'][k]} != locked {v}")
    for m, want in (lock.get("runners") or {}).items():
        have = cur["runners"].get(m)
        if not have:
            continue
        for k, v in want["flags"].items():
            if have["flags"].get(k) != v:
                drift.append(f"{m}: runner --{k}={have['flags'].get(k)} != locked {v}")
    return drift
