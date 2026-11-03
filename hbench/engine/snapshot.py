"""What the inference engine is right now: Ollama's API view plus what only its server log records.

The API gives the version, model digests, template hashes and loaded models. The server log adds the runner's
real configuration (llama-server flags: context, KV-cache precision, flash attention, batch sizes, context
shift), the server's environment (parallelism, keep-alive, ...), and the host-RAM prompt cache size. These are
part of the computation path, so they belong in every run header. A missing or unreadable log is recorded as
unavailable, never an error.
"""

from __future__ import annotations

import os
import re
import shlex

from ..runlog import ollama

SERVER_LOG = os.environ.get("HBENCH_OLLAMA_LOG", "/opt/homebrew/var/log/ollama.log")
_SERVER_CONFIG = re.compile(r'msg="server config" env="map\[(.*)\]"')
_RUNNER = re.compile(r'msg="starting llama-server" cmd="([^"]*)"')
_PROMPT_CACHE = re.compile(r"prompt cache is enabled, size limit: ([\d.]+) MiB")
SERVER_KEYS = ("OLLAMA_CONTEXT_LENGTH", "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_NUM_PARALLEL",
               "OLLAMA_KEEP_ALIVE", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_MAX_QUEUE", "OLLAMA_SCHED_SPREAD",
               "OLLAMA_NEW_ENGINE", "OLLAMA_GPU_OVERHEAD", "OLLAMA_LOAD_TIMEOUT")


def log_offset(path: str = SERVER_LOG) -> int | None:
    """Current size of the server log: runs record it at start and end and audit the bytes in between."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _scan_back(path: str, patterns: dict, chunk: int = 4 << 20, limit: int = 256 << 20) -> dict:
    """Last match of each pattern, reading the file backwards in chunks (the log grows without bound)."""
    found: dict = {}
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            end, carry = size, b""
            while end > 0 and len(found) < len(patterns) and size - end < limit:
                start = max(0, end - chunk)
                f.seek(start)
                data = f.read(end - start) + carry
                lines = data.split(b"\n")
                carry, lines = (lines[0], lines[1:]) if start > 0 else (b"", lines)
                for line in reversed(lines):
                    text = line.decode("utf-8", "replace")
                    for name, (rx, want) in patterns.items():
                        if name not in found:
                            m = rx.search(text)
                            if m and (want is None or want(m)):
                                found[name] = m
                end = start
    except OSError:
        pass
    return found


def parse_env_map(s: str) -> dict:
    return {k: v for k, v in re.findall(r"(\w+):(\[[^\]]*\]|\S*)", s)}


def parse_runner_cmd(cmd: str) -> dict:
    """llama-server flags as a dict ("--flash-attn on" -> {"flash-attn": "on"}; bare flags -> True)."""
    parts = shlex.split(cmd)
    flags, i = {}, 1
    while i < len(parts):
        p = parts[i]
        if p.startswith("-"):
            name = p.lstrip("-")
            if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                flags[name] = parts[i + 1]
                i += 2
                continue
            flags[name] = True
        i += 1
    return flags


def _model_blob(show: dict) -> str | None:
    for line in (show.get("modelfile") or "").splitlines():
        if line.startswith("FROM ") and "sha256-" in line:
            return line.split("sha256-", 1)[1].strip()
    return None


def engine_snapshot(models=(), base: str = ollama.DEFAULT_BASE, server_log: str = SERVER_LOG) -> dict:
    """API state, server configuration and, for each requested model, the runner flags it was last started with."""
    snap = {"api": ollama.server_state(models, base), "server_log": {"path": server_log, "offset": log_offset(server_log)}}
    blobs = {}
    for m in models:
        show = ollama.show(m.split("/", 1)[-1] if "/" in m else m, base)
        blobs[m] = _model_blob(show) if "error" not in show else None
    patterns = {"server_config": (_SERVER_CONFIG, None), "prompt_cache": (_PROMPT_CACHE, None)}
    for m, blob in blobs.items():
        if blob:
            patterns[f"runner:{m}"] = (_RUNNER, (lambda b: lambda mt: b in mt.group(1))(blob))
    found = _scan_back(server_log, patterns) if snap["server_log"]["offset"] is not None else {}
    if not found:
        snap["server_log"]["unavailable"] = "server log not readable or no matching lines"
    if "server_config" in found:
        env = parse_env_map(found["server_config"].group(1))
        snap["server_config"] = {k: env.get(k) for k in SERVER_KEYS if k in env}
    if "prompt_cache" in found:
        snap["prompt_cache_mib"] = float(found["prompt_cache"].group(1))
    snap["runners"] = {}
    for m in models:
        mt = found.get(f"runner:{m}")
        if mt:
            flags = parse_runner_cmd(mt.group(1))
            flags.pop("model", None)
            flags.pop("port", None)
            snap["runners"][m] = {"model_blob_sha256": blobs[m], "flags": flags}
        else:
            snap["runners"][m] = {"model_blob_sha256": blobs.get(m), "unavailable": "no runner start found in the log"}
    return snap
