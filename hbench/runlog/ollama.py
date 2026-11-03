"""Read-only queries of the local Ollama server for run headers. Failures come back as {"error": ...}."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

DEFAULT_BASE = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")


def _request(base: str, path: str, body: dict | None = None, timeout: float = 5.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001 - a header must never break a run
        return {"error": f"{type(e).__name__}: {e}"}


def version(base: str = DEFAULT_BASE) -> dict:
    return _request(base, "/api/version")


def tags(base: str = DEFAULT_BASE) -> dict:
    return _request(base, "/api/tags")


def show(model: str, base: str = DEFAULT_BASE) -> dict:
    return _request(base, "/api/show", {"model": model})


def ps(base: str = DEFAULT_BASE) -> dict:
    return _request(base, "/api/ps")


def _sha(s) -> str | None:
    return None if s is None else hashlib.sha256(str(s).encode()).hexdigest()


def model_info(model: str, base: str = DEFAULT_BASE, _tags: dict | None = None) -> dict:
    """Digest, quantization and template/system hashes: what makes "the same model" reproducible."""
    name = model.split("/", 1)[-1] if "/" in model else model  # "ollama_chat/qwen2.5:14b" -> "qwen2.5:14b"
    t = _tags if _tags is not None else tags(base)
    entry = next((m for m in t.get("models", []) if m.get("name") == name or m.get("model") == name), None)
    s = show(name, base)
    if "error" in s and entry is None:
        return {"name": name, "error": s["error"]}
    ctx = next((v for k, v in (s.get("model_info") or {}).items() if k.endswith(".context_length")), None)
    return {"name": name, "digest": entry.get("digest") if entry else None,
            "size": entry.get("size") if entry else None,
            "quantization": (s.get("details") or {}).get("quantization_level"),
            "family": (s.get("details") or {}).get("family"),
            "template_sha256": _sha(s.get("template")), "system_sha256": _sha(s.get("system")),
            "modelfile_parameters": s.get("parameters"), "max_context_length": ctx,
            "modified_at": s.get("modified_at"), "capabilities": s.get("capabilities")}


def server_state(models=(), base: str = DEFAULT_BASE) -> dict:
    """Ollama version, the requested models, and what is loaded right now (other clients share the server)."""
    v = version(base)
    if "error" in v:
        return {"api_base": base, "error": v["error"]}
    t = tags(base)
    return {"api_base": base, "version": v.get("version"),
            "models": {m: model_info(m, base, t) for m in models}, "ps": loaded(base)}


def loaded(base: str = DEFAULT_BASE) -> list:
    p = ps(base)
    if "error" in p:
        return [p]
    return [{k: m.get(k) for k in ("name", "digest", "context_length", "size_vram", "expires_at")}
            for m in p.get("models", [])]
