"""The run_header event: everything needed to tell whether two runs were set up the same way."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import SCORING_VERSION, __version__
from ..data import REPO_ROOT, Data
from ..manifest import MANIFEST_PATH, verify_manifest
from . import ollama

PACKAGES = ("bdi-kit", "litellm", "langgraph", "langchain-core", "langchain-litellm", "langchain-mcp-adapters", "mcp",
            "magneto-python", "valentine", "polyfuzz", "pandas", "numpy", "torch", "transformers",
            "sentence-transformers", "scikit-learn")
ENV_VARS = ("PYTHONHASHSEED", "BDIKIT_CACHE", "HBENCH_DATA", "HBENCH_BDIKIT_CACHE", "OLLAMA_HOST",
            "OPENAI_BASE_URL", "OPENAI_API_BASE", "LITELLM_LOCAL_MODEL_COST_MAP", "HF_HUB_OFFLINE",
            "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS")
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10,
                              check=True).stdout
    except Exception:  # noqa: BLE001
        return None


def code_info() -> dict:
    status = _git("status", "--porcelain", "--untracked-files=no")
    diff = _git("diff", "HEAD")
    return {"git_commit": (_git("rev-parse", "HEAD") or "").strip() or None,
            "git_branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
            "git_dirty": bool(status.strip()) if status is not None else None,
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "entrypoint": sys.argv[0], "argv": sys.argv[1:]}


def python_info() -> dict:
    f = sys.flags
    return {"version": platform.python_version(), "executable": sys.executable,
            "flags": {"ignore_environment": f.ignore_environment, "hash_randomization": f.hash_randomization,
                      "isolated": f.isolated, "no_user_site": f.no_user_site},
            # os.environ can say PYTHONHASHSEED=0 while `python -I` ignored it; flags say what took effect
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}


def package_info() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    every = sorted(f"{d.metadata['Name']}=={d.version}".lower() for d in md.distributions() if d.metadata["Name"])
    return {"versions": versions, "environment_sha256": hashlib.sha256("\n".join(every).encode()).hexdigest(),
            "n_distributions": len(every)}


def env_info() -> dict:
    present = sorted(k for k in os.environ if any(m in k.upper() for m in _SECRET_MARKERS))
    return {"vars": {k: os.environ[k] for k in ENV_VARS if k in os.environ and
                     not any(m in k.upper() for m in _SECRET_MARKERS)},
            "secret_like_vars_present": present}  # names only, never values


def dataset_info(data: Data, verify: bool = True, mpath: Path = MANIFEST_PATH) -> dict:
    if not mpath.exists():
        return {"root": str(data.root), "manifest": None}
    manifest = json.loads(mpath.read_text())
    out = {"root": str(data.root), "manifest_sha256": manifest.get("manifest_sha256")}
    if verify:
        problems = verify_manifest(data.root, manifest)
        out.update(verified=not problems, problems=problems[:20])
    return out


def collect_run_header(exp: str, condition: dict, task: dict, llm: dict | None = None, prompts: dict | None = None,
                       data: Data | None = None, ollama_models=(), ollama_base: str = ollama.DEFAULT_BASE,
                       verify_data: bool = True, **extra) -> dict:
    """Fields of the run_header event. prompts maps a name to the template text; only hashes are kept.
    condition should come from hbench.design.make_condition; its condition_id is recorded alongside. With
    ollama_models, the engine snapshot (API state, server config, runner flags, server-log offset) and its
    drift against env/engine-lock.json are recorded too."""
    from ..design import NAMESPACES, condition_id
    from ..engine import check as engine_check
    from ..engine import engine_snapshot

    data = data or Data()
    structured = isinstance(condition, dict) and all(isinstance(condition.get(ns), dict) for ns in NAMESPACES)
    engine = engine_snapshot(ollama_models, ollama_base) if ollama_models else None
    return {"exp": exp, "condition": condition, "condition_id": condition_id(condition) if structured else None,
            "task": task, "llm": llm or {},
            "prompts": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in (prompts or {}).items()},
            "code": code_info(), "python": python_info(), "packages": package_info(), "env": env_info(),
            "engine": engine, "engine_drift": engine_check(engine) if engine else None,
            "dataset": dataset_info(data, verify_data),
            "hbench": {"version": __version__, "scoring_version": SCORING_VERSION},
            "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
                     "cpu_count": os.cpu_count()},
            "started_at": datetime.now(timezone.utc).isoformat(), **extra}
