"""start_run(): the one way experiment scripts open a run.

It takes the Ollama lease, creates the run folder, writes the run header (condition and its id, the effective
decoding, engine snapshot with the server-log offset, engine drift, code/packages/data), performs the start
action of the engine protocol (cold_start: unload the models), and arranges that closing the run audits the
server log for this run's window and releases the lease. Scripts then only make calls and close the log.

    log = start_run(RAW_ROOT, "e1", condition, task={"kind": "sm"}, models=["qwen2.5:14b"], prompts=PROMPTS)
    ...
    log.close()            # writes engine_audit.json, run_end carries its summary
"""

from __future__ import annotations

import os
from pathlib import Path

from ..data import Data
from ..design import condition_slug, effective_seed, ollama_lease, protocol_actions, validate_condition
from ..sampler import effective, endpoint_of
from .header import collect_run_header
from .writer import RunLog


def raw_root(default: str | Path | None = None) -> Path:
    """Where run folders go: $HBENCH_RAW, else the script's default, else ~/hbench_raw. Never inside the repo by
    default: Qwen's top-k candidates in the raw logs contain Chinese tokens (E1 Stage 1)."""
    if os.environ.get("HBENCH_RAW"):
        return Path(os.environ["HBENCH_RAW"])
    return Path(default) if default is not None else Path.home() / "hbench_raw"


def start_run(root: str | Path, exp: str, condition: dict, *, task: dict, models=(), prompts: dict | None = None,
              data: Data | None = None, llm: dict | None = None, lease: bool = True, lease_wait: float = 0.0,
              audit: bool = True, verify_data: bool = True, ollama_base: str | None = None, **extra) -> RunLog:
    problems = validate_condition(condition)
    if problems:
        raise ValueError("invalid condition: " + "; ".join(problems))
    proto = (condition.get("engine") or {}).get("protocol", "natural")
    start_action, per_call = protocol_actions(condition)
    transport = (condition.get("engine") or {}).get("transport") or ""
    if per_call != "none" and transport.startswith("litellm/"):
        raise ValueError(f"engine.protocol {proto!r} has a per-call action, which only hbench.llm.LLMClient runs; "
                         f"transport {transport} cannot (use cold_start or natural)")
    if start_action == "unload" and not models:
        raise ValueError(f"engine.protocol {proto!r} unloads the models before the run: pass models=[...]")
    slug = condition_slug(condition)
    lease_cm, lease_rec = None, None
    if lease and models:
        lease_cm = ollama_lease(holder=f"{exp}:{slug}", wait=lease_wait)
        lease_rec = lease_cm.__enter__()
    try:
        log = RunLog.create(root, exp, slug, condition["data"]["study"])
    except Exception:
        if lease_cm is not None:
            lease_cm.__exit__(None, None, None)
        raise
    if audit and models:
        from ..engine import audit_run

        def engine_audit(log_):
            return {"engine_audit": audit_run(log_.run_dir)}
        log.add_close_hook(engine_audit)
    if lease_cm is not None:
        def release_lease(log_):
            lease_cm.__exit__(None, None, None)
            return {"lease_released": True}
        log.add_close_hook(release_lease)
    # the sampler the calls run with: what the condition states, over the endpoint's defaults (the audit checks it)
    log.condition = condition
    log.effective_decoding = effective(condition.get("decoding"), endpoint_of(transport))
    seed = effective_seed(condition)
    if seed is not None:
        log.effective_decoding["seed"] = seed
    base = {"ollama_base": ollama_base} if ollama_base else {}
    log.emit("run_header", **collect_run_header(exp, condition, task, llm=llm, prompts=prompts, data=data,
                                                ollama_models=models, verify_data=verify_data, lease=lease_rec,
                                                effective_decoding=log.effective_decoding, **base, **extra))
    if start_action == "unload":  # after the header (its log offset), so the audit sees the reload that follows
        from ..llm.client import LLMClient
        for m in models:
            LLMClient(log, m, **({"base": ollama_base} if ollama_base else {}), per_call="none").unload(protocol=proto)
    return log
