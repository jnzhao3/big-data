"""start_run(): the one way experiment scripts open a run.

It takes the Ollama lease, creates the run folder, writes the run header (condition and its id, engine snapshot
with the server-log offset, engine drift, code/packages/data), and arranges that closing the run audits the
server log for this run's window and releases the lease. Scripts then only make calls and close the log.

    log = start_run(RAW_ROOT, "e1", condition, task={"kind": "sm"}, models=["qwen2.5:14b"], prompts=PROMPTS)
    ...
    log.close()            # writes engine_audit.json, run_end carries its summary
"""

from __future__ import annotations

from pathlib import Path

from ..data import Data
from ..design import condition_slug, ollama_lease, validate_condition
from .header import collect_run_header
from .writer import RunLog


def start_run(root: str | Path, exp: str, condition: dict, *, task: dict, models=(), prompts: dict | None = None,
              data: Data | None = None, llm: dict | None = None, lease: bool = True, lease_wait: float = 0.0,
              audit: bool = True, verify_data: bool = True, **extra) -> RunLog:
    problems = validate_condition(condition)
    if problems:
        raise ValueError("invalid condition: " + "; ".join(problems))
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
    log.emit("run_header", **collect_run_header(exp, condition, task, llm=llm, prompts=prompts, data=data,
                                                ollama_models=models, verify_data=verify_data, lease=lease_rec,
                                                **extra))
    return log
