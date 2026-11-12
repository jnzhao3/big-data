"""BDI-Kit's MCP server, logged into an hbench run (design option A).

The agent driver starts this instead of `bdikit-mcp`:

    python -s -m hbench.runlog.mcp_logged --llm openai/qwen2.5-14b-bdikit

with HBENCH_RUN_DIR and HBENCH_RUN_ID naming the run the driver opened. The order of the steps matters:

1. stdout is taken for JSON-RPC alone: the real fd 1 is kept for the protocol and fd 1 is pointed at stderr, so a
   print() anywhere (BDI-Kit's download.py prints when it creates cache directories) cannot corrupt the channel.
2. The run's events go to events.mcp.<pid>.jsonl; the first is this process's run_header.
3. The BDI-Kit disk cache is redirected (HBENCH_BDIKIT_CACHE): `run` (default) gives the run its own empty
   runs/ directory, with models/ and embeddings/ linked to the shared ones; `off` never reads or writes results;
   `shared` uses the shared cache as is. BDIKIT_CACHE is read when bdikit is imported, so this comes first.
4. LiteLLM's completion functions are replaced, before bdikit is imported (it binds them with `from litellm import
   completion`): every tool-internal call gets the HBENCH_TOOL_LLM_* parameters, and a call to a model that is
   not allowed or to a non-local endpoint is refused loudly. BDI-Kit's matchers default to openai/gpt-4o-mini, and
   its Portkey route sets its own api_base; both are refused. The hbench LiteLLM callback logs every call.
5. bdikit.mcp_server is imported, every tool is wrapped (a tool_call per call, with result_sha256 and the
   order-insensitive result_set_sha256, then a state_snapshot of the server's matching results), and the server
   runs on the kept stdout.

Environment:
  HBENCH_RUN_DIR, HBENCH_RUN_ID     the run (required; the directory's name must be the id)
  HBENCH_TOOL_LLM_TEMPERATURE, _TOP_P, _SEED, _TOP_LOGPROBS
                                    injected into every tool-internal call; unset or "default" leaves it out
  HBENCH_TOOL_LLM_API_BASE          where openai/ calls go (default http://127.0.0.1:11434/v1); must be local
  HBENCH_TOOL_LLM_MODELS            comma-separated models tool calls may use (default: the --llm value)
  HBENCH_BDIKIT_CACHE               run | off | shared (default run)
  HBENCH_BDIKIT_SHARED              the shared cache with models/ and embeddings/ (default ~/.cache/bdikit)
"""

from __future__ import annotations

import argparse
import contextvars
import functools
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

from .writer import RunLog, sha256_json

DEFAULT_TOOL_API_BASE = "http://127.0.0.1:11434/v1"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
CACHE_MODES = ("run", "off", "shared")
TOOL_LLM_PARAMS = ("temperature", "top_p", "seed", "top_logprobs")
# The tool call in progress: its start span, cache use, LLM call count and the raw result (set by the wrapper).
CURRENT: contextvars.ContextVar[dict | None] = contextvars.ContextVar("hbench_mcp_tool", default=None)


class LocalOnlyError(RuntimeError):
    """A tool tried to call an LLM that is not allowed or not local."""


class ConfigError(SystemExit):
    pass


# ----------------------------------------------------------------------------------------------- configuration

def _param(env: dict, name: str):
    raw = env.get(f"HBENCH_TOOL_LLM_{name.upper()}")
    if raw is None or raw.strip().lower() in ("", "default", "none"):
        return None
    return int(raw) if name in ("seed", "top_logprobs") else float(raw)


def config_from_env(argv: list[str] | None = None, env: dict | None = None) -> dict:
    env = dict(os.environ if env is None else env)
    ap = argparse.ArgumentParser(prog="hbench.runlog.mcp_logged", description=__doc__.split("\n")[0])
    ap.add_argument("--llm", required=True, help="model of BDI-Kit's LLM matchers, e.g. openai/qwen2.5-14b-bdikit")
    args = ap.parse_args(argv)
    run_dir, run_id = env.get("HBENCH_RUN_DIR"), env.get("HBENCH_RUN_ID")
    if not run_dir or not run_id:
        raise ConfigError("mcp_logged: HBENCH_RUN_DIR and HBENCH_RUN_ID must be set (the driver's run)")
    if Path(run_dir).name != run_id or not Path(run_dir).is_dir():
        raise ConfigError(f"mcp_logged: HBENCH_RUN_DIR {run_dir} is not the directory of run {run_id}")
    api_base = env.get("HBENCH_TOOL_LLM_API_BASE") or DEFAULT_TOOL_API_BASE
    if (urlparse(api_base).hostname or "") not in LOCAL_HOSTS:
        raise ConfigError(f"mcp_logged: HBENCH_TOOL_LLM_API_BASE {api_base} is not local")
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        if env.get(var) and (urlparse(env[var]).hostname or "") not in LOCAL_HOSTS:
            raise ConfigError(f"mcp_logged: {var}={env[var]} is not local")
    mode = env.get("HBENCH_BDIKIT_CACHE", "run")
    if mode not in CACHE_MODES:
        raise ConfigError(f"mcp_logged: HBENCH_BDIKIT_CACHE must be one of {CACHE_MODES}, not {mode!r}")
    models = [m.strip() for m in (env.get("HBENCH_TOOL_LLM_MODELS") or args.llm).split(",") if m.strip()]
    shared = Path(env.get("HBENCH_BDIKIT_SHARED") or Path.home() / ".cache" / "bdikit")
    return {"llm": args.llm, "run_dir": Path(run_dir), "run_id": run_id, "api_base": api_base,
            "allowed_models": models, "params": {k: _param(env, k) for k in TOOL_LLM_PARAMS},
            "cache_mode": mode, "cache_shared": shared}


# ------------------------------------------------------------------------------------------------------- stdout

def claim_stdout() -> int:
    """Keep the real stdout for JSON-RPC (returned fd) and send everything else written to fd 1 to stderr."""
    sys.stdout.flush()
    rpc_fd = os.dup(1)
    os.dup2(2, 1)
    return rpc_fd


# -------------------------------------------------------------------------------------------------------- cache

def setup_cache(cfg: dict) -> dict:
    """Point BDIKIT_CACHE at this run's cache (before bdikit is imported). Returns what the header records."""
    shared = cfg["cache_shared"]
    if cfg["cache_mode"] == "shared":
        root = shared
    else:
        root = cfg["run_dir"] / "bdikit_cache"
        root.mkdir(exist_ok=False)  # a fresh run never reuses a cache
        (root / "runs").mkdir()
        for sub in ("models", "embeddings"):  # deterministic, so shared (and never downloaded twice)
            (shared / sub).mkdir(parents=True, exist_ok=True)
            (root / sub).symlink_to(shared / sub, target_is_directory=True)
    os.environ["BDIKIT_CACHE"] = str(root)
    runs = root / "runs"
    return {"mode": cfg["cache_mode"], "root": str(root), "shared": str(shared),
            "runs_files_at_start": len(list(runs.iterdir())) if runs.is_dir() else 0}


MAGNETO_BASE_KEYS = ("mpnet", "roberta", "e5", "arctic", "minilm")  # magneto.embedding_matcher.DEFAULT_MODELS


def pin_models_dir(download, shared: Path) -> str:
    """BDI-Kit's model files are read from the shared models directory by its real path, as upstream does. Magneto
    picks its base model by substring of the weights' full path (magneto/embedding_matcher.py: the first of
    mpnet, roberta, e5, arctic, minilm found in it), so a path through the run directory, whose name has random
    hex, could silently switch the base model. Refuses a models path that contains any of those keys."""
    models = str((shared / "models").resolve())
    bad = [k for k in MAGNETO_BASE_KEYS if k in models]
    if bad:
        raise ConfigError(f"mcp_logged: models directory {models} contains {bad}; Magneto would pick the wrong "
                          "base model (set HBENCH_BDIKIT_SHARED to a path without these substrings)")
    download.BDIKIT_MODELS_CACHE_DIR = models
    return models


def patch_cache(api, mode: str) -> None:
    """Record every cache lookup and save on the tool call in progress; in `off` mode never hit or save."""
    orig_load, orig_save = api.load_from_cache, api.save_in_cache

    def load_from_cache(hash_id):
        obj = None if mode == "off" else orig_load(hash_id)
        cur = CURRENT.get()
        if cur is not None:
            c = cur["cache"]
            c["keys"].append(hash_id)
            c["hits" if obj is not None else "misses"] += 1
        return obj

    def save_in_cache(obj, hash_id):
        cur = CURRENT.get()
        if cur is not None:
            cur["cache"]["saves"] += 0 if mode == "off" else 1
        if mode != "off":
            orig_save(obj, hash_id)

    api.load_from_cache, api.save_in_cache = load_from_cache, save_in_cache


# ---------------------------------------------------------------------------------------------------- LLM guard

def resolve_api_base(model: str, kwargs: dict) -> str | None:
    """Where LiteLLM would send this call (the parts of its routing BDI-Kit can reach)."""
    base = kwargs.get("api_base") or kwargs.get("base_url")
    if base:
        return base
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    if provider == "openai":
        return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
    if provider in ("ollama", "ollama_chat"):
        return os.environ.get("OLLAMA_API_BASE") or "http://localhost:11434"
    return None  # a hosted provider (deepinfra/, anthropic/, ...): LiteLLM knows its cloud endpoint


def guard_kwargs(model: str, kwargs: dict, cfg: dict) -> tuple[dict, dict]:
    """The kwargs actually sent (local api_base, injected parameters) and what was overridden; raises
    LocalOnlyError for a disallowed model or a non-local endpoint."""
    if model not in cfg["allowed_models"]:
        raise LocalOnlyError(f"tool LLM call to model {model!r} refused: only {cfg['allowed_models']} are allowed "
                             "(BDI-Kit's matchers default to openai/gpt-4o-mini when no --llm reaches them)")
    out = dict(kwargs)
    if model.startswith("openai/") and not (out.get("api_base") or out.get("base_url")):
        out["api_base"] = cfg["api_base"]
    base = resolve_api_base(model, out)
    if base is None or (urlparse(base).hostname or "") not in LOCAL_HOSTS:
        raise LocalOnlyError(f"tool LLM call to non-local endpoint refused: model {model!r}, api_base {base!r}")
    if model.startswith("openai/"):
        out.setdefault("api_key", "ollama")  # placeholder: no real key is ever in this process
    overridden = {}
    for k, v in cfg["params"].items():
        if v is None:
            continue
        if k in out and out[k] != v:
            overridden[k] = out[k]
        out[k] = v
        if k == "top_logprobs":
            out["logprobs"] = True
    return out, overridden


def install_llm_guard(litellm, cfg: dict, log: RunLog) -> dict:
    """Replace litellm.completion/acompletion (and litellm.main's) with guarded, parameter-injecting versions."""
    state = {"refused": 0, "calls": 0}
    orig_c, orig_ac = litellm.completion, litellm.acompletion

    def _before(args, kwargs):
        model = kwargs.get("model") or (args[0] if args else None)
        rest = dict(kwargs)
        rest.pop("model", None)
        try:
            sent, overridden = guard_kwargs(str(model), rest, cfg)
        except LocalOnlyError as e:
            state["refused"] += 1
            cur = CURRENT.get()
            log.emit("error", kind="cloud_call_attempted", model=model, message=str(e),
                     parent_id=(cur or {}).get("start_span"), tool=(cur or {}).get("tool"))
            print(f"[hbench.mcp_logged] REFUSED: {e}", file=sys.stderr, flush=True)
            raise
        state["calls"] += 1
        cur = CURRENT.get()
        if cur is not None:
            cur["llm_calls"] += 1
            if overridden:
                cur["overridden"].append(overridden)
        return model, sent

    @functools.wraps(orig_c)
    def completion(*args, **kwargs):
        model, sent = _before(args, kwargs)
        t = time.perf_counter()
        try:
            return orig_c(model=model, **sent)
        finally:
            cur = CURRENT.get()
            if cur is not None:
                cur["llm_s"] += time.perf_counter() - t

    @functools.wraps(orig_ac)
    async def acompletion(*args, **kwargs):
        model, sent = _before(args, kwargs)
        t = time.perf_counter()
        try:
            return await orig_ac(model=model, **sent)
        finally:
            cur = CURRENT.get()
            if cur is not None:
                cur["llm_s"] += time.perf_counter() - t

    completion.hbench_guarded = acompletion.hbench_guarded = True
    litellm.completion, litellm.acompletion = completion, acompletion
    import litellm.main as lm
    lm.completion, lm.acompletion = completion, acompletion
    state["restore"] = lambda: (setattr(litellm, "completion", orig_c), setattr(litellm, "acompletion", orig_ac),
                                setattr(lm, "completion", orig_c), setattr(lm, "acompletion", orig_ac))
    return state


def unguarded_completion_refs() -> list[str]:
    """Modules that bound LiteLLM's completion before the guard (must be empty, else a call could escape)."""
    bad = []
    for name, mod in list(sys.modules.items()):
        if not (name.startswith("bdikit") or name.startswith("magneto")):
            continue
        fn = getattr(mod, "completion", None)
        if callable(fn) and getattr(fn, "__module__", "").startswith("litellm") and \
                not getattr(fn, "hbench_guarded", False):
            bad.append(name)
    return bad


# ------------------------------------------------------------------------------------------------ tool wrapping

def rows_of(result) -> list:
    if result is None:
        return []
    return list(result) if isinstance(result, (list, tuple)) else [result]


def result_hashes(rows: list) -> tuple[str, str]:
    """(result_sha256, result_set_sha256): the rows in order, and the multiset of rows (order-insensitive:
    BDI-Kit's deterministic matchers reorder rows across processes, E0 design 2.2)."""
    return sha256_json(rows), sha256_json(sorted(sha256_json(r) for r in rows))


def df_records(df) -> list | None:
    if df is None:
        return None
    from ..predictions import jsonable
    return jsonable(df.to_dict(orient="records"))


def emit_snapshot(log: RunLog, server, parent_id: str | None, server_seq: int | None) -> str:
    fields = {}
    for name in ("schema_matching_results", "value_matching_results"):
        rows = df_records(getattr(server, name, None))
        fields[name] = None if rows is None else log.blobs.put(rows)
        fields[f"{name}_rows"] = None if rows is None else len(rows)
        fields[f"{name}_sha256"], fields[f"{name}_set_sha256"] = (None, None) if rows is None else result_hashes(rows)
    return log.emit("state_snapshot", parent_id=parent_id, server_seq=server_seq, **fields)


def _default_method(tool) -> str | None:
    return ((tool.parameters or {}).get("properties", {}).get("method") or {}).get("default")


def wrap_tools(server, log: RunLog, cb=None) -> list[str]:
    """Log every call of every tool of a FastMCP server. Returns the wrapped tool names."""
    tm = server._tool_manager
    tools = {t.name: t for t in tm.list_tools()}
    for t in tools.values():
        orig = t.fn

        async def fn(*a, __orig=orig, **kw):
            res = await __orig(*a, **kw)
            cur = CURRENT.get()
            if cur is not None:
                cur["result"], cur["has_result"] = res, True
            return res
        t.fn = functools.wraps(orig)(fn)
    orig_call = tm.call_tool
    counter = {"n": 0}

    async def call_tool(name, arguments, context=None, convert_result=False):
        counter["n"] += 1
        seq = counter["n"]
        tool = tools.get(name)
        method = (arguments or {}).get("method") or (_default_method(tool) if tool else None)
        caller = f"tool:{name}/{method}" if method else f"tool:{name}"
        args_sha = sha256_json(arguments or {})
        start = log.emit("note", kind="tool_start", tool=name, server_seq=seq, args_sha256=args_sha)
        cur = {"start_span": start, "tool": name, "cache": {"keys": [], "hits": 0, "misses": 0, "saves": 0},
               "llm_calls": 0, "llm_s": 0.0, "overridden": [], "result": None, "has_result": False}
        token = CURRENT.set(cur)
        t0, err = time.perf_counter(), None
        try:
            from .litellm_cb import caller as caller_ctx, parent_span
            with caller_ctx(caller), parent_span(start):
                return await orig_call(name, arguments, context=context, convert_result=convert_result)
        except BaseException as e:  # noqa: BLE001 - logged, then re-raised unchanged
            err = e
            raise
        finally:
            duration = time.perf_counter() - t0
            CURRENT.reset(token)
            if cb is not None:
                cb.wait(30)  # the tool's LLM calls are on disk before its tool_call
            rows = rows_of(cur["result"]) if cur["has_result"] else None
            from ..predictions import jsonable
            rows = None if rows is None else jsonable(rows)
            h, hs = result_hashes(rows) if rows is not None else (None, None)
            cause = (err.__cause__ or err) if err is not None else None
            tb = "".join(traceback.format_exception(cause)) if cause is not None else None
            snap = emit_snapshot(log, server, start, seq)
            log.emit("tool_call", side="server", tool=name, method=method, server_seq=seq, start_span=start,
                     args=arguments, args_sha256=args_sha, result=rows, result_sha256=h, result_set_sha256=hs,
                     result_rows=None if rows is None else len(rows), duration_s=round(duration, 3),
                     llm_calls=cur["llm_calls"], llm_s=round(cur["llm_s"], 3), llm_params_overridden=cur["overridden"],
                     bdikit_cache=cur["cache"], state_after=snap,
                     error=None if err is None else f"{type(err).__name__}: {err}"[:4000],
                     error_type=None if cause is None else type(cause).__name__,
                     traceback=tb, traceback_sha256=None if tb is None else hashlib.sha256(tb.encode()).hexdigest())
    tm.call_tool = call_tool
    return sorted(tools)


# --------------------------------------------------------------------------------------------------------- main

async def serve(server, rpc_fd: int) -> None:
    import io

    import anyio
    from mcp.server.stdio import stdio_server

    out = anyio.wrap_file(io.TextIOWrapper(os.fdopen(rpc_fd, "wb"), encoding="utf-8"))
    async with stdio_server(stdout=out) as (read, write):
        await server._mcp_server.run(read, write, server._mcp_server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    rpc_fd = claim_stdout()
    cfg = config_from_env(argv)
    # A fixed process name: seeds of sampled calls are scoped by process (hbench.seeds), so a pid here would make
    # the tool LLM's seeds differ between two runs with the same base seed. Each event still carries its pid.
    log = RunLog.attach(cfg["run_dir"], proc="mcp")
    status, state, cb = "ok", None, None
    try:
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no network fetch of the price map
        cache = setup_cache(cfg)
        from .header import env_info, package_info, python_info
        log.emit("run_header", role="mcp_server", argv=sys.argv[1:], llm=cfg["llm"],
                 tool_llm={"api_base": cfg["api_base"], "allowed_models": cfg["allowed_models"], **cfg["params"]},
                 bdikit_cache=cache, python=python_info(), packages=package_info(), env=env_info())
        import litellm
        state = install_llm_guard(litellm, cfg, log)
        from . import litellm_cb
        cb = litellm_cb.install(log, caller="tool")
        import bdikit.api as bdi_api
        from bdikit import mcp_server
        # the matchers import LiteLLM lazily (matcher_factory); load them now, so the check below covers them
        import bdikit.schema_matching.llm  # noqa: F401
        import bdikit.value_matching.llm  # noqa: F401
        import bdikit.value_matching.llm_numeric  # noqa: F401
        import magneto.llm_reranker  # noqa: F401
        escaped = unguarded_completion_refs()
        if escaped:
            raise RuntimeError(f"LiteLLM completion bound before the guard in: {escaped}")
        import bdikit.download as bdi_download
        models_dir = pin_models_dir(bdi_download, cfg["cache_shared"])
        patch_cache(bdi_api, cfg["cache_mode"])
        names = wrap_tools(mcp_server.server, log, cb)
        mcp_server.server.kwargs = {"llm": cfg["llm"]}  # what run_mcp_server() would set from --llm
        log.emit("note", kind="mcp_ready", tools=names, models_dir=models_dir)
        import anyio
        anyio.run(serve, mcp_server.server, rpc_fd)
    except BaseException as e:  # noqa: BLE001
        status = "error"
        log.emit("error", kind="mcp_server_failed", message=f"{type(e).__name__}: {e}")
        raise
    finally:
        if cb is not None:
            cb.wait(30)
        runs = Path(os.environ.get("BDIKIT_CACHE", "")) / "runs"
        log.close(status=status, tool_llm_calls=(state or {}).get("calls"),
                  cloud_calls_refused=(state or {}).get("refused"),
                  bdikit_runs_files_at_end=len(list(runs.iterdir())) if runs.is_dir() else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
