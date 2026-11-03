"""LiteLLM callback that writes every completion (Magneto's reranker, BDI-Kit matchers, agents) as an llm_call.

    from hbench.runlog import litellm_cb
    cb = litellm_cb.install(log, caller="method:magneto")
    ...                                   # any litellm.completion / acompletion in this process
    litellm_cb.uninstall(cb)

Besides the messages LiteLLM was given, it records the request body actually sent to the server
(additional_args["complete_input_dict"]): routes rewrite messages, e.g. LiteLLM 1.82.6's ollama_chat/
route drops assistant tool_calls. Calls to a non-local api_base are flagged with cloud_call=True.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from urllib.parse import urlparse

from litellm.integrations.custom_logger import CustomLogger

from .llm import log_llm_call
from .writer import RunLog

PARENT_SPAN: contextvars.ContextVar[str | None] = contextvars.ContextVar("hbench_parent_span", default=None)
STEP: contextvars.ContextVar[int | None] = contextvars.ContextVar("hbench_step", default=None)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_LOCAL_BY_DEFAULT = {"ollama", "ollama_chat"}  # LiteLLM's default api_base for these is localhost:11434


@contextlib.contextmanager
def parent_span(span_id: str | None, step: int | None = None):
    """LLM calls made inside this block are logged with this parent span (and step).
    LiteLLM copies the context into its callback threads, so this also works for sync calls."""
    t1, t2 = PARENT_SPAN.set(span_id), STEP.set(step)
    try:
        yield
    finally:
        PARENT_SPAN.reset(t1)
        STEP.reset(t2)


def is_local(api_base: str | None, provider: str | None) -> bool:
    if not api_base:
        return (provider or "") in _LOCAL_BY_DEFAULT
    return (urlparse(api_base).hostname or "") in LOCAL_HOSTS


def _plain(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "model_dump"):
        return _plain(x.model_dump())
    if hasattr(x, "dict"):
        return _plain(x.dict())
    return str(x)


class HBenchLiteLLMLogger(CustomLogger):
    def __init__(self, log: RunLog, caller: str = "litellm"):
        super().__init__()
        self.log = log
        self.caller = caller
        self._done: set[str] = set()
        self._requests: dict[str, object] = {}
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def log_pre_api_call(self, model, messages, kwargs):
        # Some routes (e.g. ollama_chat in 1.82.6) replace additional_args after the call, dropping the
        # request body, so keep it from here.
        body = (kwargs.get("additional_args") or {}).get("complete_input_dict")
        call_id = kwargs.get("litellm_call_id")
        if call_id:
            with self._lock:
                self._pending.add(call_id)
                if body is not None:
                    self._requests[call_id] = _plain(body)

    def wait(self, timeout: float = 10.0) -> bool:
        """Block until every call seen by log_pre_api_call has been logged (LiteLLM reports successes from a
        background thread). False on timeout."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if not self._pending:
                    return True
            time.sleep(0.05)
        return False

    def _emit(self, kwargs, response_obj, start_time, end_time, failed: bool):
        call_id = kwargs.get("litellm_call_id")
        with self._lock:  # LiteLLM may report one call through both its sync and async handlers
            if call_id in self._done:
                return
            if call_id:
                self._done.add(call_id)
                self._pending.discard(call_id)
            request_body = self._requests.pop(call_id, None)
        if request_body is None:
            request_body = _plain((kwargs.get("additional_args") or {}).get("complete_input_dict"))
        slo = kwargs.get("standard_logging_object") or {}
        lp = kwargs.get("litellm_params") or {}
        provider = slo.get("custom_llm_provider") or lp.get("custom_llm_provider")
        api_base = slo.get("api_base") or lp.get("api_base")
        response, usage, error = None, None, None
        if failed:
            exc = kwargs.get("exception")
            error = f"{type(exc).__name__}: {exc}" if exc is not None else (slo.get("error_str") or "unknown error")
        else:
            try:
                msg = response_obj.choices[0].message
                response = {"content": getattr(msg, "content", None),
                            "tool_calls": _plain(getattr(msg, "tool_calls", None)),
                            "finish_reason": getattr(response_obj.choices[0], "finish_reason", None),
                            "model": getattr(response_obj, "model", None)}
            except Exception:  # noqa: BLE001 - embeddings or unexpected shapes
                response = {"raw": _plain(response_obj)}
            u = getattr(response_obj, "usage", None)
            usage = {"prompt_tokens": getattr(u, "prompt_tokens", slo.get("prompt_tokens")),
                     "completion_tokens": getattr(u, "completion_tokens", slo.get("completion_tokens"))}
        latency = None
        if start_time is not None and end_time is not None:
            try:
                latency = round((end_time - start_time).total_seconds(), 3)
            except AttributeError:
                latency = round(float(end_time) - float(start_time), 3)
        log_llm_call(self.log, caller=self.caller, provider=provider, model=kwargs.get("model") or slo.get("model"),
                     api_base=api_base, endpoint=None, messages=_plain(kwargs.get("messages") or slo.get("messages")),
                     params=_plain(slo.get("model_parameters") or kwargs.get("optional_params") or {}),
                     response=response, usage=usage, latency_s=latency,
                     request_body=request_body, error=error, parent_id=PARENT_SPAN.get(), step=STEP.get(), litellm_call_id=call_id,
                     cloud_call=not is_local(api_base, provider))

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._emit(kwargs, response_obj, start_time, end_time, failed=False)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._emit(kwargs, response_obj, start_time, end_time, failed=True)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._emit(kwargs, response_obj, start_time, end_time, failed=False)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._emit(kwargs, response_obj, start_time, end_time, failed=True)


def install(log: RunLog, caller: str = "litellm") -> HBenchLiteLLMLogger:
    import litellm
    cb = HBenchLiteLLMLogger(log, caller)
    litellm.callbacks.append(cb)
    return cb


def uninstall(cb: HBenchLiteLLMLogger, timeout: float = 10.0) -> bool:
    """Wait for pending callbacks, then remove the logger. False if some call was never reported."""
    import litellm
    ok = cb.wait(timeout)
    if cb in litellm.callbacks:
        litellm.callbacks.remove(cb)
    return ok
