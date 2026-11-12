"""LangChain callback that logs an agent's step structure into an hbench run (the driver side of an agent run).

    handler = AgentStepLogger(log, llm_logger=cb)       # cb: the hbench LiteLLM logger of the same process
    await agent.ainvoke(..., config={"callbacks": [handler], ...})

Events:
  agent_step   one per agent LLM call: step number, wall time, the tool calls it issued (names, arguments, ids),
               text length, finish reason, usage, the hash of the messages it was given, and failure flags
               (text_json_tool_call: a tool call written as JSON text, which LangGraph takes for a final answer;
               non_ascii_share: language drift)
  tool_call    side="client": what the agent received for each tool call (tool_call_id, arguments, the result as
               rows, result_sha256 / result_set_sha256 as on the server side, status, wall time); parent_id is
               the agent_step that issued it

The LiteLLM logger records the raw request and response of each agent call (llm_call, caller "agent"); this logger
gives both the same step number (AgentLLMLogger numbers calls in the order they are sent, which is the order of
the agent's steps, since a ReAct agent makes one LLM call at a time).
"""

from __future__ import annotations

import json
import re
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler

from .litellm_cb import STEP, HBenchLiteLLMLogger
from .mcp_logged import result_hashes
from .writer import RunLog, sha256_json

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```|<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)


class AgentLLMLogger(HBenchLiteLLMLogger):
    """The LiteLLM logger for an agent's own calls: numbers them 1, 2, ... in the order they are sent and logs
    each llm_call with that step."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._step_of: dict[str, int] = {}
        self.last_step = 0
        self._step_lock = threading.Lock()

    def log_pre_api_call(self, model, messages, kwargs):
        call_id = kwargs.get("litellm_call_id")
        with self._step_lock:
            if call_id and call_id not in self._step_of:
                self.last_step += 1
                self._step_of[call_id] = self.last_step
        super().log_pre_api_call(model, messages, kwargs)

    def _write(self, kwargs, *a, **kw):
        token = STEP.set(self._step_of.get(kwargs.get("litellm_call_id")))
        try:
            super()._write(kwargs, *a, **kw)
        finally:
            STEP.reset(token)


def text_json_tool_call(text: str | None) -> dict | None:
    """A tool call written as JSON text ({"tool": ..., "parameters": ...} as BDI-Kit's prompt asks, or
    {"name": ..., "arguments": ...}), bare, in a ```json block, or in a raw <tool_call> block that the server did
    not parse into tool_calls (an Ollama template whose tool-call prefix it cannot infer); None otherwise."""
    if not text:
        return None
    cands = [m.group(1) or m.group(2) for m in _JSON_BLOCK.finditer(text)] or [text.strip()]
    for c in cands:
        try:
            obj = json.loads(c)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict) and (("tool" in obj and "parameters" in obj) or ("name" in obj and "arguments" in obj)):
            return obj
    return None


def non_ascii_share(text: str | None) -> float | None:
    if not text:
        return None
    letters = [c for c in text if c.isalpha()]
    return round(sum(ord(c) > 127 for c in letters) / len(letters), 4) if letters else 0.0


def message_dict(m) -> dict:
    from langchain_core.messages import message_to_dict
    try:
        return message_to_dict(m)
    except Exception:  # noqa: BLE001
        return {"type": type(m).__name__, "content": str(getattr(m, "content", m))}


def tool_rows(content) -> list:
    """The rows of a tool result as the agent received them: each MCP text block holds one JSON row."""
    blocks = content if isinstance(content, list) else [content]
    rows = []
    for b in blocks:
        text = b.get("text") if isinstance(b, dict) else b
        try:
            rows.append(json.loads(text))
        except (TypeError, json.JSONDecodeError):
            rows.append(text)
    return rows


class AgentStepLogger(BaseCallbackHandler):
    run_inline = True  # called in the agent's own task, in order
    raise_error = False

    def __init__(self, log: RunLog, llm_logger: AgentLLMLogger | None = None):
        self.log = log
        self.llm_logger = llm_logger
        self.steps: list[dict] = []   # per-step timing, also returned to the driver
        self.tools: list[dict] = []
        self._llm: dict = {}          # run_id -> (start, n, input hash)
        self._tool: dict = {}         # run_id -> (start, name, args, tool_call_id)
        self._issuer: dict = {}       # tool_call_id -> agent_step span
        self.n_steps = 0
        self.turn = 0

    # -- agent LLM calls
    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        msgs = [message_dict(m) for m in (messages[0] if messages else [])]
        self._llm[run_id] = (time.perf_counter(), time.time(), len(msgs), sha256_json(msgs))

    def _end(self, run_id, response=None, error=None):
        t0, wall0, n_in, in_sha = self._llm.pop(run_id, (None, None, None, None))
        dur = None if t0 is None else round(time.perf_counter() - t0, 3)
        self.n_steps += 1
        msg = None
        if response is not None and response.generations and response.generations[0]:
            msg = getattr(response.generations[0][0], "message", None)
        text = None if msg is None else (msg.content if isinstance(msg.content, str) else json.dumps(msg.content))
        calls = [] if msg is None else [{"name": c.get("name"), "args": c.get("args"), "id": c.get("id")}
                                        for c in (getattr(msg, "tool_calls", None) or [])]
        invalid = [] if msg is None else [dict(c) for c in (getattr(msg, "invalid_tool_calls", None) or [])]
        meta = {} if msg is None else (getattr(msg, "response_metadata", None) or {})
        jtc = None if calls else text_json_tool_call(text)
        ak = {} if msg is None else (getattr(msg, "additional_kwargs", None) or {})
        reasoning = ak.get("reasoning_content") or ak.get("reasoning")
        llm_step = self.llm_logger.last_step if self.llm_logger is not None else None
        span = self.log.emit(
            "agent_step", step=self.n_steps, turn=self.turn, llm_step=llm_step, duration_s=dur, started_at=wall0,
            n_input_messages=n_in, input_messages_sha256=in_sha, tool_calls=calls, n_tool_calls=len(calls),
            invalid_tool_calls=invalid, text_chars=len(text or ""), text=text,
            reasoning_chars=len(reasoning) if isinstance(reasoning, str) else None,
            finish_reason=meta.get("finish_reason"), usage=None if msg is None else getattr(msg, "usage_metadata", None),
            text_json_tool_call=jtc, non_ascii_share=non_ascii_share(text),
            error=None if error is None else f"{type(error).__name__}: {error}")
        for c in calls:
            if c.get("id"):
                self._issuer[c["id"]] = span
        self.steps.append({"step": self.n_steps, "turn": self.turn, "llm_s": dur, "n_tool_calls": len(calls),
                           "tools": [c["name"] for c in calls], "prompt_tokens":
                           ((getattr(msg, "usage_metadata", None) or {}).get("input_tokens") if msg else None),
                           "completion_tokens":
                           ((getattr(msg, "usage_metadata", None) or {}).get("output_tokens") if msg else None),
                           "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
                           "text_json_tool_call": jtc is not None, "error": error is not None})

    def on_llm_end(self, response, *, run_id, **kw):
        self._end(run_id, response=response)

    def on_llm_error(self, error, *, run_id, **kw):
        self._end(run_id, error=error)

    # -- tool calls as the agent sees them
    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, tool_call_id=None, **kw):
        name = (serialized or {}).get("name") or kw.get("name")
        args = inputs if inputs is not None else input_str
        self._tool[run_id] = (time.perf_counter(), time.time(), name, args, tool_call_id)

    def _tool_end(self, run_id, output=None, error=None):
        t0, wall0, name, args, tcid = self._tool.pop(run_id, (None, None, None, None, None))
        dur = None if t0 is None else round(time.perf_counter() - t0, 3)
        content = getattr(output, "content", output) if output is not None else None
        tcid = tcid or getattr(output, "tool_call_id", None)
        status = getattr(output, "status", None) or ("error" if error is not None else "success")
        rows = None if content is None else tool_rows(content)
        h, hs = result_hashes(rows) if rows is not None else (None, None)
        self.log.emit("tool_call", parent_id=self._issuer.get(tcid), step=self.n_steps, side="client", tool=name,
                      tool_call_id=tcid, args=args, args_sha256=sha256_json(args), result=rows, result_sha256=h,
                      result_set_sha256=hs, result_rows=None if rows is None else len(rows), status=status,
                      duration_s=dur, started_at=wall0,
                      error=None if error is None else f"{type(error).__name__}: {error}"[:4000])
        self.tools.append({"step": self.n_steps, "turn": self.turn, "tool": name, "tool_s": dur, "status": status})

    def on_tool_end(self, output, *, run_id, **kw):
        self._tool_end(run_id, output=output)

    def on_tool_error(self, error, *, run_id, **kw):
        self._tool_end(run_id, error=error)
