"""E3 driver: BDI-Kit's chat agent, headless, on a local model, with every step logged through hbench.

The agent is rebuilt exactly as bdikit/chatbot.py builds it (that module relaunches itself under Streamlit, so it
cannot be imported): its system prompt is read from the installed chatbot.py with ast, the model is
ChatLiteLLM(max_retries=1, timeout=300), the agent is create_react_agent(llm, tools, prompt=PROMPT,
checkpointer=InMemorySaver()) on thread "1", and the tools are BDI-Kit's MCP server over stdio. Differences from
the Streamlit app, all recorded in the condition:
  - the MCP server is hbench.runlog.mcp_logged (BDI-Kit's server, wrapped for logging; same tools and code);
  - the model is reached through LiteLLM's openai/ route on Ollama's /v1, with temperature and top_p stated
    (the app states neither) and token logprobs requested, for decision evidence;
  - streaming is off by default, so the logged response is the one the agent used (--streaming to turn it on);
  - every run has its own empty BDI-Kit result cache.

One run = one hbench run = a fresh MCP server process. The user turns are scripted (tasks.py): the task, then the
fixed follow-up, at most --max-followups times, whenever a turn ends without a final markdown table (the agent
asked a question, only suggested a step, wrote a tool call as JSON text, or hit the recursion limit). Nothing ever
waits for a human. The final mapping is the MCP server's last state snapshot (hbench.adapters.bdikit_agent);
the agent's final text is kept as its claims (claims.md) for E7.

Usage (repo root, the harmonize env):
  python -s experiments/e3_agent/run_agent.py --study clark --task sm --temperature 0 --top-p 1 --seed 1
Writes <raw root>/<run_id>/: events.driver.jsonl, events.mcp.<pid>.jsonl, predictions.jsonl, timing.json,
transcript.json, claims.md, mcp_server.stderr.log, engine_audit.json. Prints the run directory.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))  # this checkout's hbench (its commit is in the run header)
sys.path.insert(0, str(HERE))

from hbench.data import Data  # noqa: E402
from hbench.design import make_condition  # noqa: E402
from hbench.predictions import write_predictions  # noqa: E402
from hbench.runlog.session import raw_root, start_run  # noqa: E402
from hbench.runlog.writer import read_events, sha256_json  # noqa: E402
from tasks import FOLLOWUP, TASK_KIND, TASK_VERSION, TASKS, VM_MODE, task_text  # noqa: E402

DEFAULT_MODEL = "qwen2.5-14b-bdikit"  # the derived model (Modelfile.qwen2.5-14b-bdikit): fixed num_ctx and template
LOCAL = {"localhost", "127.0.0.1", "::1"}
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
AGENT_DECISION = [{"type": "tool_call", "field": "name"}]  # an agent step decides which tool it calls next
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.M)
OTHER_CLIENTS = ("run_plan.py", "tool_loop.py", "stage1.py", "magneto_ollama.py", "clark_match.py")


def chatbot_prompt() -> tuple[str, str, str]:
    """(system prompt, default --llm, chatbot.py sha256), read from the installed chatbot.py without importing it."""
    import importlib.util
    path = Path(importlib.util.find_spec("bdikit").origin).parent / "chatbot.py"
    src = path.read_text()
    prompt, default_llm = None, None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "prompt" for t in node.targets):
            prompt = ast.literal_eval(node.value)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument" and \
                node.args and ast.literal_eval(node.args[0]) == "--llm":
            default_llm = next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "default")
    if not prompt:
        raise RuntimeError(f"no prompt found in {path}")
    return prompt, default_llm, hashlib.sha256(src.encode()).hexdigest()


# A declared deviation (BDI-Kit agent test plan 7, item 5(ii)): the prompt's response-format line asks for tool calls
# as JSON text, which qwen2.5 follows instead of making native tool calls (first live E3 run, risk R5).
JSON_FORMAT_LINE = "- If calling a tool, output a JSON object with `tool` and `parameters`\n"
PROMPT_VARIANTS = ("verbatim", "no_json_format")


def apply_prompt_variant(prompt: str, variant: str) -> str:
    """The system prompt of a variant: verbatim, or without the JSON-format line (refuses if it is not there)."""
    if variant == "verbatim":
        return prompt
    if variant == "no_json_format":
        if prompt.count(JSON_FORMAT_LINE) != 1:
            raise RuntimeError("the installed chatbot.py prompt no longer has the JSON-format line exactly once")
        return prompt.replace(JSON_FORMAT_LINE, "")
    raise ValueError(f"unknown prompt variant {variant!r}")


def has_final_table(text: str | None) -> bool:
    return bool(text) and bool(_TABLE_SEP.search(text))


def message_text(content) -> tuple[str | None, str | None]:
    """The visible text and the reasoning of an AIMessage content: a string as is; a list of blocks (a thinking
    model's reply via langchain_litellm: [{"type": "thinking"}, {"type": "text"}]) as its text blocks joined by
    newlines, and its thinking blocks likewise. The turn checks (final table, question, JSON tool call, language)
    and claims.md use the visible text."""
    if content is None or isinstance(content, str):
        return content, None
    texts, thoughts = [], []
    for b in content:
        if isinstance(b, str):
            texts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            texts.append(b.get("text", ""))
        elif isinstance(b, dict) and b.get("type") in THINKING_BLOCKS:
            thoughts.append(b.get("thinking", ""))
    return "\n".join(texts), ("\n".join(thoughts) if thoughts else None)


def stage_input(data: Data, study: str, root: Path) -> Path:
    """The source CSV at a fixed path without spaces (the same text in every run's task prompt)."""
    src = data.source_path(study)
    dst = root / "_inputs" / "e3" / f"{study}.csv"  # every study's file is source.csv
    if re.search(r"\s", str(dst)):
        raise SystemExit(f"input path {dst} contains whitespace; set --raw-dir to a path without spaces")
    want = hashlib.sha256(src.read_bytes()).hexdigest()
    if dst.exists():
        if hashlib.sha256(dst.read_bytes()).hexdigest() != want:
            raise SystemExit(f"{dst} differs from {src}; remove it and rerun")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    return dst


def other_ollama_clients() -> list[str]:
    try:
        out = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    me = str(os.getpid())
    return [ln.strip() for ln in out.splitlines()
            if any(c in ln for c in OTHER_CLIENTS) and not ln.strip().startswith(me) and "grep" not in ln]


def clean_env(base: dict, **extra) -> dict:
    """The environment without any secret-like variable (llm_numeric execs model-written code), plus extra."""
    env = {k: v for k, v in base.items() if not any(m in k.upper() for m in SECRET_MARKERS)}
    env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study", default="clark")
    ap.add_argument("--task", choices=TASKS, default="sm")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name (agent and tool matchers)")
    ap.add_argument("--temperature", type=float, required=True, help="agent temperature (always stated)")
    ap.add_argument("--top-p", type=float, required=True, help="agent top_p (always stated)")
    ap.add_argument("--seed", default=None, help="integer, 'per_replicate', or omit")
    ap.add_argument("--tool-temperature", type=float, default=None, help="tool LLM temperature (default: agent's)")
    ap.add_argument("--tool-top-p", type=float, default=None, help="tool LLM top_p (default: agent's)")
    ap.add_argument("--top-logprobs", type=int, default=5, help="logprob candidates per token (0: none)")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--block", default=None)
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--engine-protocol", default="cold_start", choices=["cold_start", "natural"])
    ap.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="verbatim",
                    help="no_json_format: drop the prompt line asking for tool calls as JSON text (a declared deviation)")
    ap.add_argument("--history-thinking", choices=["strip", "keep"], default="strip",
                    help="strip: drop past thinking blocks from the history sent back (Ollama /v1 rejects them)")
    ap.add_argument("--recursion-limit", type=int, default=25, help="LangGraph's default, as in chatbot.py")
    ap.add_argument("--max-followups", type=int, default=2)
    ap.add_argument("--turn-timeout", type=float, default=3600.0, help="seconds per user turn before giving up")
    ap.add_argument("--streaming", action="store_true", help="stream like chatbot.py (logging may lose logprobs)")
    ap.add_argument("--bdikit-cache", choices=["run", "off", "shared"], default="run")
    ap.add_argument("--bdikit-shared", default=None, help="shared BDI-Kit cache with models/ and embeddings/")
    ap.add_argument("--hashseed", default="0", help="PYTHONHASHSEED of the MCP server ('random' to leave unset)")
    ap.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--no-lease", action="store_true", help=argparse.SUPPRESS)  # tests: no lease, no audit
    ap.add_argument("--allow-other-clients", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.seed not in (None, "per_replicate"):
        a.seed = int(a.seed)
    if (urlparse(a.ollama_base).hostname or "") not in LOCAL:
        ap.error(f"--ollama-base {a.ollama_base} is not local")
    return a


THINKING_BLOCKS = ("thinking", "redacted_thinking")


def strip_history_thinking(message_dicts: list[dict]) -> tuple[list[dict], int]:
    """Drop thinking blocks from the assistant messages of the history sent to the model.

    langchain_litellm 0.6.4 stores a reply's reasoning as a {"type": "thinking"} block in the AIMessage content and
    sends it back as is; Ollama's /v1 rejects such a message ("invalid message format"), so with any thinking model
    the agent dies at its second step. Qwen's own chat template drops past thinking anyway. The remaining blocks are
    joined back into a plain string when they are all text. Returns the new list and the number of blocks dropped."""
    out, n = [], 0
    for m in message_dicts:
        c = m.get("content")
        if m.get("role") == "assistant" and isinstance(c, list):
            kept = [b for b in c if not (isinstance(b, dict) and b.get("type") in THINKING_BLOCKS)]
            n += len(c) - len(kept)
            if len(kept) != len(c):
                if all(isinstance(b, dict) and b.get("type") == "text" for b in kept):
                    kept = "".join(b.get("text", "") for b in kept)
                m = {**m, "content": kept}
        out.append(m)
    return out, n


def renumber_tool_call_ids(message_dicts: list[dict]) -> tuple[list[dict], int]:
    """Replace the tool-call ids in the history sent to the model with fixed names: call_<k>_<j> for the j-th tool
    call of the k-th assistant message, in both the assistant's tool_calls and the matching tool messages.

    Ollama gives every tool call a random id and the next request echoes it back, so the request body (the prompt
    hash, and with it a sampled call's per-call seed, hbench.seeds) would differ in every run from the second agent
    call on. The names depend only on the position in the history, so they are the same in every request of a run
    and in every run. The agent's own messages keep Ollama's ids (agent_step and client tool_call events log those);
    only the request copy is renamed. Ids of tool messages with no matching call are left alone. Returns the new
    list and the number of ids renamed."""
    out, ren, k = [], {}, 0
    for m in message_dicts:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for j, tc in enumerate(m["tool_calls"]):
                if isinstance(tc, dict) and tc.get("id") is not None:
                    ren[tc["id"]] = f"call_{k}_{j}"
                    tc = {**tc, "id": ren[tc["id"]]}
                tcs.append(tc)
            m = {**m, "tool_calls": tcs}
            k += 1
        elif m.get("role") == "tool" and m.get("tool_call_id") in ren:
            m = {**m, "tool_call_id": ren[m["tool_call_id"]]}
        out.append(m)
    return out, len(ren)


def drop_block_ids(message_dicts: list[dict]) -> tuple[list[dict], int]:
    """Drop the random "id" ("lc_<uuid>") that langchain_mcp_adapters puts on every content block of a tool result.

    It is LangChain bookkeeping, not part of the message: Ollama renders only the block's text, so the model input is
    unchanged, but the id is in the request body, so the prompt hash (and with it a sampled call's per-call seed,
    hbench.seeds) would differ in every run. Returns the new list and the number of ids dropped."""
    out, n = [], 0
    for m in message_dicts:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(b, dict) and str(b.get("id", "")).startswith("lc_") for b in c):
            new = []
            for b in c:
                if isinstance(b, dict) and str(b.get("id", "")).startswith("lc_"):
                    b = {k: v for k, v in b.items() if k != "id"}
                    n += 1
                new.append(b)
            m = {**m, "content": new}
        out.append(m)
    return out, n


def build_condition(a) -> dict:
    tool_t = a.temperature if a.tool_temperature is None else a.tool_temperature
    tool_p = a.top_p if a.tool_top_p is None else a.tool_top_p
    return make_condition(
        method={"name": "bdikit_agent", "bdikit": "0.11.0", "task": a.task, "prompt_id": TASK_VERSION,
                "prompt_variant": a.prompt_variant, "recursion_limit": a.recursion_limit, "max_followups": a.max_followups, "tool_output": "raw",
                "history_clean": False, "history_thinking": a.history_thinking,
                "content_block_ids": "dropped", "tool_call_ids": "renumbered", "mcp_server": "hbench.runlog.mcp_logged"},
        model={"name": a.model, "tool_llm": a.model},
        decoding={"temperature": a.temperature, "top_p": a.top_p, "seed": a.seed,
                  "tool_llm_temperature": tool_t, "tool_llm_top_p": tool_p},
        engine={"transport": "litellm/openai", "protocol": a.engine_protocol, "streaming": a.streaming,
                "top_logprobs": a.top_logprobs or None, "bdikit_cache": a.bdikit_cache,
                "pythonhashseed": None if a.hashseed == "random" else a.hashseed},
        data={"study": a.study}, replicate=a.replicate, block=a.block, order=a.order)


async def drive(a, log, prompt: str, task_msg: str, timing: dict) -> dict:
    """Start the MCP server and the agent, play the scripted turns, return what happened."""
    from langchain_core.messages import AIMessage, messages_to_dict
    from langchain_litellm import ChatLiteLLM
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from hbench.runlog import litellm_cb
    from hbench.runlog.langchain_cb import AgentLLMLogger, AgentStepLogger, non_ascii_share, text_json_tool_call

    cond = log.condition
    dec = cond["decoding"]
    v1 = a.ollama_base.rstrip("/") + "/v1"
    llm_name = f"openai/{a.model}"
    seed = log.effective_decoding.get("seed")
    tool_t, tool_p = dec["tool_llm_temperature"], dec["tool_llm_top_p"]
    env = clean_env(os.environ, OPENAI_API_KEY="ollama", OPENAI_BASE_URL=v1, LITELLM_LOCAL_MODEL_COST_MAP="True",
                    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", BDIKIT_DEVICE=os.environ.get("BDIKIT_DEVICE", "cpu"),
                    HBENCH_RUN_DIR=log.run_dir, HBENCH_RUN_ID=log.run_id, HBENCH_TOOL_LLM_API_BASE=v1,
                    HBENCH_TOOL_LLM_MODELS=llm_name,
                    HBENCH_TOOL_LLM_TEMPERATURE="default" if tool_t is None else tool_t,
                    HBENCH_TOOL_LLM_TOP_P="default" if tool_p is None else tool_p,
                    HBENCH_TOOL_LLM_SEED="default" if seed is None else seed,
                    HBENCH_TOOL_LLM_TOP_LOGPROBS=a.top_logprobs or "default",
                    HBENCH_BDIKIT_CACHE=a.bdikit_cache, HBENCH_BDIKIT_SHARED=a.bdikit_shared,
                    PYTHONPATH=os.pathsep.join([str(REPO)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    env.pop("BDIKIT_CACHE", None)  # mcp_logged sets it per run
    if a.hashseed == "random":
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = a.hashseed
    params = StdioServerParameters(command=sys.executable,
                                   args=["-s", "-m", "hbench.runlog.mcp_logged", "--llm", llm_name],
                                   env=env, cwd=str(REPO))
    model_kwargs = {"top_p": dec["top_p"]}
    if seed is not None:
        model_kwargs["seed"] = seed
    if a.top_logprobs:
        model_kwargs.update(logprobs=True, top_logprobs=a.top_logprobs)
    import litellm
    # ChatLiteLLM writes api_base/api_key/organization onto the litellm module itself; put them back afterwards
    saved_globals = {k: getattr(litellm, k, None) for k in ("api_base", "api_key", "organization", "extra_headers")}
    cb = litellm_cb.install(log, caller="agent", decision_spec=AGENT_DECISION, logger_cls=AgentLLMLogger)
    handler = AgentStepLogger(log, llm_logger=cb)
    out = {"turns": [], "outcome": None, "followups": 0}
    errlog = open(log.run_dir / "mcp_server.stderr.log", "x", encoding="utf-8")
    t_start = time.perf_counter()
    try:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=300)
                listed = await session.list_tools()
                tools = await load_mcp_tools(session)
                timing["mcp_startup_s"] = round(time.perf_counter() - t_start, 3)
                schema = [t.model_dump(mode="json") for t in listed.tools]
                log.emit("note", kind="tools_loaded", n_tools=len(tools), tools=[t.name for t in tools],
                         tools_schema=log.blobs.put(schema), tools_schema_sha256=sha256_json(schema))
                strip_thinking = a.history_thinking == "strip"

                class ChatLiteLLMHistory(ChatLiteLLM):
                    """The history as sent: no LangChain block ids, tool-call ids renamed by position, past thinking dropped
                    (--history-thinking)."""
                    def _create_message_dicts(self, messages, stop):
                        dicts, params = super()._create_message_dicts(messages, stop)
                        dicts, n_ids = drop_block_ids(dicts)
                        dicts, _ = renumber_tool_call_ids(dicts)
                        n_think = 0
                        if strip_thinking:
                            dicts, n_think = strip_history_thinking(dicts)
                        if n_think:
                            log.emit("note", kind="history_thinking_stripped", blocks=n_think)
                        return dicts, params

                llm = ChatLiteLLMHistory(model=llm_name, api_base=v1, api_key="ollama", streaming=a.streaming,
                                  max_retries=1, timeout=300, temperature=dec["temperature"], model_kwargs=model_kwargs)
                agent = create_react_agent(llm, tools, prompt=prompt, checkpointer=InMemorySaver())
                config = {"configurable": {"thread_id": "1"}, "recursion_limit": a.recursion_limit,
                          "callbacks": [handler]}
                message, result = task_msg, None
                for turn in range(a.max_followups + 1):
                    handler.turn = turn
                    t0, info = time.perf_counter(), {"turn": turn, "user": message}
                    log.emit("note", kind="user_turn", turn=turn, text=message)
                    try:
                        result = await asyncio.wait_for(agent.ainvoke({"messages": message}, config),
                                                        timeout=a.turn_timeout)
                        info["end"] = "answered"
                    except GraphRecursionError as e:
                        info["end"], info["error"] = "recursion_limit", str(e)[:500]
                    except asyncio.TimeoutError:
                        info["end"] = "turn_timeout"
                    except Exception as e:  # noqa: BLE001 - an agent error ends the run, recorded
                        info["end"], info["error"] = "agent_error", f"{type(e).__name__}: {e}"[:2000]
                    info["wall_s"] = round(time.perf_counter() - t0, 3)
                    state = await agent.aget_state(config)
                    msgs = state.values.get("messages", []) if state and state.values else []
                    last = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
                    text, reasoning = message_text(None if last is None else last.content)
                    info.update(final_table=has_final_table(text), asked_question=bool(text) and "?" in text[-300:],
                                text_json_tool_call=text_json_tool_call(text) is not None and not (last and last.tool_calls),
                                non_ascii_share=non_ascii_share(text), n_messages=len(msgs), last_text=text,
                                last_reasoning_chars=None if reasoning is None else len(reasoning))
                    log.emit("note", kind="turn_end", **info)
                    out["turns"].append(info)
                    out["final_text"], out["messages"] = text, messages_to_dict(msgs)
                    if info["end"] in ("turn_timeout", "agent_error"):
                        out["outcome"] = info["end"]
                        break
                    if info["final_table"]:
                        out["outcome"] = "final_table"
                        break
                    if turn < a.max_followups:
                        out["followups"] += 1
                        message = FOLLOWUP
                else:
                    out["outcome"] = "recursion_limit" if out["turns"][-1]["end"] == "recursion_limit" else "no_final_table"
    finally:
        timing["agent_wall_s"] = round(time.perf_counter() - t_start, 3)
        errlog.close()
        if not litellm_cb.uninstall(cb, timeout=30):
            log.emit("error", kind="unlogged_agent_calls", message="some agent LLM calls were never reported")
        for k, v in saved_globals.items():
            setattr(litellm, k, v)
    out["steps"], out["client_tools"] = handler.steps, handler.tools
    return out


def server_tool_timing(run_dir: Path) -> list[dict]:
    return [{"server_seq": e.get("server_seq"), "tool": e.get("tool"), "method": e.get("method"),
             "tool_s": e.get("duration_s"), "llm_calls": e.get("llm_calls"), "llm_s": e.get("llm_s"),
             "error_type": e.get("error_type"), "result_rows": e.get("result_rows"),
             "cache": {k: (e.get("bdikit_cache") or {}).get(k) for k in ("hits", "misses", "saves")}}
            for e in read_events(run_dir) if e["type"] == "tool_call" and e.get("side") == "server"]


def summarize_timing(timing: dict, out: dict, server_tools: list[dict]) -> dict:
    llm = [s["llm_s"] or 0 for s in out.get("steps", [])]
    tools = [t["tool_s"] or 0 for t in server_tools]
    matcher = [t["llm_s"] or 0 for t in server_tools]
    return {**timing, "n_agent_steps": len(llm), "agent_llm_s": round(sum(llm), 3),
            "n_tool_calls": len(tools), "tool_s": round(sum(tools), 3),
            "tool_llm_calls": sum(t["llm_calls"] or 0 for t in server_tools), "tool_llm_s": round(sum(matcher), 3),
            "steps": out.get("steps", []), "server_tools": server_tools, "turns": [
                {k: t.get(k) for k in ("turn", "end", "wall_s", "final_table", "asked_question", "text_json_tool_call")}
                for t in out.get("turns", [])]}


def score_into_log(log, records, data: Data, study: str, task: str) -> list[dict]:
    from hbench.score_sm import score_sm
    from hbench.score_vm import score_vm
    out = []
    if any(r["kind"] == "sm" for r in records):
        _, s = score_sm([r for r in records if r["kind"] == "sm"], data, study, "gdc-sm")
        log.emit("score", gt=s["gt"], summary=s)
        out.append(s)
    if any(r["kind"] == "vm" for r in records):
        _, s = score_vm([r for r in records if r["kind"] == "vm"], data, study, mode=VM_MODE.get(task, "end_to_end"))
        log.emit("score", gt=s.get("gt"), summary=s)
        out.append(s)
    return out


def main(argv=None) -> Path:
    a = parse_args(argv)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    for k in [k for k in os.environ if any(m in k.upper() for m in SECRET_MARKERS)]:
        del os.environ[k]  # nothing in this process or its children can reach a cloud API with a real key
    if not a.allow_other_clients:
        others = other_ollama_clients()
        if others:
            raise SystemExit("other Ollama clients are running (the engine audit would fail):\n  " + "\n  ".join(others))
    data = Data(a.data_root) if a.data_root else Data()
    root = Path(a.raw_dir) if a.raw_dir else raw_root()
    src = stage_input(data, a.study, root)
    prompt, upstream_llm, chatbot_sha = chatbot_prompt()
    prompt = apply_prompt_variant(prompt, a.prompt_variant)
    task_msg = task_text(a.task, str(src), a.study)
    cond = build_condition(a)
    timing = {"t0_unix": time.time()}
    t0 = time.perf_counter()
    log = start_run(root, "e3", cond, task={"study": a.study, "kind": TASK_KIND[a.task], "mode": "end_to_end",
                                            "task": a.task, "prompt_id": TASK_VERSION, "followup": FOLLOWUP,
                                            "input_path": str(src)},
                    models=[] if a.no_lease else [a.model], data=data, lease=not a.no_lease,
                    audit=not a.no_lease, ollama_base=a.ollama_base,
                    llm={"agent_model": f"openai/{a.model}", "api_base": a.ollama_base.rstrip("/") + "/v1",
                         "temperature": a.temperature, "top_p": a.top_p, "seed": cond["decoding"]["seed"],
                         "tool_llm": {"model": f"openai/{a.model}", "temperature": cond["decoding"]["tool_llm_temperature"],
                                      "top_p": cond["decoding"]["tool_llm_top_p"]},
                         "upstream_default_llm": upstream_llm, "streaming": a.streaming,
                         "chat_model": "langchain_litellm.ChatLiteLLM(max_retries=1, timeout=300)"},
                    prompts={"system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                             "task_prompt_sha256": hashlib.sha256(task_msg.encode()).hexdigest(),
                             "followup_sha256": hashlib.sha256(FOLLOWUP.encode()).hexdigest(),
                             "chatbot_py_sha256": chatbot_sha})
    timing["start_run_s"] = round(time.perf_counter() - t0, 3)
    status, out = "ok", {}
    try:
        out = asyncio.run(drive(a, log, prompt, task_msg, timing))
        t = time.perf_counter()
        from hbench.adapters.bdikit_agent import predictions_from_run
        records, info = predictions_from_run(log.run_dir, a.study, log.run_id)
        write_predictions(log.run_dir / "predictions.jsonl", records)
        log.emit("prediction_batch", path="predictions.jsonl", n=len(records), sha256=hashlib.sha256(
            (log.run_dir / "predictions.jsonl").read_bytes()).hexdigest(), **info)
        scores = score_into_log(log, records, data, a.study, a.task)
        log.write_file("claims.md", out.get("final_text") or "")
        log.emit("note", kind="claims", path="claims.md", text=out.get("final_text"))
        log.write_file("transcript.json", json.dumps({"messages": out.get("messages"), "turns": out.get("turns")},
                                                     ensure_ascii=False, indent=1, default=str))
        timing["adapter_score_s"] = round(time.perf_counter() - t, 3)
        timing["total_wall_s"] = round(time.perf_counter() - t0, 3)
        summary = summarize_timing(timing, out, server_tool_timing(log.run_dir))
        log.write_file("timing.json", json.dumps(summary, indent=1, default=str))
        if out.get("outcome") in ("agent_error", "turn_timeout"):
            status = "error"
    except BaseException as e:
        status = "error"
        log.emit("error", kind="driver_failed", message=f"{type(e).__name__}: {e}")
        raise
    finally:
        steps = out.get("steps") or []
        log.close(status=status, outcome=out.get("outcome"), followups=out.get("followups"),
                  n_agent_steps=len(steps), text_json_tool_calls=sum(s["text_json_tool_call"] for s in steps),
                  wall_s=round(time.perf_counter() - t0, 3),
                  scores=[{k: s.get(k) for k in ("kind", "gt", "top1_correct", "n_annotated", "mrr")} for s in
                          (locals().get("scores") or [])])
    print(log.run_dir)
    return log.run_dir


if __name__ == "__main__":
    main()
