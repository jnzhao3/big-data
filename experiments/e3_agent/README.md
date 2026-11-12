# E3: the BDI-Kit chat agent, headless, end to end

BDI-Kit 0.11.0's chat agent (`bdikit/chatbot.py`: a LangGraph ReAct agent over BDI-Kit's stdio MCP server, 10 tools) runs here without its Streamlit UI on a local Qwen model. Every agent step, tool call and tool-internal LLM call is logged through hbench, and the final mapping is scored by hbench.

## Run

From the repo root, in `~/venvs/harmonize`, with Ollama free (no other experiment running):

```bash
ollama create qwen2.5-14b-bdikit -f experiments/e3_agent/Modelfile.qwen2.5-14b-bdikit   # once
python -s experiments/e3_agent/run_agent.py --study clark --task smoke --temperature 0 --top-p 1 --seed 1
python -s experiments/e3_agent/run_agent.py --study clark --task vm_pairs --temperature 0 --top-p 1 --seed 1
python -s experiments/e3_agent/run_agent.py --study clark --task sm --temperature 0 --top-p 1 --seed 1
python -m hbench log validate <run_dir>
python -m hbench score <run_dir>/predictions.jsonl --gt gdc-sm --vm-mode oracle
```

Tasks (`tasks.py`, version `task_v1`): `smoke` (list the schema matchers), `sm` (schema matching, test plan Task A), `vm_pairs` (value matching for 5 given categorical pairs, Task B1), `sm_vm` (both). `sm` and `sm_vm` need the Magneto weights `magneto-gdc-v0.1` in the shared BDI-Kit cache (`~/.cache/bdikit/models`, about 438 MB), because the agent's default schema matcher is `magneto_ft_llm`.

## What the driver does

- Rebuilds the agent as `chatbot.py` builds it: the system prompt is read from the installed `chatbot.py` with `ast` (its sha256 is in the run header), with `ChatLiteLLM(max_retries=1, timeout=300)`, `create_react_agent(..., checkpointer=InMemorySaver())` and thread `"1"`. `chatbot.py` itself cannot be imported: it relaunches itself under Streamlit.
- Starts the MCP server as `python -s -m hbench.runlog.mcp_logged --llm openai/<model>`. This is BDI-Kit's own server and tools, plus:
  - one `tool_call` (with `result_sha256` and the order-insensitive `result_set_sha256`) and one `state_snapshot` per call;
  - the tool-internal LLM calls, logged under their tool;
  - the run's sampling parameters injected into those calls;
  - a refusal, logged, of any call to a model other than the run's or to a non-local endpoint (BDI-Kit's matchers default to `openai/gpt-4o-mini`);
  - its own empty BDI-Kit result cache for every run;
  - stdout reserved for JSON-RPC.
- Plays scripted turns: the task, then the fixed follow-up (`tasks.FOLLOWUP`) whenever a turn ends without a final markdown table, at most `--max-followups` times (default 2). Nothing waits for a human. Each turn's end is recorded as `answered`, `recursion_limit`, `turn_timeout` or `agent_error`, with these flags: asked a question, tool call written as JSON text, and non-ASCII share.
- Logs both sides of each step. Agent LLM calls are `llm_call` events, caller `agent`, numbered by step, with the decision spec `tool_call/name`. LangChain callbacks add `agent_step` and the client side of each `tool_call`.
- Writes `predictions.jsonl` from the last state snapshot (`hbench.adapters.bdikit_agent`; VM is `not_attempted` if `match_values` never succeeded), scores it into the log, saves the final text as `claims.md` (for E7), and writes `transcript.json` and `timing.json`.

## Deviations from the Streamlit app (all in the run condition)

| App | Here | Why |
|---|---|---|
| `bdikit-mcp` | `hbench.runlog.mcp_logged` (same server and tools, wrapped) | logging, cache isolation, local-only guard |
| model and sampling from LiteLLM defaults | `openai/` route on Ollama `/v1`, temperature and top_p stated, logprobs requested | `ollama_chat/` drops tool calls from the history; the server default differs by endpoint |
| `qwen2.5:14b` template | derived `qwen2.5-14b-bdikit` (`Modelfile.qwen2.5-14b-bdikit`) | the stock template drops tool calls from assistant messages that also have text |
| `qwen3.8:27b` (`--model qwen3.8-27b-bdikit`) | derived `qwen3.8-27b-bdikit` (`Modelfile.qwen3.8-27b-bdikit`): `num_ctx 32768` only | the context size would otherwise follow Ollama's memory-based default; the built-in renderer and parser need no fix |
| past thinking sent back as `{"type": "thinking"}` content blocks | dropped from the history (`--history-thinking strip`, the default; no-op for models that do not think) | Ollama `/v1` rejects those blocks, so a thinking model dies at its second step; Qwen's own template drops past thinking too |
| tool results sent with LangChain's random block ids (`lc_<uuid>`) | ids dropped from the request | they are not part of the message (Ollama renders only the text) but would change every run's prompt hash, and so a sampled call's per-call seed |
| `streaming=True` | `streaming=False` (`--streaming` restores it) | the logged response is the one the agent used, with logprobs |
| shared result cache | one per run | the cache would hide matcher variance |

Repeated runs (E4) keep `--engine-protocol cold_start` (the default): back-to-back runs change how calls are computed (E1).
