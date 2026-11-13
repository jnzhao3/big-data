"""E3 agent infrastructure without Ollama: the logged MCP server (hbench.runlog.mcp_logged), the LangChain step
logger, the agent adapter, and the headless driver end to end against a scripted OpenAI-compatible mock that
stands in for both the agent model and BDI-Kit's LLM matcher. The MCP server is BDI-Kit's real one."""

import asyncio
import json
import uuid
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from hbench.data import REPO_ROOT
from hbench.runlog.check import check_run
from hbench.runlog.mcp_logged import (ConfigError, LocalOnlyError, config_from_env, guard_kwargs,
                                      result_hashes)
from hbench.runlog.writer import BlobStore, RunLog, read_events

E3 = REPO_ROOT / "experiments" / "e3_agent"
sys.path.insert(0, str(E3))
import run_agent  # noqa: E402
from tasks import FOLLOWUP, task_text  # noqa: E402

MODEL = "qwen2.5-14b-bdikit"


# ------------------------------------------------------------------------------------------- scripted model mock

class ScriptedOpenAI(BaseHTTPRequestHandler):
    """POST /v1/chat/completions. Requests with tools (the agent) get the next scripted reply; requests without
    (BDI-Kit's value-matching LLM) get a fixed JSON answer. Every request body is kept."""
    script: list = []
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        ScriptedOpenAI.seen.append((self.path, body))
        if body.get("tools"):
            msg = ScriptedOpenAI.script.pop(0) if ScriptedOpenAI.script else {"content": "| a | b |\n|---|---|\n| x | y |"}
        else:
            msg = {"content": '{"term": "male", "score": 0.9}'}
        # random ids, as Ollama gives them: a later request echoes them back
        tcs = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(msg.get("tool_calls", []))]
        out = {"id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": body.get("model"),
               "choices": [{"index": 0, "finish_reason": "tool_calls" if tcs else "stop", "logprobs": None,
                            "message": {"role": "assistant", "content": msg.get("content", ""),
                                        **({"reasoning": msg["reasoning"]} if msg.get("reasoning") else {}),
                                        **({"tool_calls": tcs} if tcs else {})}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def scripted():
    ScriptedOpenAI.seen, ScriptedOpenAI.script = [], []
    srv = HTTPServer(("127.0.0.1", 0), ScriptedOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


# ------------------------------------------------------------------------------------------------ mcp_logged units

def _env(tmp_path, **kw):
    rd = tmp_path / "e3-x-clark-1"
    rd.mkdir(exist_ok=True)
    return {"HBENCH_RUN_DIR": str(rd), "HBENCH_RUN_ID": rd.name, **kw}


def test_config_requires_the_run_and_local_endpoints(tmp_path):
    with pytest.raises(ConfigError):
        config_from_env(["--llm", "openai/m"], env={})
    with pytest.raises(ConfigError):
        config_from_env(["--llm", "openai/m"], env=_env(tmp_path, OPENAI_BASE_URL="https://api.openai.com/v1"))
    with pytest.raises(ConfigError):
        config_from_env(["--llm", "openai/m"], env=_env(tmp_path, HBENCH_TOOL_LLM_API_BASE="https://x.example/v1"))
    with pytest.raises(SystemExit):  # no --llm: the matchers would fall back to openai/gpt-4o-mini
        config_from_env([], env=_env(tmp_path))
    cfg = config_from_env(["--llm", "openai/m"], env=_env(tmp_path, HBENCH_TOOL_LLM_TEMPERATURE="0",
                                                         HBENCH_TOOL_LLM_SEED="7", HBENCH_TOOL_LLM_TOP_P="default"))
    assert cfg["params"] == {"temperature": 0.0, "top_p": None, "seed": 7, "top_logprobs": None}
    assert cfg["allowed_models"] == ["openai/m"] and cfg["cache_mode"] == "run"


def test_guard_refuses_cloud_and_disallowed_models_and_injects_params(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    cfg = config_from_env(["--llm", f"openai/{MODEL}"], env=_env(
        tmp_path, HBENCH_TOOL_LLM_TEMPERATURE="0", HBENCH_TOOL_LLM_TOP_P="1", HBENCH_TOOL_LLM_SEED="3",
        HBENCH_TOOL_LLM_TOP_LOGPROBS="5"))
    for model, kw in [("openai/gpt-4o-mini", {}),                       # BDI-Kit's matcher default
                      ("deepinfra/openai/gpt-oss-120b", {}),            # chatbot.py's default
                      (f"openai/{MODEL}", {"api_base": "https://portkey.example/v1"})]:  # Portkey route
        with pytest.raises(LocalOnlyError):
            guard_kwargs(model, kw, cfg)
    sent, overridden = guard_kwargs(f"openai/{MODEL}", {"messages": [], "temperature": 0.7}, cfg)
    assert sent["api_base"] == "http://127.0.0.1:11434/v1" and sent["api_key"] == "ollama"
    assert (sent["temperature"], sent["top_p"], sent["seed"], sent["top_logprobs"], sent["logprobs"]) == (0, 1, 3, 5, True)
    assert overridden == {"temperature": 0.7}


def test_models_dir_is_pinned_and_never_names_a_magneto_base_model(tmp_path):
    import types
    from hbench.runlog.mcp_logged import pin_models_dir
    dl = types.SimpleNamespace(BDIKIT_MODELS_CACHE_DIR="<run>/bdikit_cache/models")
    (tmp_path / "ok").mkdir()
    assert pin_models_dir(dl, tmp_path / "ok") == dl.BDIKIT_MODELS_CACHE_DIR == str((tmp_path / "ok" / "models").resolve())
    with pytest.raises(ConfigError):
        pin_models_dir(dl, tmp_path / "x-a1e5")


def test_result_set_hash_ignores_row_order():
    a, b = [{"x": 1}, {"x": 2}], [{"x": 2}, {"x": 1}]
    assert result_hashes(a)[1] == result_hashes(b)[1] and result_hashes(a)[0] != result_hashes(b)[0]


def test_check_run_flags_refused_cloud_calls(tmp_path):
    log = RunLog.create(tmp_path, "e3", "c", "clark")
    log.emit("run_header")
    log.emit("error", kind="cloud_call_attempted", model="openai/gpt-4o-mini")
    log.close()
    assert any("non-local or disallowed" in p for p in check_run(log.run_dir)["problems"])


# ------------------------------------------------------------------------- mcp_logged as a real stdio MCP server

def _server_env(rd: Path, base: str, shared: Path, **kw) -> dict:
    env = run_agent.clean_env(os.environ, HBENCH_RUN_DIR=rd, HBENCH_RUN_ID=rd.name, OPENAI_BASE_URL=f"{base}/v1",
                              HBENCH_TOOL_LLM_API_BASE=f"{base}/v1", HBENCH_TOOL_LLM_TEMPERATURE=0,
                              HBENCH_TOOL_LLM_TOP_P=1, HBENCH_TOOL_LLM_SEED=11, HBENCH_BDIKIT_SHARED=shared,
                              HF_HUB_OFFLINE=1, PYTHONPATH=REPO_ROOT, PYTHONHASHSEED=0, **kw)
    env.pop("BDIKIT_CACHE", None)
    return env


def _call_tools(env: dict, calls: list) -> list:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        p = StdioServerParameters(command=sys.executable, env=env, cwd=str(REPO_ROOT),
                                  args=["-s", "-m", "hbench.runlog.mcp_logged", "--llm", f"openai/{MODEL}"])
        out = []
        with open(Path(env["HBENCH_RUN_DIR"]) / "mcp_server.stderr.log", "w") as err:
            async with stdio_client(p, errlog=err) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    out.append(len((await s.list_tools()).tools))
                    for name, args in calls:
                        out.append(await s.call_tool(name, args))
        return out
    return asyncio.run(go())


def test_logged_server_logs_tools_llm_calls_cache_and_state(tmp_path, scripted, data):
    src = str(data.source_path("clark"))
    rd = tmp_path / "e3-mcp-clark-1"
    rd.mkdir()
    pair = [{"source_attribute": "Gender", "target_attribute": "gender"}]
    vm = {"source_dataset_path": src, "target_dataset_path": "gdc", "attribute_matches": pair, "method": "llm"}
    res = _call_tools(_server_env(rd, scripted, tmp_path / "shared"), [
        ("get_available_value_matching_algorithms", {}),
        ("match_values", vm), ("match_values", vm),         # the second is a cache hit inside this run
        ("update_schema_matching", {"source_attribute": "a", "new_target_attribute": "b", "new_similarity": 1})])
    assert res[0] == 10 and not res[1].isError and not res[2].isError and res[4].isError
    ev = read_events(rd)
    procs = {e["proc"] for e in ev}
    assert len(procs) == 1 and next(iter(procs)) == "mcp"
    server = [e for e in ev if e["type"] == "tool_call"]
    assert [e["tool"] for e in server] == ["get_available_value_matching_algorithms", "match_values", "match_values",
                                          "update_schema_matching"]
    mv1, mv2 = server[1], server[2]
    assert mv1["method"] == "llm" and mv1["llm_calls"] == 2 and mv1["bdikit_cache"]["misses"] == 1
    assert mv2["llm_calls"] == 0 and mv2["bdikit_cache"]["hits"] == 1 and mv1["result_set_sha256"] == mv2["result_set_sha256"]
    assert server[3]["error_type"] == "ValueError" and server[3]["traceback_sha256"]
    # matcher LLM calls: logged under the tool, with the injected parameters actually sent
    llm = [e for e in ev if e["type"] == "llm_call"]
    assert len(llm) == 2 and all(e["caller"] == "tool:match_values/llm" for e in llm)
    assert all(e["parent_id"] == mv1["start_span"] and not e["cloud_call"] for e in llm)
    sent = [b for p, b in ScriptedOpenAI.seen if p == "/v1/chat/completions"]
    assert len(sent) == 2 and all((b["temperature"], b["top_p"], b["seed"]) == (0, 1, 11) for b in sent)
    # state after every call; the last one holds the value matches
    snaps = [e for e in ev if e["type"] == "state_snapshot"]
    assert len(snaps) == 4 and snaps[-1]["value_matching_results_rows"] == 4  # appended twice: 2 values x 2 calls
    assert all(t["state_after"] == s["span_id"] for t, s in zip(server, snaps))
    # per-run cache: created empty, models/embeddings shared
    assert (rd / "bdikit_cache" / "models").is_symlink() and len(list((rd / "bdikit_cache" / "runs").iterdir())) == 1
    hdr = next(e for e in ev if e["type"] == "run_header")
    assert hdr["bdikit_cache"]["runs_files_at_start"] == 0 and hdr["tool_llm"]["temperature"] == 0
    end = ev[-1]
    assert end["type"] == "run_end" and end["tool_llm_calls"] == 2 and end["cloud_calls_refused"] == 0
    assert check_run(rd)["problems"] == []


def test_logged_server_refuses_a_disallowed_model_loudly(tmp_path, scripted, data):
    rd = tmp_path / "e3-mcp-clark-2"
    rd.mkdir()
    vm = {"source_dataset_path": str(data.source_path("clark")), "target_dataset_path": "gdc",
          "attribute_matches": [{"source_attribute": "Gender", "target_attribute": "gender"}], "method": "llm"}
    res = _call_tools(_server_env(rd, scripted, tmp_path / "shared", HBENCH_TOOL_LLM_MODELS="openai/other"),
                      [("match_values", vm)])
    assert res[1].isError and "refused" in res[1].content[0].text
    assert not [b for p, b in ScriptedOpenAI.seen]                      # nothing reached the model server
    ev = read_events(rd)
    assert [e["kind"] for e in ev if e["type"] == "error"] == ["cloud_call_attempted"]
    assert any("non-local or disallowed" in p for p in check_run(rd)["problems"])
    assert "REFUSED" in (rd / "mcp_server.stderr.log").read_text()


# ------------------------------------------------------------------------------------------------ driver pieces

def test_driver_uses_the_installed_chatbot_prompt():
    prompt, default_llm, sha = run_agent.chatbot_prompt()
    assert "tool-augmented assistant" in prompt and "SCHEMA MATCHING WORKFLOW" in prompt
    assert default_llm == "deepinfra/openai/gpt-oss-120b" and len(sha) == 64


def test_prompt_variant_drops_only_the_json_format_line():
    prompt = run_agent.chatbot_prompt()[0]
    assert run_agent.apply_prompt_variant(prompt, "verbatim") == prompt
    patched = run_agent.apply_prompt_variant(prompt, "no_json_format")
    assert "output a JSON object" not in patched and "- Otherwise respond with markdown tables" in patched
    assert len(prompt) - len(patched) == len(run_agent.JSON_FORMAT_LINE)


def test_turn_classifiers():
    from hbench.runlog.langchain_cb import non_ascii_share, text_json_tool_call
    assert run_agent.has_final_table("Done.\n\n| Source | Target |\n|---|---|\n| Age | age_at_diagnosis |")
    assert not run_agent.has_final_table("Shall I continue with value matching?")
    assert text_json_tool_call('{"tool": "match_schema", "parameters": {"source_dataset_path": "x"}}')
    assert text_json_tool_call('Calling:\n```json\n{"name": "match_schema", "arguments": {}}\n```')
    assert text_json_tool_call('<tool_call>\n{"name": "match_schema", "arguments": {"source_dataset_path": "x"}}\n</tool_call>')
    assert text_json_tool_call("| a | b |") is None
    assert non_ascii_share("abc") == 0.0 and non_ascii_share("ab\u00e9\u00e8") == 0.5


def test_task_texts_have_no_spaces_in_paths_and_are_stable():
    t = task_text("vm_pairs", "/x/clark.csv", "clark")
    assert '"source_attribute": "Tumor/Normal"' in t and "Do not run schema matching" in t
    assert task_text("sm", "/x/clark.csv", "clark") == task_text("sm", "/x/clark.csv", "clark")


def _snapshot_run(tmp_path, sm_rows, vm_rows, tools=("match_schema", "match_values")):
    log = RunLog.create(tmp_path, "e3", "c", "clark", proc="mcp.1")
    log.emit("run_header")
    for t in tools:
        log.emit("tool_call", side="server", tool=t)
    log.emit("state_snapshot",
             schema_matching_results=None if sm_rows is None else log.blobs.put(sm_rows),
             value_matching_results=None if vm_rows is None else log.blobs.put(vm_rows))
    log.close()
    return log.run_dir


def test_adapter_takes_the_last_snapshot_and_keeps_the_last_vm_row(tmp_path):
    from hbench.adapters.bdikit_agent import predictions_from_run
    sm = [{"source_attribute": "Age", "target_attribute": "age_at_diagnosis", "similarity": 0.9},
          {"source_attribute": "Case_ID", "target_attribute": None, "similarity": None}]
    vm = [{"source_attribute": "Gender", "target_attribute": "gender", "source_value": "Male",
           "target_value": "female", "similarity": 0.2},
          {"source_attribute": "Gender", "target_attribute": "gender", "source_value": "Male",
           "target_value": "male", "similarity": 1.0}]
    recs, info = predictions_from_run(_snapshot_run(tmp_path, sm, vm), "clark")
    smr = {r["source_column"]: r["targets"] for r in recs if r["kind"] == "sm"}
    vmr = [r for r in recs if r["kind"] == "vm"]
    assert smr == {"Age": ["age_at_diagnosis"], "Case_ID": []}
    assert len(vmr) == 1 and vmr[0]["targets"] == ["male"] and info["vm_duplicate_rows_dropped"] == 1
    recs, info = predictions_from_run(_snapshot_run(tmp_path, sm, None, tools=("match_schema",)), "clark")
    assert info["vm"] == "not_attempted" and info["sm"] == "ok" and all(r["kind"] == "sm" for r in recs)


# ------------------------------------------------------------------------------------------ driver end to end

def test_driver_end_to_end_with_scripted_agent(tmp_path, scripted, data):
    """vm_pairs on clark: a tool call, a question (no table: the fixed follow-up is sent), a tool call written as
    JSON text (a failure type; another follow-up), then the final table. Real MCP server, tfidf matcher."""
    src = str(tmp_path / "raw" / "_inputs" / "e3" / "clark.csv")
    pairs = [{"source_attribute": "Gender", "target_attribute": "gender"},
             {"source_attribute": "Tumor_Focality", "target_attribute": "tumor_focality"}]
    ScriptedOpenAI.script = [
        {"content": "", "tool_calls": [("match_values", {"source_dataset_path": src, "target_dataset_path": "gdc",
                                                         "attribute_matches": pairs, "method": "tfidf"})]},
        {"content": "I matched the values. Should I also update anything?"},
        {"content": '{"tool": "preview_domain", "parameters": {"dataset_path": "gdc", "attribute": "gender"}}'},
        {"content": "", "tool_calls": [("update_value_matching", {
            "source_attribute": "Gender", "target_attribute": "gender", "source_value": "Male",
            "new_target_value": "male", "new_similarity": 1.0})]},
        {"content": "Final value matches:\n\n| Source | Value | Target |\n|---|---|---|\n| Gender | Male | male |"},
    ]
    run_dir = run_agent.main(["--study", "clark", "--task", "vm_pairs", "--model", MODEL, "--temperature", "0",
                              "--top-p", "1", "--seed", "5", "--engine-protocol", "natural", "--ollama-base", scripted,
                              "--raw-dir", str(tmp_path / "raw"), "--bdikit-shared", str(tmp_path / "shared"),
                              "--max-followups", "2", "--no-lease", "--allow-other-clients"])
    ev = read_events(run_dir)
    turns = [e for e in ev if e["type"] == "note" and e.get("kind") == "turn_end"]
    assert [t["end"] for t in turns] == ["answered"] * 3
    assert [t["final_table"] for t in turns] == [False, False, True] and turns[0]["asked_question"]
    assert turns[1]["text_json_tool_call"]
    users = [e["text"] for e in ev if e["type"] == "note" and e.get("kind") == "user_turn"]
    assert users[1:] == [FOLLOWUP, FOLLOWUP]
    end = next(e for e in ev if e["type"] == "run_end" and e["proc"] == "driver")
    assert end["outcome"] == "final_table" and end["followups"] == 2 and end["n_agent_steps"] == 5
    assert end["text_json_tool_calls"] == 1
    # agent calls: logged by LiteLLM with step numbers and the tool-call decision spec; requests state T and top_p
    agent = [e for e in ev if e["type"] == "llm_call" and e["caller"] == "agent"]
    assert [e["step"] for e in agent] == [1, 2, 3, 4, 5]
    assert all(e["decision_spec"] == run_agent.AGENT_DECISION and not e["cloud_call"] for e in agent)
    bodies = [b for p, b in ScriptedOpenAI.seen if b.get("tools")]
    assert all((b["temperature"], b["top_p"], b["seed"], b["logprobs"]) == (0, 1, 5, True) for b in bodies)
    assert "tool-augmented assistant" in bodies[0]["messages"][0]["content"]
    # steps and both sides of each tool call
    steps = [e for e in ev if e["type"] == "agent_step"]
    assert [s["step"] for s in steps] == [1, 2, 3, 4, 5] and all(s["duration_s"] is not None for s in steps)
    client = [e for e in ev if e["type"] == "tool_call" and e["side"] == "client"]
    server = [e for e in ev if e["type"] == "tool_call" and e["side"] == "server"]
    assert [c["tool"] for c in client] == [s["tool"] for s in server] == ["match_values", "update_value_matching"]
    assert client[0]["parent_id"] == steps[0]["span_id"] and client[0]["args_sha256"] == server[0]["args_sha256"]
    assert client[0]["result_set_sha256"] == server[0]["result_set_sha256"]
    # predictions from the last snapshot, scored; the run validates
    preds = [json.loads(x) for x in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert preds and all(p["kind"] == "vm" for p in preds)
    assert next(p for p in preds if p["source_value"] == "Male")["targets"] == ["male"]
    assert [e for e in ev if e["type"] == "score"]
    timing = json.loads((run_dir / "timing.json").read_text())
    assert timing["n_agent_steps"] == 5 and timing["n_tool_calls"] == 2 and "mcp_startup_s" in timing
    assert (run_dir / "claims.md").read_text().startswith("Final value matches")
    assert check_run(run_dir)["problems"] == []
    out = subprocess.run([sys.executable, "-m", "hbench", "score", str(run_dir / "predictions.jsonl"), "--gt",
                          "gdc-sm", "--vm-mode", "oracle"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (run_dir / "items.gdc-sm.csv").exists()


def test_driver_stops_after_max_followups_without_a_table(tmp_path, scripted, data):
    ScriptedOpenAI.script = [{"content": "Which file do you mean?"}] * 3
    run_dir = run_agent.main(["--study", "clark", "--task", "sm", "--model", MODEL, "--temperature", "0",
                              "--top-p", "1", "--engine-protocol", "natural", "--ollama-base", scripted,
                              "--raw-dir", str(tmp_path / "raw"), "--bdikit-shared", str(tmp_path / "shared"),
                              "--max-followups", "1", "--no-lease", "--allow-other-clients"])
    ev = read_events(run_dir)
    end = next(e for e in ev if e["type"] == "run_end" and e["proc"] == "driver")
    assert end["outcome"] == "no_final_table" and end["followups"] == 1 and end["n_agent_steps"] == 2
    batch = next(e for e in ev if e["type"] == "prediction_batch")
    assert batch["sm"] == "not_attempted" and batch["n"] == 0
    assert check_run(run_dir)["problems"] == []


def test_modelfile_keeps_ollamas_tool_call_prefix_right_after_the_first_toolcalls_if():
    """Ollama 0.40.0 infers the tool-call format from the text after the template's first {{ if .ToolCalls }};
    anything else there makes it return tool calls as plain text (found in the first live E3 run)."""
    import re
    text = (E3 / "Modelfile.qwen2.5-14b-bdikit").read_text()
    tmpl = text.split('TEMPLATE """', 1)[1]
    first = re.search(r"\{\{-?\s*(?:else )?if \.ToolCalls\s*-?\}\}", tmpl)
    assert first and tmpl[first.end():].startswith("<tool_call>")
    assert "{{ end }}{{- if .ToolCalls }}<tool_call>" in tmpl          # text and tool calls both rendered
    assert re.findall(r"^PARAMETER .*$", text, re.M) == ["PARAMETER num_ctx 32768"]  # no sampling parameters


def test_modelfile_qwen38_only_pins_context():
    import re
    text = (E3 / "Modelfile.qwen3.8-27b-bdikit").read_text()
    assert re.findall(r"^FROM .*$", text, re.M) == ["FROM qwen3.8:27b-q4_K_M"]
    assert "TEMPLATE" not in re.sub(r"^#.*$", "", text, flags=re.M)        # built-in renderer and parser
    assert re.findall(r"^PARAMETER .*$", text, re.M) == ["PARAMETER num_ctx 32768"]


def test_strip_history_thinking():
    strip_history_thinking = run_agent.strip_history_thinking
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Ok."}],
             "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": [{"type": "text", "text": "row"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}, {"type": "image_url", "image_url": {}}]},
            {"role": "assistant", "content": "plain"}]
    out, n = strip_history_thinking(msgs)
    assert n == 2
    assert out[1] == {"role": "assistant", "content": "Ok.", "tool_calls": [{"id": "c1"}]}
    assert out[2] == msgs[2]                                   # tool results untouched
    assert out[3]["content"] == [{"type": "image_url", "image_url": {}}]
    assert out[4] == msgs[4] and out[0] == msgs[0]
    assert msgs[1]["content"][0]["type"] == "thinking"         # input not mutated
    assert strip_history_thinking([{"role": "assistant", "content": [{"type": "thinking", "thinking": "t"}]}])[0][0]["content"] == ""


def test_strip_history_thinking_on_chatlitellm_dicts(monkeypatch):
    """The message dicts langchain_litellm builds from a thinking reply come out as Ollama /v1 accepts them."""
    import litellm
    for k in ("api_base", "api_key", "organization", "extra_headers"):  # ChatLiteLLM writes these module globals
        monkeypatch.setattr(litellm, k, getattr(litellm, k, None), raising=False)
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_litellm import ChatLiteLLM
    strip_history_thinking = run_agent.strip_history_thinking
    llm = ChatLiteLLM(model="openai/x", api_base="http://127.0.0.1:1/v1", api_key="ollama")
    ai = AIMessage(content=[{"type": "thinking", "thinking": "plan"}, {"type": "text", "text": "Calling."}],
                   tool_calls=[{"name": "match_schema", "args": {"a": 1}, "id": "c1"}])
    dicts, _ = llm._create_message_dicts([HumanMessage("go"), ai], None)
    assert any(b.get("type") == "thinking" for b in dicts[1]["content"])  # what Ollama rejects
    out, n = strip_history_thinking(dicts)
    assert n == 1 and out[1]["content"] == "Calling." and out[1]["tool_calls"]


def test_driver_sampled_run_sends_per_call_seeds_reproducibly(tmp_path, scripted, data):
    """At T>0 every call (the agent's and the tool LLM's in the MCP server) is sent its own per-call seed, the log
    records the seed actually sent, and a second run with the same base seed sends exactly the same seeds."""
    src = str(tmp_path / "raw" / "_inputs" / "e3" / "clark.csv")
    pair = [{"source_attribute": "Gender", "target_attribute": "gender"}]

    def one_run():
        ScriptedOpenAI.seen.clear()
        ScriptedOpenAI.script = [
            {"content": "", "tool_calls": [("match_values", {"source_dataset_path": src, "target_dataset_path": "gdc",
                                                             "attribute_matches": pair, "method": "llm"})]},
            {"content": "Final value matches:\n\n| Source | Value | Target |\n|---|---|---|\n| Gender | Male | male |"}]
        rd = run_agent.main(["--study", "clark", "--task", "vm_pairs", "--model", MODEL, "--temperature", "0.7",
                             "--top-p", "0.9", "--seed", "5", "--engine-protocol", "natural", "--ollama-base", scripted,
                             "--raw-dir", str(tmp_path / "raw"), "--bdikit-shared", str(tmp_path / "shared"),
                             "--no-lease", "--allow-other-clients"])
        sent = [(bool(b.get("tools")), b["seed"]) for p, b in ScriptedOpenAI.seen if p == "/v1/chat/completions"]
        return rd, sent

    rd1, sent1 = one_run()
    ev = read_events(rd1)
    calls = [e for e in ev if e["type"] == "llm_call"]
    agent = [e for e in calls if e["caller"] == "agent"]
    tool = [e for e in calls if e["caller"] == "tool:match_values/llm"]
    assert len(agent) == 2 and len(tool) == 2 and {e["proc"] for e in tool} == {"mcp"}
    assert all(e["seed_base"] == 5 and e["seed_rule"] for e in calls)
    seeds = [s for _, s in sent1]
    assert 5 not in seeds and len(set(seeds)) == len(seeds) == 4           # derived, one per call
    assert sorted(e["decoding"]["seed"] for e in calls) == sorted(seeds)    # the log records the seed sent
    assert check_run(rd1)["problems"] == []
    rd2, sent2 = one_run()
    assert rd2 != rd1 and sent2 == sent1                                    # reproducible from the base seed


def test_drop_block_ids_keeps_everything_else():
    msgs = [{"role": "tool", "tool_call_id": "c1",
             "content": [{"type": "text", "text": "row", "id": "lc_123"}, {"type": "text", "text": "r2"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi", "id": "keep"}]}]
    out, n = run_agent.drop_block_ids(msgs)
    assert n == 1 and out[0]["content"] == [{"type": "text", "text": "row"}, {"type": "text", "text": "r2"}]
    assert out[0]["tool_call_id"] == "c1" and out[1] == msgs[1] and msgs[0]["content"][0]["id"] == "lc_123"


def test_renumber_tool_call_ids_by_position():
    hist = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_ab12", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call_cd34", "type": "function", "function": {"name": "b", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_cd34", "content": "rb"},
            {"role": "tool", "tool_call_id": "call_ab12", "content": "ra"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_ef56", "type": "function",
                                                                 "function": {"name": "c", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_ef56", "content": "rc"},
            {"role": "tool", "tool_call_id": "orphan", "content": "x"}]
    out, n = run_agent.renumber_tool_call_ids(hist)
    assert n == 3
    assert [tc["id"] for tc in out[1]["tool_calls"]] == ["call_0_0", "call_0_1"]
    assert [m["tool_call_id"] for m in out[2:4]] == ["call_0_1", "call_0_0"]
    assert out[4]["tool_calls"][0]["id"] == "call_1_0" and out[5]["tool_call_id"] == "call_1_0"
    assert out[6] == hist[6] and out[0] == hist[0]
    assert hist[1]["tool_calls"][0]["id"] == "call_ab12"            # input not mutated
    other = [dict(m) for m in hist]                                  # same history with other random ids
    other[1] = {**hist[1], "tool_calls": [{**hist[1]["tool_calls"][0], "id": "call_zz"},
                                          {**hist[1]["tool_calls"][1], "id": "call_yy"}]}
    other[2] = {**hist[2], "tool_call_id": "call_yy"}
    other[3] = {**hist[3], "tool_call_id": "call_zz"}
    assert run_agent.renumber_tool_call_ids(other)[0][:4] == out[:4]


def test_message_text_and_final_table_with_thinking_blocks():
    """A thinking model's reply is a list [thinking, text]; the visible text must be judged, not its JSON dump
    (json.dumps escapes the newlines, so the table separator was never found)."""
    table = "Final schema matches:\n\n| Source | Target |\n|---|---|\n| Gender | gender |"
    content = [{"type": "thinking", "thinking": "done; present the tables? yes"}, {"type": "text", "text": table}]
    assert not run_agent.has_final_table(json.dumps(content))      # the old path
    text, reasoning = run_agent.message_text(content)
    assert text == table and reasoning == "done; present the tables? yes"
    assert run_agent.has_final_table(text)
    assert run_agent.message_text(table) == (table, None)
    assert run_agent.message_text(None) == (None, None)
    two = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]
    assert run_agent.message_text(two) == ("a\nb", None)


def test_driver_sees_final_table_of_a_thinking_reply(tmp_path, scripted, data):
    """With reasoning in the reply (as qwen3.8 on /v1 sends it) the final table is found: no follow-ups."""
    ScriptedOpenAI.seen.clear()
    ScriptedOpenAI.script = [{"reasoning": "All done; present the tables.",
                              "content": "Final schema matches:\n\n| Source | Target |\n|---|---|\n| Gender | gender |"}]
    rd = run_agent.main(["--study", "clark", "--task", "smoke", "--model", MODEL, "--temperature", "0", "--top-p", "1",
                         "--seed", "1", "--engine-protocol", "natural", "--ollama-base", scripted,
                         "--raw-dir", str(tmp_path / "raw"), "--bdikit-shared", str(tmp_path / "shared"),
                         "--no-lease", "--allow-other-clients"])
    ev = read_events(rd)
    turns = [e for e in ev if e["type"] == "note" and e.get("kind") == "turn_end"]
    end = next(e for e in ev if e["type"] == "run_end" and e["proc"] == "driver")
    assert len(turns) == 1 and turns[0]["final_table"] and turns[0]["last_reasoning_chars"] > 0
    assert end["outcome"] == "final_table" and end["followups"] == 0
    assert (Path(rd) / "claims.md").read_text().startswith("Final schema matches:")


def test_modelfile_nb1024_only_pins_batch():
    import re
    text = (E3 / "Modelfile.qwen3.8-27b-bdikit-nb1024").read_text()
    assert re.findall(r"^FROM .*$", text, re.M) == ["FROM qwen3.8-27b-bdikit"]
    assert re.findall(r"^(?!#)\S.*$", text, re.M) == ["FROM qwen3.8-27b-bdikit", "PARAMETER num_batch 1024"]
