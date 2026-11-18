"""End to end on the real clark study, both modes, without a live Claude session:
session start -> MCP server over stdio (real client) -> every tool -> submissions -> finish (convert, score,
session.json) -> viewer endpoints. Skipped when the benchmark data is not downloaded."""

import asyncio
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from cvm import HARNESS_DIR
from hbench.data import Data
from hbench.predictions import read_predictions, validate

pytestmark = pytest.mark.skipif(not Data().exists(), reason="benchmark data not downloaded")
PY = sys.executable


def _start(env, *args) -> dict:
    out = subprocess.run([PY, "-I", str(HARNESS_DIR / "session.py"), "start", *args], env=env, capture_output=True,
                         text=True, check=True).stdout
    # values can span lines (START_PROMPT holds the task text), so tokenize the whole output as bash would
    return dict(tok.split("=", 1) for tok in shlex.split(out) if tok != "export")


async def _drive(mcp_cfg: dict, mode: str) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    srv = mcp_cfg["mcpServers"]["matcher"]
    params = StdioServerParameters(command=srv["command"], args=srv["args"], env={**srv["env"],
                                                                                  "PATH": os.environ["PATH"]})
    calls = 0
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        names = {t.name for t in (await s.list_tools()).tools}

        async def call(tool_name, **args):
            nonlocal calls
            calls += 1
            res = await s.call_tool(tool_name, args)
            assert not res.isError, (tool_name, res.content)
            return json.loads(res.content[0].text)

        d = await call("describe_study")
        if mode == "oracle":
            sa, ta = d["pairs"][0]["source_attribute"], d["pairs"][0]["target_attribute"]
            sa, ta = next(((p["source_attribute"], p["target_attribute"]) for p in d["pairs"]
                           if p["target_attribute"] == "race"), (sa, ta))
            q = await call("query", sql='SELECT count(*) AS n FROM source')
            assert q["rows"] == [[110]]
            await call("save_code", name="norm", code="def run(i):\n    return sorted(set(v.lower() for v in "
                                                      "values(i)))", description="lowercase distinct")
            r = await call("run_code", name="norm", inputs=sa, columns=[sa])
            assert r["version"] == 1 and r["total"] >= 1
            assert (await call("list_code"))["functions"][0]["name"] == "norm"
        else:
            sa, ta = "Race", "race"
            hits = await call("search_attributes", query="race")
            assert hits["results"][0]["attribute"] == "race"
            await call("map_attribute", source_attribute=sa, target_attribute=ta)
        await call("profile_column", column=sa)
        await call("sample_rows", n=5, seed=1, stratify_by=sa)
        vals = (await call("list_values", column=sa))["values"]
        await call("describe_attribute", attribute=ta)
        if mode == "oracle":
            await call("list_permissible", attribute=ta)
        best = await call("search_permissible", attribute=ta, value=vals[0]["value"], method="token_set")
        sub = await call("submit_value_matches", matches=[
            {"source_attribute": sa, "target_attribute": ta, "source_value": vals[0]["value"],
             "target_value": best["results"][0]["value"], "confidence": 0.6},
            {"source_attribute": sa, "target_attribute": ta, "source_value": vals[1]["value"],
             "target_value": None, "confidence": 0.3}])
        assert sub["accepted"] == 2, sub
        prog = await call("get_progress")
        assert prog["totals"]["submitted_values"] == 1 and prog["totals"]["explicit_none"] == 1
    return {"names": names, "calls": calls}


@pytest.mark.parametrize("mode", ["oracle", "end_to_end"])
def test_session_end_to_end(tmp_path, mode):
    env = {**os.environ, "MATCH_EVAL_LOG_ROOT": str(tmp_path / "logs")}
    args = ["--study", "clark", "--mode", mode] + (["--allow-sql", "--allow-code"] if mode == "oracle"
                                                   else ["--search-only"])
    ex = _start(env, *args)
    sid = ex["SESSION_ID"]
    mcp_cfg = json.loads(open(ex["MCP_CONFIG"]).read())
    assert mcp_cfg["mcpServers"]["matcher"]["env"]["MATCH_EVAL_MODE"] == mode
    settings = json.loads(open(os.path.join(ex["WORKSPACE"], ".claude", "settings.json")).read())
    assert "archive_transcript.py" in settings["hooks"]["Stop"][0]["hooks"][0]["command"]

    got = asyncio.run(_drive(mcp_cfg, mode))
    base = {"describe_study", "profile_column", "sample_rows", "list_values", "describe_attribute",
            "list_permissible", "search_attributes", "search_permissible", "submit_value_matches", "get_progress"}
    if mode == "oracle":
        want = base | {"query", "save_code", "run_code", "list_code"}
    else:  # end_to_end runs here in search-only mode: no list_permissible
        want = (base - {"list_permissible"}) | {"map_attribute", "unmap_attribute"}
        assert "search_permissible" in open(os.path.join(ex["WORKSPACE"], "CLAUDE.md")).read()
        assert "returns at most 5 results" in ex["START_PROMPT"]
    assert got["names"] == want

    fin = subprocess.run([PY, "-I", str(HARNESS_DIR / "session.py"), "finish", sid, "--exit-code", "0"], env=env,
                         capture_output=True, text=True)
    assert fin.returncode == 0, fin.stdout + fin.stderr
    run_dir = tmp_path / "logs" / "runs" / sid
    recs = read_predictions(run_dir / "predictions.jsonl")
    assert len(recs) == 2 and validate(recs)[0] == []
    scores = json.loads((run_dir / "scores.json").read_text())
    assert scores["mode"] == mode and scores["hbench_summary"]["categorical"]["coverage"] > 0
    idx = json.loads((tmp_path / "logs" / "session_index.json").read_text())[sid]
    assert idx["status"] == "finished" and idx["headline"]

    # viewer
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    v = subprocess.Popen([PY, "-I", str(HARNESS_DIR / "viewer" / "serve.py"), "--port", str(port), "--root",
                          str(tmp_path / "logs")], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                sessions = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sessions").read())
                break
            except OSError:
                time.sleep(0.1)
        assert [s["session_id"] for s in sessions["sessions"]] == [sid]
        doc = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/session/{sid}").read())
        server_calls = [e for e in doc["events"] if e["kind"] == "server_log" and e.get("tool")]
        assert len(server_calls) == got["calls"]
        assert doc["summary"]["tool_calls"] == got["calls"] and doc["scores"]["mode"] == mode
        assert doc["summary"]["submissions"]["explicit_none"] == 1
        cmp = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/compare?study=clark&e3=1").read())
        mine = [r for r in cmp["rows"] if r["session_id"] == sid]
        assert len(mine) == 1 and mine[0]["mode"] == mode and mine[0]["tools"]["tool_calls"] == got["calls"]
        assert mine[0]["search_only"] == (mode != "oracle") and mine[0]["code"] == (mode == "oracle")
        assert any(r["source"] == "e3_bdikit" for r in cmp["rows"])
        diff = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/diff?a={sid}&b={sid}").read())
        assert diff["rows"] == [] and diff["n_common"] > 0
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/diff?a={sid}&b=../../etc")
            raise AssertionError("bad diff id accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 400
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert 'id="view-compare"' in page
        assert "<title>Session Timeline</title>" in page
        for bad in ("/api/session/..%2f..%2fetc", "/etc/passwd", "/api/session/not-a-uuid"):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{bad}")
                raise AssertionError(f"{bad} was served")
            except urllib.error.HTTPError as e:
                assert e.code in (400, 404)
    finally:
        v.terminate()
        v.wait(5)
