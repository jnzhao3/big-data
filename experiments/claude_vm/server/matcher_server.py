"""The `matcher` MCP server (stdio): one process serves one session, one study and one mode.

Configured by environment variables, which run_eval.sh writes into the session's MCP config:
    MATCH_EVAL_SESSION_ID   session id (a UUID; also the Claude Code session id)
    MATCH_EVAL_STUDY        GDC study, e.g. clark
    MATCH_EVAL_MODE         oracle | end_to_end
    ALLOW_SQL               true to expose the read-only `query` tool (default false)
    MATCH_EVAL_ROW_CAP      row cap per call (default 50)
    MATCH_EVAL_SEARCH_ONLY  true: no list_permissible; permissible values only through search_permissible
    MATCH_EVAL_SEARCH_TOP_K results per search_permissible call in search-only mode (default 5)
    MATCH_EVAL_ALLOW_CODE   true to expose save_code / run_code / list_code (restricted Python, see cvm/codebox.py)
    MATCH_EVAL_LOG_ROOT     log root (default ~/hbench_raw/claude_vm)
    HBENCH_DATA             benchmark data root (default <repo>/datasets)

Every tool call is appended to <log root>/server/<session_id>.jsonl with its arguments, result and timing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/claude_vm, for `cvm`

from pydantic import BaseModel, Field  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from cvm.codebox import ALLOWED_MODULES, CodeStore  # noqa: E402
from cvm.paths import SessionPaths  # noqa: E402
from cvm.sqlguard import SqlGuard, build_db  # noqa: E402
from cvm.store import SubmissionStore, now_iso  # noqa: E402
from cvm.study import StudyView  # noqa: E402
from cvm.tools import DEFAULT_CAP, DEFAULT_SEARCH_TOP_K, METHODS, MatcherTools  # noqa: E402
from hbench.data import Data  # noqa: E402


class ServerLog:
    def __init__(self, path: Path, session_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id

    def write(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"ts": now_iso(), "session_id": self.session_id, **rec}, ensure_ascii=False,
                               default=str) + "\n")

    def call(self, tool: str, arguments: dict, fn):
        t0 = time.perf_counter()
        result, error = None, None
        try:
            result = fn()
            return result
        except Exception as e:  # logged, then returned to the model as a tool error
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            payload = json.dumps(result, ensure_ascii=False, default=str) if result is not None else ""
            rows = result.get("returned") if isinstance(result, dict) else None
            self.write({"event": "tool_call", "tool": tool, "arguments": arguments, "result_rows": rows,
                        "result_bytes": len(payload.encode()), "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                        "error": error, "result": result})


def _env_session_id() -> str:
    sid = os.environ.get("MATCH_EVAL_SESSION_ID")
    if sid:
        return sid
    sid = str(uuid.uuid4())
    print(f"warning: MATCH_EVAL_SESSION_ID is not set; using generated id {sid}", file=sys.stderr)
    return sid


class Match(BaseModel):
    source_attribute: str = Field(description="Source column")
    target_attribute: str = Field(description="GDC attribute")
    source_value: str | int | float = Field(description='Source value as listed by list_values ("" = empty cell)')
    target_value: str | int | float | None = Field(
        description="GDC value, or null for an explicit NONE (no GDC value fits)")
    confidence: float = Field(ge=0, le=1, description="Probability that this answer is correct")
    rationale: str | None = Field(default=None, description="Optional short reason")


def build_server() -> FastMCP:
    sid = _env_session_id()
    paths = SessionPaths(sid)
    log = ServerLog(paths.server_log, sid)
    if not os.environ.get("MATCH_EVAL_SESSION_ID"):
        log.write({"event": "warning", "message": "MATCH_EVAL_SESSION_ID not set; generated a session id"})
    study = os.environ.get("MATCH_EVAL_STUDY", "clark")
    mode = os.environ.get("MATCH_EVAL_MODE", "oracle")
    allow_sql = os.environ.get("ALLOW_SQL", "false").lower() == "true"
    cap = int(os.environ.get("MATCH_EVAL_ROW_CAP", DEFAULT_CAP))
    search_only = os.environ.get("MATCH_EVAL_SEARCH_ONLY", "false").lower() == "true"
    search_top_k = int(os.environ.get("MATCH_EVAL_SEARCH_TOP_K", DEFAULT_SEARCH_TOP_K))
    allow_code = os.environ.get("MATCH_EVAL_ALLOW_CODE", "false").lower() == "true"

    view = StudyView(Data(), study, mode)
    sql = None
    if allow_sql:
        db = paths.sql_db if paths.sql_db.exists() else build_db(paths.sql_db, view)
        sql = SqlGuard(db, cap)
    tools = MatcherTools(view, SubmissionStore(paths.submissions), cap=cap, sql=sql, search_only=search_only,
                         search_top_k=search_top_k, code=CodeStore(paths.run_dir / "code") if allow_code else None)
    if search_only and paths.server_log.exists():
        # a restarted server keeps what earlier searches of this session returned
        for line in paths.server_log.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("tool") == "search_permissible" and isinstance(e.get("result"), dict):
                r = e["result"]
                tools.searched.update((r.get("attribute"), x.get("value")) for x in r.get("results") or [])
    log.write({"event": "server_start", "study": view.study, "mode": mode, "allow_sql": allow_sql, "row_cap": cap,
               "search_only": search_only, "search_top_k": tools.search_top_k if search_only else None,
               "allow_code": allow_code,
               "pid": os.getpid()})

    mcp = FastMCP("matcher", instructions=(
        f"Value matching of GDC study '{view.study}' ({mode} mode). Map each source value to a GDC value or to "
        "NONE with these tools, and submit every answer with submit_value_matches."))

    def run(name, args, fn):
        return log.call(name, args, fn)

    @mcp.tool(description="The study's source columns (paged), row count and, in oracle mode, the attribute "
                          "pairs to map.")
    def describe_study(offset: int = 0, limit: int = cap, pairs_offset: int = 0) -> dict:
        a = dict(offset=offset, limit=limit, pairs_offset=pairs_offset)
        return run("describe_study", a, lambda: tools.describe_study(**a))

    @mcp.tool(description="Profile a source column: distinct count, null rate, top 20 values, lengths, value "
                          "shapes and casing.")
    def profile_column(column: str) -> dict:
        return run("profile_column", {"column": column}, lambda: tools.profile_column(column))

    @mcp.tool(description="Seedable random sample of source rows, optionally stratified by a column (capped).")
    def sample_rows(n: int = 10, seed: int = 0, stratify_by: str | None = None) -> dict:
        a = dict(n=n, seed=seed, stratify_by=stratify_by)
        return run("sample_rows", a, lambda: tools.sample_rows(**a))

    @mcp.tool(description='Distinct values of a source column with counts (paged): the values to map. "" stands '
                          "for empty cells.")
    def list_values(column: str, offset: int = 0, limit: int = cap) -> dict:
        a = dict(column=column, offset=offset, limit=limit)
        return run("list_values", a, lambda: tools.list_values(**a))

    @mcp.tool(description="A GDC attribute's description, whether it is enumerated, and its number of "
                          "permissible values.")
    def describe_attribute(attribute: str) -> dict:
        return run("describe_attribute", {"attribute": attribute}, lambda: tools.describe_attribute(attribute))

    if not search_only:
        @mcp.tool(description="Permissible values of a GDC attribute with their descriptions (paged).")
        def list_permissible(attribute: str, offset: int = 0, limit: int = cap) -> dict:
            a = dict(attribute=attribute, offset=offset, limit=limit)
            return run("list_permissible", a, lambda: tools.list_permissible(**a))

    @mcp.tool(description=f"Find GDC attributes by name or description. method: one of {', '.join(METHODS)}.")
    def search_attributes(query: str, method: str = "token_set", top_k: int = 10) -> dict:
        a = dict(query=query, method=method, top_k=top_k)
        return run("search_attributes", a, lambda: tools.search_attributes(**a))

    search_note = (f" Returns at most {tools.search_top_k} results per call: search once per source value."
                   if search_only else "")

    @mcp.tool(description=f"Closest permissible values of one GDC attribute to a value. method: one of "
                          f"{', '.join(METHODS)}.{search_note}")
    def search_permissible(attribute: str, value: str, method: str = "token_set", top_k: int = 10,
                           min_score: float = 0.0) -> dict:
        a = dict(attribute=attribute, value=value, method=method, top_k=top_k, min_score=min_score)
        return run("search_permissible", a, lambda: tools.search_permissible(**a))

    @mcp.tool(description="Submit answers. One GDC value per item, or target_value null for an explicit NONE. "
                          "Re-submitting an item replaces the earlier answer.")
    def submit_value_matches(matches: list[Match]) -> dict:
        ms = [m.model_dump() for m in matches]
        return run("submit_value_matches", {"matches": ms}, lambda: tools.submit_value_matches(ms))

    @mcp.tool(description="Per attribute pair: values submitted, explicit NONEs and values remaining. Does not "
                          "say whether answers are correct.")
    def get_progress(offset: int = 0, limit: int = cap) -> dict:
        a = dict(offset=offset, limit=limit)
        return run("get_progress", a, lambda: tools.get_progress(**a))

    if mode == "end_to_end":
        @mcp.tool(description="Record that a source column corresponds to a GDC attribute. A column may be mapped "
                              "to several attributes.")
        def map_attribute(source_attribute: str, target_attribute: str) -> dict:
            a = dict(source_attribute=source_attribute, target_attribute=target_attribute)
            return run("map_attribute", a, lambda: tools.map_attribute(**a))

        @mcp.tool(description="Remove an attribute mapping recorded with map_attribute.")
        def unmap_attribute(source_attribute: str, target_attribute: str) -> dict:
            a = dict(source_attribute=source_attribute, target_attribute=target_attribute)
            return run("unmap_attribute", a, lambda: tools.unmap_attribute(**a))

    if allow_code:
        mods = ", ".join(ALLOWED_MODULES)
        @mcp.tool(description=(
            "Save a Python function for later use. The code must define run(inputs) and may import only "
            f"{mods}. No files, network, other imports, eval/exec/getattr, or names starting with '_'. "
            "Inside run you can call values(column) and permissible(attribute) for the columns/attributes "
            "named in run_code. Saving under an existing name adds a new version."))
        def save_code(name: str, code: str, description: str = "") -> dict:
            a = dict(name=name, code=code, description=description)
            return run("save_code", a, lambda: tools.save_code(**a))

        @mcp.tool(description=(
            "Run a saved function: run(inputs) gets `inputs` (any JSON). columns=[...] makes values(column) "
            "available (the column's distinct values; '' is the empty cell); attributes=[...] makes "
            "permissible(attribute) available. A list or dict result is paged with offset/limit. Limits: "
            "10 s, 5 s CPU, 768 MB. The model still submits answers with submit_value_matches."))
        def run_code(name: str, inputs: dict | list | str | int | float | None = None,
                     columns: list[str] | None = None, attributes: list[str] | None = None,
                     version: int | None = None, offset: int = 0, limit: int = cap) -> dict:
            a = dict(name=name, inputs=inputs, columns=columns, attributes=attributes, version=version,
                     offset=offset, limit=limit)
            return run("run_code", a, lambda: tools.run_code(**a))

        @mcp.tool(description="List the saved functions and their versions.")
        def list_code() -> dict:
            return run("list_code", {}, tools.list_code)

    if allow_sql:
        @mcp.tool(description="Read-only SQL (one SELECT) over tables source, attributes(name, description, "
                              "enumerated, n_permissible) and permissible_values(attribute, value, description). "
                              "Capped rows.")
        def query(sql: str) -> dict:
            return run("query", {"sql": sql}, lambda: tools.query(sql))

    return mcp


if __name__ == "__main__":
    build_server().run()
