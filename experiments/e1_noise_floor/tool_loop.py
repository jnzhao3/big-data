"""E1 Stage 2 driver: a synthetic tool loop (agent-shaped requests with fixed tool returns).

An agent's calls are computed differently from a single-turn method's: every turn reuses the previous turn's
prompt and part of its generated tool-call tokens (the chat template re-serializes the arguments, so reuse stops
partway through them), and a repeated episode can find its first prompt already cached. These are the lineages
E1 Stage 2 measures. The loop isolates the engine: tool returns are a deterministic function of the arguments,
so two episodes of one column can differ only through the engine (or the sampler, above T=0). It is not the
BDI-Kit agent, whose tools have run-to-run noise of their own (row order); E3's agent is measured separately.

One episode = one hbench run: the model matches one source column to a GDC target column with three tools,

    list_candidate_columns(source_column)   the 10-column shortlist (the same rule as clark_match's shortlist)
    get_column_values(column, side)         up to 5 sample values of a source or target column
    submit_match(source_column, target)     the decision; ends the episode

Every call carries a decision spec (tool name, the column argument, the submitted target, or a target in JSON
text if the model answers without a tool), so hbench gives each tool decision a margin. The episode's run folder
holds the usual run log, predictions.jsonl (one record, with the evidence of the deciding call) and episode.json
(the steps).

Usage: python -I tool_loop.py --study clark --column Tumor_Size_cm [--replicate 1] [--block NAME] [--order N]
       [--engine-protocol natural|cold_start] [--transport ollama|openai_v1] [--model qwen2.5:14b]
       [--temperature 0] [--seed 1] [--max-steps 5] [--raw-dir <run logs>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the hbench of this checkout (its commit is in the run header)
from hbench.data import Data, default_root  # noqa: E402
from hbench.design import make_condition  # noqa: E402
from hbench.llm import LLMClient  # noqa: E402
from hbench.predictions import sm_record, write_predictions  # noqa: E402
from hbench.runlog.session import raw_root, start_run  # noqa: E402

MODEL = "qwen2.5:14b"
CALLER = "method:e1_tool_loop"
TOOLS_VERSION = "v1"  # bump when the tools, their returns or the prompt change: it is part of the condition
SHORTLIST_K = 10
TOOLS = [
    {"type": "function", "function": {
        "name": "list_candidate_columns",
        "description": "List the GDC target columns most likely to match a source column.",
        "parameters": {"type": "object", "properties": {"source_column": {"type": "string"}},
                       "required": ["source_column"]}}},
    {"type": "function", "function": {
        "name": "get_column_values",
        "description": "Return up to 5 sample values of a column. side is 'source' or 'target'.",
        "parameters": {"type": "object", "properties": {"column": {"type": "string"},
                                                        "side": {"type": "string", "enum": ["source", "target"]}},
                       "required": ["column", "side"]}}},
    {"type": "function", "function": {
        "name": "submit_match",
        "description": "Submit the final answer: the GDC target column that matches the source column, or NONE.",
        "parameters": {"type": "object", "properties": {"source_column": {"type": "string"},
                                                        "target": {"type": "string"}},
                       "required": ["source_column", "target"]}}},
]
SYSTEM = ("You match columns of a source clinical dataset to the GDC (Genomic Data Commons) data model. "
          "Use the tools: list the candidate target columns first, look at sample values when that helps, "
          "then call submit_match exactly once. Submit NONE if no candidate fits.")
USER = 'Source column: "{col}". Find the matching GDC target column.'
# Where each call's decisions are: which tool it called, which column it looked at, and the submitted target
# (or a target in JSON text, if the model answered without a tool). The model may make several calls in one
# turn, so every field is named for the first MAX_PARALLEL calls; keys are "tool", "tool@1", ... Absent parts
# yield no decision.
MAX_PARALLEL = 3
DECISION = [{"type": "tool_call", "field": f, "index": i, "key": k if i == 0 else f"{k}@{i}"}
            for i in range(MAX_PARALLEL)
            for f, k in (("name", "tool"), ("arguments.column", "column"), ("arguments.target", "target"))]
DECISION.append({"type": "json_field", "field": "target", "key": "text_target"})


def norm(s) -> str:  # "Tumor_Size_cm" -> "tumor size cm"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def samples(series: pd.Series, n: int = 5) -> list[str]:
    return [v[:60] for v in series.dropna().astype(str).unique()[:n]]


class Tables:
    """A study's source and target tables, and the fixed tool returns computed from them."""

    def __init__(self, data: Data, study: str):
        d = data.source_path(study).parent
        self.study = study
        self.src = pd.read_csv(d / "source.csv", low_memory=False)
        self.tgt = pd.read_csv(d / "target.csv", low_memory=False)
        self._tvals = {c: set(v.lower() for v in self.tgt[c].dropna().astype(str).unique()) for c in self.tgt.columns}
        self._tnorm = {c: norm(c) for c in self.tgt.columns}

    def shortlist(self, col: str, k: int = SHORTLIST_K) -> list[str]:
        """Half by value overlap, then by name similarity: the rule of clark_match's shortlist, so the loop sees
        the candidates the single-turn method shows."""
        n = norm(col)
        toks = set(n.split())
        name = {c: SequenceMatcher(None, n, tn).ratio() + 0.5 * len(toks & set(tn.split())) / max(len(toks), 1)
                for c, tn in self._tnorm.items()}
        svals = set(v.lower() for v in self.src[col].dropna().astype(str).unique())

        def hit(v, tv):
            return v in tv or (len(v) >= 2 and any(v in x or (len(x) >= 2 and x in v) for x in tv))
        overlap = {c: sum(hit(v, tv) for v in svals) / max(len(svals), 1) for c, tv in self._tvals.items() if tv}
        by_name = sorted(name, key=name.get, reverse=True)
        by_val = [c for c in sorted(overlap, key=overlap.get, reverse=True) if overlap[c] > 0]
        out = []
        for c in by_val[: k // 2] + by_name:
            if c not in out:
                out.append(c)
            if len(out) == k:
                break
        return out

    def tool_result(self, name: str, args: dict | None) -> str:
        """The tool's return as JSON text: a function of its arguments only."""
        args = args if isinstance(args, dict) else {}
        if name == "list_candidate_columns":
            col = args.get("source_column")
            if col not in self.src.columns:
                return json.dumps({"error": f"unknown source column {col!r}"})
            return json.dumps(self.shortlist(col))
        if name == "get_column_values":
            col, side = args.get("column"), args.get("side")
            table = self.src if side == "source" else self.tgt if side == "target" else None
            if table is None:
                return json.dumps({"error": "side must be 'source' or 'target'"})
            if col not in table.columns:
                return json.dumps({"error": f"unknown {side} column {col!r}"})
            return json.dumps(samples(table[col]))
        if name == "submit_match":
            return json.dumps({"recorded": True})
        return json.dumps({"error": f"unknown tool {name!r}"})


def next_messages(res, transport: str, results: list[str]) -> list[dict]:
    """The assistant turn as the server returned it, then one tool message per call, in the transport's format."""
    if transport == "ollama":
        msg = res.response["message"]
        out = [{"role": "assistant", "content": msg.get("content", ""), "tool_calls": msg.get("tool_calls")}]
        out += [{"role": "tool", "content": r, "tool_name": tc["name"]} for tc, r in zip(res.tool_calls, results)]
    else:
        msg = res.response["choices"][0]["message"]
        raw = msg.get("tool_calls") or []
        out = [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": raw}]
        out += [{"role": "tool", "tool_call_id": tc.get("id"), "content": r} for tc, r in zip(raw, results)]
    return out


def text_target(text: str | None) -> str | None:
    try:
        obj = json.loads(text or "")
    except json.JSONDecodeError:
        return None
    t = obj.get("target") if isinstance(obj, dict) else None
    return t.strip() if isinstance(t, str) else None


def condition(study: str, column: str, *, replicate: int, block, order, protocol: str, transport: str, model: str,
              temperature, seed, max_steps: int) -> dict:
    return make_condition(method={"name": "e1_tool_loop", "tools": TOOLS_VERSION, "k": SHORTLIST_K,
                                  "max_steps": max_steps},
                          model={"name": model}, decoding={"temperature": temperature, "seed": seed},
                          engine={"transport": transport, "num_ctx": 32768, "protocol": protocol},
                          data={"study": study, "column": column}, replicate=replicate, block=block, order=order)


def run_episode(study: str, column: str, *, replicate: int = 1, block=None, order=None, protocol: str = "natural",
                transport: str = "ollama", model: str = MODEL, temperature=0, seed=1, max_steps: int = 5,
                raw_dir=None, data: Data | None = None, tables: Tables | None = None) -> dict:
    """Run one episode as its own hbench run; returns its summary (also written to <run_dir>/episode.json)."""
    data = data or Data(default_root())
    tables = tables or Tables(data, study)
    if column not in tables.src.columns:
        raise ValueError(f"{column!r} is not a source column of {study}")
    cond = condition(study, column, replicate=replicate, block=block, order=order, protocol=protocol,
                     transport=transport, model=model, temperature=temperature, seed=seed, max_steps=max_steps)
    log = start_run(Path(raw_dir) if raw_dir else raw_root(), "e1_tool_loop", cond,
                    task={"kind": "sm", "study": study, "column": column}, models=[model],
                    prompts={"system": SYSTEM, "user": USER, "tools": json.dumps(TOOLS)}, data=data)
    client = LLMClient(log, model, transport=transport)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(col=column)}]
    steps, final, final_res, final_key, status = [], None, None, None, "max_steps"
    try:
        for step in range(max_steps):
            res = client.chat(msgs, caller=CALLER, step=step, tools=TOOLS, decision=DECISION)
            calls = res.tool_calls or []
            steps.append({"step": step, "span_id": res.span_id, "path_class": res.path_class,
                          "prompt_tokens": res.prompt_tokens, "cached_tokens": res.cached_tokens,
                          "tool_calls": calls, "text": res.text})
            if not calls:
                final, final_res, final_key, status = text_target(res.text), res, "text_target", "text_answer"
                break
            results = [tables.tool_result(tc["name"], tc.get("arguments")) for tc in calls]
            i = next((i for i, tc in enumerate(calls) if tc["name"] == "submit_match"), None)
            if i is not None:
                t = (calls[i].get("arguments") or {}).get("target")
                final, final_res, status = (t.strip() if isinstance(t, str) else None), res, "submitted"
                final_key = "target" if i == 0 else f"target@{i}"
                break
            msgs += next_messages(res, transport, results)
        shown = tables.shortlist(column)
        ev = final_res.evidence(key=final_key) if final_res is not None else None
        pred = [] if final in (None, "", "NONE") else [final]
        rec = sm_record(log.run_id, study, column, pred, shown=shown,
                        status="ok" if final is not None else "invalid_output", evidence=ev, loop_status=status,
                        n_steps=len(steps))
        write_predictions(log.run_dir / "predictions.jsonl", [rec])
        log.emit("prediction_batch", path="predictions.jsonl", n=1)
        summary = {"run_id": log.run_id, "run_dir": str(log.run_dir), "study": study, "column": column,
                   "replicate": replicate, "block": block, "order": order, "protocol": protocol,
                   "status": status, "final": final, "n_steps": len(steps), "steps": steps}
        log.write_file("episode.json", json.dumps(summary, indent=1))
        log.close(status="ok", loop_status=status, n_steps=len(steps))
    except BaseException:
        log.close(status="error")
        raise
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", required=True)
    ap.add_argument("--column", required=True)
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--block", default=None, help="design block (e.g. back_to_back), recorded in the condition")
    ap.add_argument("--order", type=int, default=None, help="position in the run plan, recorded in the condition")
    ap.add_argument("--engine-protocol", choices=["natural", "cold_start"], default="natural")
    ap.add_argument("--transport", choices=["ollama", "openai_v1"], default="ollama")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--raw-dir", default=None, help="run logs (default: $HBENCH_RAW or ~/hbench_raw, outside the repo)")
    a = ap.parse_args(argv)
    s = run_episode(a.study, a.column, replicate=a.replicate, block=a.block, order=a.order,
                    protocol=a.engine_protocol, transport=a.transport, model=a.model, temperature=a.temperature,
                    seed=a.seed, max_steps=a.max_steps, raw_dir=a.raw_dir)
    print(json.dumps({k: v for k, v in s.items() if k != "steps"}))
    for st in s["steps"]:
        print(f"  step {st['step']}: {st['path_class']:12s} {json.dumps(st['tool_calls'] or st['text'])[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
