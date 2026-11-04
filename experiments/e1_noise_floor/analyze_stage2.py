"""E1 Stage 2 analysis: agent-shaped lineages, from the runs of a run_plan.py stage2 root (reads logs only).

  2a  repeated episodes per column in three orders: is every replicate bitwise identical (unload_each), or does
      replicate 1 differ from replicates 2.. while those agree (back_to_back, interleaved: prediction P4)?
      Same tool-call sequence and same final answer rates, and the lineage of each episode's first call.
  2b  replay fidelity: per-call cold replays and a whole-run cold_start replay of the unload_each episodes.
      A per-call cold replay recomputes everything in batches, so a call whose context reused generated tokens
      should differ from the live call and every other call should be bitwise identical (P5); a whole-run cold
      replay should reproduce the episode bitwise (P1). Gap shifts at every position up to the first divergence.
  2c  margins of the tool decisions (tool name, inspected column, submitted target), and the expected number of
      decisions per episode that a lineage change flips: sum over decisions of P(shift < -margin), with the
      shift distribution measured in 2b (and Stage 1's, if given).
  st  single-turn back-to-back: clark shortlist runs 6, 7, 8 (is run 8 bitwise equal to run 7?).

Usage: python -I analyze_stage2.py <stage2 root> [--stage1-shifts <stage1_gap_shifts.csv.gz>] [--out <dir>]
Writes stage2_analysis.json, stage2_episodes.csv, stage2_decisions.csv and stage2_shifts.csv.gz to --out
(default: the root).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from hbench.decisions import evidence_from_call, parser_from_spec  # noqa: E402
from hbench.engine.lineage import _audit_rows  # noqa: E402
from hbench.llm.record import load_tokens  # noqa: E402
from hbench.runlog.writer import BlobStore, read_events  # noqa: E402

THRESHOLDS = (0.05, 0.15, 0.5)


def margin_of(entry, tok):
    """Logprob of tok minus the best other candidate's at one position (tok need not be the chosen token)."""
    cands = {c: lp for c, lp in (entry[2] or []) if lp is not None}
    if entry[1] is not None:
        cands.setdefault(entry[0], entry[1])
    if tok not in cands or len(cands) < 2:
        return None
    return cands[tok] - max(lp for c, lp in cands.items() if c != tok)


def shifts(ta, tb) -> list[float]:
    """Gap shifts (b minus a, at a's chosen token) at every position up to and including the first divergence."""
    out = []
    for x, y in zip(ta or [], tb or []):
        ga, gb = margin_of(x, x[0]), margin_of(y, x[0])
        if ga is not None and gb is not None:
            out.append(gb - ga)
        if x[0] != y[0]:
            break
    return out


def lineage_class(canonical: str | None) -> str:
    """Coarse class of a lineage string such as "B237 G16 B54" or "B505 S1"."""
    if not canonical:
        return "unknown"
    parts = canonical.split()
    kinds = {p.rstrip("0123456789") for p in parts}
    return "+".join(k for k in ("B", "b", "G", "S", "?") if k in kinds)


def quantiles(x) -> dict:
    x = np.abs(np.asarray(x, dtype=float))
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)),
            "p99": float(np.quantile(x, 0.99)), "max": float(x.max())}


def load_run(run_dir: Path) -> dict:
    events = read_events(run_dir)
    header = next(e for e in events if e["type"] == "run_header")
    blobs = BlobStore(run_dir.parent / "_blobs")
    lin = _audit_rows(run_dir)
    calls = [e for e in events if e["type"] == "llm_call"]
    return {"dir": run_dir, "name": run_dir.name, "cond": header.get("condition") or {}, "calls": calls,
            "blobs": blobs, "lineage": lin}


def call_signature(c: dict) -> str:
    out = c.get("output") or {}
    return json.dumps({"tool_calls": out.get("tool_calls"), "text": out.get("text")}, sort_keys=True)


def compare_replicates(runs: list[dict]) -> dict:
    """Pairwise comparison of the replicates of one column within one block."""
    runs = sorted(runs, key=lambda r: r["cond"].get("replicate") or 0)
    seq = [[call_signature(c) for c in r["calls"]] for r in runs]
    lp = [[c.get("logprobs_sha256") for c in r["calls"]] for r in runs]
    bit = {(i, j): lp[i] == lp[j] for i, j in itertools.combinations(range(len(runs)), 2)}
    if all(bit.values()):
        shape = "all_bitwise_equal"
    elif len(runs) >= 3 and not bit[(0, 1)] and all(bit[(i, j)] for i, j in bit if i >= 1):
        shape = "first_differs_rest_equal"
    else:
        shape = "other"
    first = [lineage_class((r["lineage"].get(r["calls"][0]["span_id"]) or {}).get("canonical")) if r["calls"]
             else "none" for r in runs]
    return {"n_replicates": len(runs), "same_sequence": len({json.dumps(s) for s in seq}) == 1,
            "same_final": len({s[-1] if s else None for s in seq}) == 1, "shape": shape,
            "pairs_bitwise": sum(bit.values()), "n_pairs": len(bit), "first_call_lineage": first}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--stage1-shifts", default=None, help="Stage 1 stage1_gap_shifts.csv.gz (column dgap)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    out_dir = Path(a.out) if a.out else root
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"root": str(root)}

    runs = [load_run(p.parent) for p in sorted(root.glob("raw/**/events.driver.jsonl"))]
    episodes = [r for r in runs if r["cond"].get("method", {}).get("name") == "e1_tool_loop"]
    replays = [r for r in runs if r["cond"].get("method", {}).get("name") == "replay"]
    single = [r for r in runs if r["cond"].get("method", {}).get("name") == "clark_match"]
    by_span = {c["span_id"]: (r, c) for r in episodes for c in r["calls"]}

    # ---- per-episode table and 2a
    ep_rows = []
    for r in episodes:
        cond, calls = r["cond"], r["calls"]
        pred = next((json.loads(x) for x in open(r["dir"] / "predictions.jsonl")), {}) \
            if (r["dir"] / "predictions.jsonl").exists() else {}
        ep_rows.append({"run": r["name"], "block": cond.get("block"), "order": cond.get("order"),
                        "replicate": cond.get("replicate"), "study": cond["data"]["study"],
                        "column": cond["data"].get("column"), "protocol": cond["engine"].get("protocol"),
                        "n_calls": len(calls), "loop_status": pred.get("loop_status"),
                        "final": (pred.get("targets") or [None])[0], "final_margin": (pred.get("evidence") or {}).get("margin"),
                        "lineages": " | ".join(lineage_class((r["lineage"].get(c["span_id"]) or {}).get("canonical"))
                                               for c in calls)})
    eps = pd.DataFrame(ep_rows)
    eps.to_csv(out_dir / "stage2_episodes.csv", index=False)
    groups = defaultdict(list)
    for r in episodes:
        groups[(r["cond"].get("block"), r["cond"]["data"]["study"], r["cond"]["data"].get("column"))].append(r)
    rep_rows = [{"block": b, "study": s, "column": c, **compare_replicates(g)} for (b, s, c), g in groups.items()]
    reps = pd.DataFrame(rep_rows)
    report["2a"] = {}
    for b, g in reps.groupby("block") if len(reps) else []:
        report["2a"][b] = {"n_columns": len(g), "shapes": g["shape"].value_counts().to_dict(),
                           "same_sequence": int(g.same_sequence.sum()), "same_final": int(g.same_final.sum()),
                           "pairs_bitwise": f"{int(g.pairs_bitwise.sum())}/{int(g.n_pairs.sum())}",
                           "first_call_lineage_by_replicate": [dict(Counter(x[i] for x in g.first_call_lineage
                                                                            if len(x) > i))
                                                               for i in range(int(g.n_replicates.max()))]}
    report["lineage_classes_by_step"] = {}
    for r in episodes:
        for c in r["calls"]:
            k = f"step{c.get('step')}"
            cls = lineage_class((r["lineage"].get(c["span_id"]) or {}).get("canonical"))
            report["lineage_classes_by_step"].setdefault(k, Counter())[cls] += 1
    report["lineage_classes_by_step"] = {k: dict(v) for k, v in sorted(report["lineage_classes_by_step"].items())}

    # ---- 2b: replay fidelity and gap shifts
    shift_rows, fid_rows = [], []
    for rp in replays:
        proto = rp["cond"]["method"].get("protocol")
        for c in rp["calls"]:
            src = by_span.get(c.get("replay_of"))
            if src is None or c.get("caller") == "replay:pre":
                continue
            srun, scall = src
            src_lin = (srun["lineage"].get(scall["span_id"]) or {}).get("canonical")
            ta, tb = load_tokens(scall, srun["blobs"]), load_tokens(c, rp["blobs"])
            sh = shifts(ta, tb)
            fid_rows.append({"protocol": proto, "step": scall.get("step"), "source_lineage": lineage_class(src_lin),
                             "reused_generated": "G" in lineage_class(src_lin),
                             "bitwise": scall.get("logprobs_sha256") == c.get("logprobs_sha256"),
                             "same_output": call_signature(scall) == call_signature(c),
                             "max_abs_shift": max((abs(x) for x in sh), default=0.0)})
            shift_rows += [{"protocol": proto, "step": scall.get("step"), "source_lineage": lineage_class(src_lin),
                            "dgap": x} for x in sh]
    fid = pd.DataFrame(fid_rows)
    shf = pd.DataFrame(shift_rows)
    if len(shf):
        shf.to_csv(out_dir / "stage2_shifts.csv.gz", index=False)
    report["2b"] = {}
    if len(fid):
        for (proto, reused), g in fid.groupby(["protocol", "reused_generated"]):
            report["2b"][f"{proto}:{'reused_generated' if reused else 'no_generated'}"] = {
                "n_calls": len(g), "bitwise": int(g.bitwise.sum()), "same_output": int(g.same_output.sum())}
        report["2b"]["shifts"] = {f"{p}": quantiles(g.dgap) for p, g in shf.groupby("protocol")} if len(shf) else {}

    # ---- 2c: decision margins and the engine-noise budget
    dec_rows = []
    for r in episodes:
        for c in r["calls"]:
            parser = parser_from_spec(c.get("decision_spec"))
            for d in parser(c.get("output") or {}) if parser else []:
                ev = evidence_from_call(c, r["blobs"], key=d["key"])
                if ev.get("margin_found"):
                    dec_rows.append({"run": r["name"], "block": r["cond"].get("block"), "step": c.get("step"),
                                     "key": d["key"].split("@")[0], "call_index": d.get("index", 0),
                                     "margin": ev.get("margin"), "margin_value": ev.get("margin_value"),
                                     "runner_up": ev.get("runner_up")})
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(out_dir / "stage2_decisions.csv", index=False)
    report["2c"] = {}
    for k, g in dec.groupby("key") if len(dec) else []:
        m = g.margin.dropna()
        report["2c"][k] = {"n": len(m), **{f"below_{t}": int((m < t).sum()) for t in THRESHOLDS},
                           "q": {q: float(m.quantile(q)) for q in (0.01, 0.1, 0.5)} if len(m) else {}}
    dists = {}
    if len(shf):
        dists["stage2_cold"] = np.sort(shf[shf.protocol == "cold"].dgap.to_numpy())
    if a.stage1_shifts:
        dists["stage1_single_token"] = np.sort(pd.read_csv(a.stage1_shifts).dgap.to_numpy())
    budget = {}
    finals = dec[dec.key.isin(["target", "text_target"])] if len(dec) else dec
    for name, d in dists.items():
        if not len(d) or not len(finals):
            continue
        p = np.searchsorted(d, -finals.margin.fillna(np.inf).to_numpy(), side="left") / len(d)
        budget[name] = {"expected_final_flips_per_episode": float(p.sum() / max(finals.run.nunique(), 1)),
                        "n_final_decisions": int(len(finals)), "n_shift_samples": int(len(d))}
    report["2c"]["noise_budget"] = budget

    # ---- single-turn back-to-back
    st = sorted(single, key=lambda r: r["cond"].get("replicate") or 0)
    if len(st) >= 2:
        lp = [[c.get("logprobs_sha256") for c in r["calls"]] for r in st]
        report["single_turn"] = {
            "runs": [r["name"] for r in st],
            "bitwise_vs_previous": [sum(x == y for x, y in zip(lp[i - 1], lp[i])) for i in range(1, len(st))],
            "n_calls": [len(x) for x in lp],
            "single_token_calls": [sum("S" in lineage_class((r["lineage"].get(c["span_id"]) or {}).get("canonical"))
                                       for c in r["calls"]) for r in st]}

    report["counts"] = {"episodes": len(episodes), "replay_runs": len(replays), "single_turn_runs": len(single)}
    (out_dir / "stage2_analysis.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
