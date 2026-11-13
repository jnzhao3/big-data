"""E1 Stage 3 analysis: sampling at the endpoints' default temperatures, from a run_plan.py stage3 root (logs only).

  3b  calibration (P7). Each resampled decision has p = p_path, the probability that the sampler reproduces the
      greedy tokens through the end of the decision value. Does the observed rate of "path kept" match p (bins of
      1 - p with Wilson intervals, the sum of 1 - p against the observed count, Brier score against a constant
      forecast)? Controls (p = 1 exactly) must never change. The value can be reached by another path, so the
      value is kept at least as often as the path. On /api/chat, clark's format=json grammar can only raise the
      chance of the greedy token, so there p is a lower bound and the test is one-sided. Agent calls write free
      text before the tool call, so their p_path is near 0 at T=1; for them the forecast that matters is
      p_value_local (the value's own tokens given the greedy prefix) against "value kept", a heuristic.
  3a  seed x lineage (P6). Same seed and lineage (a whole pass repeated): bitwise equal? Same seed on the
      single-token lineage (resend): how often does the text change? Different seeds, for scale.
  Value changes are split by whether the sample gave a value at all (no_answer, empty_output): a thinking model
  can stop after its reasoning without answering.
  3c  whole episodes on /v1. The T=0 reference: per-call cold replays (P5 on /v1) and, given Stage 2's episodes,
      whether /v1 and /api/chat give the same final answers. At the default temperature: how often the final answer
      differs from the T=0 answer, how many distinct answers a column gets, accuracy per seed, whether a repeated
      seed reproduces the episode, and whether the episode changes its answer more often than its final call does
      when resampled alone (the loop amplifying sampling).

Usage: python -I analyze_stage3.py <stage3 root> [--stage2-episodes <stage2_episodes.csv>] [--out <dir>]
Writes stage3_analysis.json, stage3_samples.csv, stage3_decisions.csv, stage3_episodes.csv and stage3_columns.csv
to --out (default: the root).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from hbench.data import Data, default_root  # noqa: E402
from hbench.runlog.writer import read_events  # noqa: E402
from analyze_stage2 import call_signature, candidate_status, lineage_class, load_run  # noqa: E402
from hbench.runlog.writer import BlobStore  # noqa: E402
from resample import compare_run, measured_calls, read_run_events, run_sources, source_key  # noqa: E402

Q_BINS = ((-1e-12, 1e-12), (1e-12, 0.05), (0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.8), (0.8, 1.0))


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def lin_class(canonical) -> str:
    """lineage_class of a CSV cell (missing lineages are NaN)."""
    return lineage_class(canonical if isinstance(canonical, str) else None)


def key_class(key) -> str:
    return str(key).split("@")[0] if key is not None else "none"


def calibration(df: pd.DataFrame, p_col: str = "p_path", kept_col: str = "path_kept") -> dict:
    """Observed changes against a forecast: the path against p_path (the test of the sampler model), or the value
    against p_value_local (a forecast for decisions that follow free text). Value changes are reported too."""
    d = df[df[p_col].notna() & df[kept_col].notna()]
    if d.empty:
        return {"n_samples": 0}
    q = 1 - d[p_col].astype(float)
    changed = (~d[kept_col].astype(bool)).astype(float)
    vchanged = (~d["value_kept"].astype(bool)).astype(float)
    var = float((q * (1 - q)).sum())
    brier = float(((q - changed) ** 2).mean())
    base = float(((changed.mean() - changed) ** 2).mean())
    bins = []
    for lo, hi in Q_BINS:
        m = (q > lo) & (q <= hi)
        n, k = int(m.sum()), int(changed[m].sum())
        if n:
            bins.append({"q": [max(lo, 0.0), hi], "n": n, "mean_predicted": float(q[m].mean()),
                         "observed": k / n, "ci95": wilson(k, n),
                         "observed_value": float(vchanged[m].mean())})
    return {"forecast": p_col, "outcome": kept_col,
            "n_decisions": int(d.groupby(["source_span_id", "decision_key"]).ngroups), "n_samples": int(len(d)),
            "expected_changes": float(q.sum()), "observed_changes": int(changed.sum()),
            "z": (float(changed.sum()) - float(q.sum())) / math.sqrt(var) if var > 0 else None,
            "observed_value_changes": int(vchanged.sum()),
            # value changes with no value: the sample stopped after its reasoning, or wrote no decision
            "no_answer": int(d["no_answer"].astype(bool).sum()) if "no_answer" in d else None,
            "empty_output": int(d["empty_output"].astype(bool).sum()) if "empty_output" in d else None,
            "path_changed_value_kept": int(((changed == 1) & (vchanged == 0)).sum()),
            "brier": brier, "brier_constant": base, "skill": 1 - brier / base if base > 0 else None,
            "inexact_samples": int((~d["p_exact"].astype(bool)).sum()), "bins": bins}


def samples_of(run_dir: Path) -> dict:
    """source key (run/span) -> the measured call of a resampling run."""
    events = read_run_events(run_dir)
    header = next(e for e in events if e["type"] == "run_header")
    sources = run_sources(header, BlobStore(Path(run_dir).parent / "_blobs"))
    return {source_key(s): e for s, e in zip(sources, measured_calls(events))}


def first_diff(a: dict, b: dict):
    ta = ((a.get("output") or {}).get("text") or "") + json.dumps((a.get("output") or {}).get("tool_calls"))
    tb = ((b.get("output") or {}).get("text") or "") + json.dumps((b.get("output") or {}).get("tool_calls"))
    return next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None if len(ta) == len(tb) else min(len(ta), len(tb)))


def pair_stats(pairs: list[tuple[dict, dict]]) -> dict:
    same_text = sum(call_signature(a) == call_signature(b) for a, b in pairs)
    bitwise = sum(a.get("logprobs_sha256") is not None and a.get("logprobs_sha256") == b.get("logprobs_sha256")
                  for a, b in pairs)
    pos = [first_diff(a, b) for a, b in pairs if call_signature(a) != call_signature(b)]
    return {"n": len(pairs), "bitwise": bitwise, "same_text": same_text,
            "first_char_diff_median": float(np.median([p for p in pos if p is not None])) if pos else None}


def seed_lineage(runs: dict, ep: str, p6_spans: set) -> dict:
    """P6 for one endpoint. runs: block -> {seed: run_dir}."""
    cal, rep, res = runs.get(f"{ep}_cal", {}), runs.get(f"{ep}_repeat", {}), runs.get(f"{ep}_resend", {})
    load = {}

    def calls(rd):
        if rd not in load:
            load[rd] = samples_of(rd)
        return load[rd]

    same_lin = [(calls(cal[k])[s], calls(rep[k])[s]) for k in rep if k in cal for s in calls(rep[k])]
    other_lin = [(calls(cal[k])[s], calls(res[k])[s]) for k in res if k in cal for s in calls(res[k])]
    seeds = sorted(k for k in res if k in cal)
    diff_seed = [(calls(cal[a])[s], calls(cal[b])[s]) for a, b in itertools.combinations(seeds, 2)
                 for s in p6_spans if s in calls(cal[a]) and s in calls(cal[b])]
    lin = Counter()
    for k in res:
        from hbench.engine.lineage import _audit_rows
        rows = _audit_rows(res[k])
        lin.update(lineage_class((rows.get(e["span_id"]) or {}).get("canonical")) for e in samples_of(res[k]).values())
    return {"same_seed_same_lineage": pair_stats(same_lin), "same_seed_single_token": pair_stats(other_lin),
            "different_seeds": pair_stats(diff_seed), "resend_lineages": dict(lin)}


def episode_rows(root: Path, steps: dict, data) -> pd.DataFrame:
    tables, rows = {}, []
    for rd, st in steps.items():
        if st.get("kind") != "tool_loop":
            continue
        run = load_run(root / rd)
        gold = data.sm_gold(st["study"]).get(st["column"])
        final = run["final"]
        correct = None if final is None else (final in gold if gold else final == "NONE")
        rows.append({"run": run["name"], "block": st["block"], "study": st["study"], "column": st["column"],
                     "seed": st.get("seed"), "temperature": st.get("temperature"), "final": final,
                     "correct": correct, "n_calls": len(run["calls"]),
                     "loop_status": (run["pred"] or {}).get("loop_status"),
                     "candidates": candidate_status(st["study"], st["column"], data, tables),
                     "sequence": json.dumps([call_signature(c) for c in run["calls"]]),
                     "logprobs": json.dumps([c.get("logprobs_sha256") for c in run["calls"]])})
    return pd.DataFrame(rows)


def agent_section(ep: pd.DataFrame, samples: pd.DataFrame, stage2: pd.DataFrame | None, root: Path,
                  steps: dict) -> tuple[dict, pd.DataFrame]:
    ref = ep[ep.block == "v1_ref"].set_index(["study", "column"])
    smp = ep[ep.block == "v1_sampled"]
    rep = ep[ep.block == "v1_sampled_repeat"].set_index(["study", "column"])
    final_cal = samples[(samples.endpoint == "v1") & (samples.group == "final")
                        & samples.decision_key.map(lambda k: key_class(k) in ("target", "text_target"))]
    cols = []
    for (s, c), g in smp.groupby(["study", "column"]):
        r0 = ref.loc[(s, c)] if (s, c) in ref.index else None
        f0 = None if r0 is None else r0["final"]
        one = final_cal[(final_cal.study == s) & (final_cal.column == c)] if "study" in final_cal else final_cal[:0]
        s1 = g[g.seed == 1]
        rp = rep.loc[(s, c)] if (s, c) in rep.index else None
        cols.append({"study": s, "column": c, "candidates": g["candidates"].iloc[0], "final_T0": f0,
                     "correct_T0": None if r0 is None else r0["correct"], "n_sampled": len(g),
                     "finals": json.dumps(Counter(map(str, g["final"])).most_common()),
                     "n_distinct": g["final"].astype(str).nunique(),
                     "changed_vs_T0": int((g["final"].astype(str) != str(f0)).sum()),
                     "same_sequence_as_T0": int((g["sequence"] == (None if r0 is None else r0["sequence"])).sum()),
                     "accuracy": float(g["correct"].astype(float).mean()),
                     "final_call_value_change": float((~one["value_kept"].astype(bool)).mean()) if len(one) else None,
                     "final_call_no_answer": float(one["decision_replay"].isna().mean()) if len(one) else None,
                     "final_call_other_answer": float((one["decision_replay"].notna()
                                                       & ~one["value_kept"].astype(bool)).mean()) if len(one) else None,
                     "final_call_predicted": float((1 - one["p_value_local"].astype(float)).mean()) if len(one) else None,
                     "seed1_repeat_bitwise": None if rp is None or s1.empty else bool(rp["logprobs"] == s1["logprobs"].iloc[0]),
                     "seed1_repeat_same_final": None if rp is None or s1.empty else bool(str(rp["final"]) == str(s1["final"].iloc[0]))})
    cdf = pd.DataFrame(cols)
    out = {"n_columns": len(cdf), "n_sampled_episodes": int(len(smp))}
    if not cdf.empty:
        out.update({
            "episodes_changed_vs_T0": f"{int(cdf.changed_vs_T0.sum())}/{int(cdf.n_sampled.sum())}",
            "columns_with_any_change": int((cdf.changed_vs_T0 > 0).sum()),
            "distinct_finals_per_column": dict(Counter(cdf.n_distinct.astype(int))),
            "same_sequence_as_T0": f"{int(cdf.same_sequence_as_T0.sum())}/{int(cdf.n_sampled.sum())}",
            "accuracy_T0": float(ref["correct"].astype(float).mean()),
            "accuracy_by_seed": {int(k): float(v) for k, v in smp.groupby("seed")["correct"].apply(
                lambda x: x.astype(float).mean()).items()},
            "seed1_repeat_bitwise": f"{int(cdf.seed1_repeat_bitwise.fillna(False).sum())}/{int(cdf.seed1_repeat_bitwise.notna().sum())}",
            "episodes_without_final": f"{int(smp['final'].isna().sum())}/{int(len(smp))}",
            "amplification": {
                "episode_final_change_rate": float(cdf.changed_vs_T0.sum() / cdf.n_sampled.sum()),
                "final_call_alone_value_change_rate": float(cdf.final_call_value_change.dropna().mean())
                if cdf.final_call_value_change.notna().any() else None,
                "final_call_alone_no_answer_rate": float(cdf.final_call_no_answer.dropna().mean())
                if cdf.final_call_no_answer.notna().any() else None,
                "final_call_alone_other_answer_rate": float(cdf.final_call_other_answer.dropna().mean())
                if cdf.final_call_other_answer.notna().any() else None,
                "final_call_alone_predicted_value_change": float(cdf.final_call_predicted.dropna().mean())
                if cdf.final_call_predicted.notna().any() else None},
            "by_candidates": {k: {"columns": int(len(g)), "changed": f"{int(g.changed_vs_T0.sum())}/{int(g.n_sampled.sum())}",
                                  "accuracy_sampled": float(smp[smp.candidates == k]["correct"].astype(float).mean())}
                              for k, g in cdf.groupby("candidates")},
            "loop_status": dict(Counter(smp["loop_status"].astype(str)))})
    # P5 on /v1 and transport equivalence of the reference
    p5 = Counter()
    for rd, st in steps.items():
        if st.get("kind") == "replay" and (root / rd / "replay_compare.csv").exists():
            rc = pd.read_csv(root / rd / "replay_compare.csv").drop_duplicates("source_span")
            for _, r in rc.iterrows():
                g = "reused_generated" if "G" in lineage_class(r.get("lineage_source")).split("+") else "no_generated"
                p5[(g, bool(r["bitwise_same_logprobs"]))] += 1
    out["p5_v1_cold_replay"] = {g: f"{p5[(g, True)]}/{p5[(g, True)] + p5[(g, False)]} bitwise"
                                for g in ("no_generated", "reused_generated")}
    if stage2 is not None and not ref.empty:
        s2 = stage2[(stage2.block == "unload_each")].groupby(["study", "column"])["final"].first()
        both = [(k, str(ref.loc[k, "final"]), str(s2[k])) for k in ref.index if k in s2.index]
        out["T0_v1_vs_api_chat_finals"] = {"same": sum(a == b for _, a, b in both), "n": len(both),
                                           "different": [f"{k[0]}:{k[1]}" for k, a, b in both if a != b]}
    return out, cdf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--stage2-episodes", default=None, help="stage2_episodes.csv from analyze_stage2.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out_dir = Path(a.out or root)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())["plan"]
    by_name = {s["name"]: s for s in plan["steps"]}
    steps = {}  # run dir (relative) -> step
    for line in (root / "execution.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("status") == "ok":
            for rd in rec.get("runs") or []:
                steps[rd] = by_name[rec["name"]]
    data = Data(default_root())

    # samples of every resampling run, with the source call's study and column where known
    runs, rows = {}, []
    for rd, st in steps.items():
        if st["kind"] != "resample":
            continue
        runs.setdefault(st["block"], {})[st["seed"]] = root / rd
        for r in compare_run(root / rd):
            rows.append({"endpoint": st["block"].split("_")[0], "block": st["block"], "pass_seed": st["seed"], **r})
    samples = pd.DataFrame(rows)
    sel = {ep: json.loads((root / f"selections/{ep}.json").read_text()) for ep in ("chat", "v1")
           if (root / f"selections/{ep}.json").exists()}
    meta = {}
    for ep, s in sel.items():
        for src in s["sources"]:
            meta[src["span_id"]] = src
    report = {"root": str(root), "counts": {"resample_runs": sum(len(v) for v in runs.values()),
                                            "samples": int(len(samples))}}
    if not samples.empty:
        samples["key_class"] = samples["decision_key"].map(key_class)
        ep_of = {}
        for rd, st in steps.items():
            if st["kind"] == "tool_loop":
                ep_of[Path(rd).name] = (st["study"], st["column"])
        for rd, st in steps.items():  # replay runs -> the episode they replayed
            if st["kind"] == "replay":
                hdr = next(e for e in read_events(root / rd) if e["type"] == "run_header")
                ep_of[Path(rd).name] = ep_of.get(Path((hdr.get("task") or {}).get("source_run", "")).name, (None, None))
        samples["study"] = samples["source_run"].map(lambda r: (ep_of.get(r) or (None, None))[0])
        samples["column"] = samples["source_run"].map(lambda r: (ep_of.get(r) or (None, None))[1])
        cal = samples[samples.block.str.endswith("_cal")]
        report["3b"] = {}
        for ep, g in cal.groupby("endpoint"):
            nc = g[g.group != "control"]
            report["3b"][ep] = {"path": calibration(nc),
                                "value_local": calibration(nc, "p_value_local", "value_kept"),
                                "value_local_by_key": {k: calibration(x, "p_value_local", "value_kept")
                                                       for k, x in nc.groupby("key_class")},
                                "control": {"n_samples": int(len(g[g.group == "control"])),
                                            "path_changes": int((~g[g.group == "control"]["path_kept"].astype(bool)).sum()),
                                            "value_changes": int((~g[g.group == "control"]["value_kept"].astype(bool)).sum())},
                                "path_by_key": {k: calibration(x) for k, x in nc.groupby("key_class")},
                                "by_group": {k: calibration(x) for k, x in g.groupby("group") if k != "control"},
                                "sample_lineages": dict(Counter(lin_class(x) for x in g["lineage_replay"])),
                                "source_lineages": dict(Counter(lin_class(x) for x in g["lineage_source"]))}
            if ep in sel and sel[ep].get("population"):
                pop = np.asarray(sel[ep]["population"], dtype=float)
                report["3b"][ep]["population"] = {"n_decisions": int(len(pop)), "p_eq_1": int((pop == 1).sum()),
                                                  "expected_path_changes_per_run": float((1 - pop).sum())}
        report["3a"] = {ep: seed_lineage(runs, ep, {source_key(s) for s in sel[ep]["p6"]}) for ep in sel}
        dec = (cal.groupby(["endpoint", "group", "stratum", "source_run", "source_span_id", "decision_key"])
               .agg(p_path=("p_path", "first"), p_exact=("p_exact", "first"),
                    p_value_local=("p_value_local", "first"), n=("path_kept", "size"),
                    path_changes=("path_kept", lambda x: int((~x.astype(bool)).sum())),
                    value_changes=("value_kept", lambda x: int((~x.astype(bool)).sum())),
                    no_answer=("decision_replay", lambda x: int(x.isna().sum())),
                    decision=("decision", "first"),
                    values=("decision_replay", lambda x: json.dumps(Counter(map(str, x)).most_common())))
               .reset_index())
        dec.to_csv(out_dir / "stage3_decisions.csv", index=False)
        samples.drop(columns=[c for c in ("decision", "decision_replay") if c in samples]).to_csv(
            out_dir / "stage3_samples.csv", index=False)
    ep = episode_rows(root, steps, data)
    if not ep.empty:
        stage2 = pd.read_csv(a.stage2_episodes) if a.stage2_episodes else None
        report["3c"], cols = agent_section(ep, samples if not samples.empty else pd.DataFrame(
            columns=["endpoint", "group", "decision_key", "study", "column", "value_kept", "p_value_local",
                     "decision_replay"]), stage2, root, steps)
        ep.drop(columns=["sequence", "logprobs"]).to_csv(out_dir / "stage3_episodes.csv", index=False)
        cols.to_csv(out_dir / "stage3_columns.csv", index=False)
    (out_dir / "stage3_analysis.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
