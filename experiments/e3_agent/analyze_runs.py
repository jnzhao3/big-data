"""Summarize E3 runs into experiments/e3_agent/results/ (no raw logs).

    python -s experiments/e3_agent/analyze_runs.py [--raw ~/hbench_raw] [--out experiments/e3_agent/results]

For each run in RUNS: the condition, how the run ended, timing, the engine audit, the SM score and, per column,
what the agent changed between Magneto's initial output (first state snapshot) and the final state. For runs that
reached value matching: the categorical VM score right after match_values and at the end, every changed item, and
why each GT item was not attempted. For the acceptance pair: a call-by-call comparison of the two logs.
"""
import argparse
import csv
import json
from pathlib import Path

from hbench.adapters.bdikit import from_bdikit_schema_df, from_bdikit_value_df
from hbench.adapters.bdikit_agent import SM_COLS, VM_COLS, _df, _sv_key
from hbench.data import Data
from hbench.runlog.writer import BlobStore, read_events
from hbench.score_sm import score_sm
from hbench.score_vm import score_vm

STUDY = "clark"
# label -> run-dir suffix under the raw dir. Pipeline failures (template broke tool calls; thinking blocks
# rejected by /v1) are left out; RESULTS.md describes them.
RUNS = {
    "14b_verbatim": "e10c",
    "27b_verbatim_1h": "f69e",
    "27b_dropline_1h": "70fe",
    "27b_verbatim_smvm": "5058",
    "27b_nb1024_T0.6_r1": "f948",
    "27b_nb1024_T0.6_r2": "44c3",
}
PAIR = ("27b_nb1024_T0.6_r1", "27b_nb1024_T0.6_r2")


def find_run(raw: Path, suffix: str) -> Path:
    hits = sorted(raw.glob(f"e3-*-{suffix}"))
    if len(hits) != 1:
        raise SystemExit(f"run *-{suffix}: expected one dir in {raw}, found {len(hits)}")
    return hits[0]


def snapshots(events, field):
    return sorted((e for e in events if e["type"] == "state_snapshot" and e.get(field)),
                  key=lambda e: (e["ts_ns"], e["seq"]))


def sm_records(blobs, snap, run_id):
    df = _df(blobs.get(snap["schema_matching_results"]), SM_COLS).drop_duplicates("source_attribute", keep="last")
    return from_bdikit_schema_df(df, run_id, STUDY, output_type="top1")


def vm_records(blobs, snap, run_id):
    df = _df(blobs.get(snap["value_matching_results"]), VM_COLS)
    keys = df.apply(lambda r: (r.source_attribute, r.target_attribute, _sv_key(r.source_value)), axis=1)
    return from_bdikit_value_df(df[~keys.duplicated(keep="last")], run_id, STUDY, output_type="top1")


def sm_change(annotated, c0, c1, p0, p1):
    if p0 == p1:
        return "unchanged"
    if not annotated:
        return "changed_unannotated"
    return {(False, True): "fixed", (True, False): "broken", (False, False): "wrong_to_wrong",
            (True, True): "correct_to_correct"}[(bool(c0), bool(c1))]


def vm_cat(summary):
    c = summary["categorical"]
    return {"n_items": c["n_items"], "micro": round(c["micro"], 3), "coverage": round(c["coverage"], 3)}


def analyze(label, rd, data, sm_rows):
    events = read_events(rd)
    blobs = BlobStore(rd.parent / "_blobs")
    header = next(e for e in events if e["type"] == "run_header" and e["proc"] == "driver")
    end = next((e for e in events if e["type"] == "run_end" and e["proc"] == "driver"), {})
    audit = json.loads((rd / "engine_audit.json").read_text())["summary"]
    timing = json.loads((rd / "timing.json").read_text())
    tools = [e["tool"] for e in events if e["type"] == "tool_call" and e.get("side") == "server"]
    cond = header["condition"]
    out = {
        "run_id": rd.name,
        "code": {k: header["code"].get(k) for k in ("git_commit", "git_dirty")},
        "model": cond["model"]["name"], "task": cond["method"]["task"],
        "prompt_variant": cond["method"].get("prompt_variant", "verbatim"), "decoding": cond["decoding"],
        "outcome": end.get("outcome"), "followups": end.get("followups"),
        "turn_ends": [t["end"] for t in timing.get("turns", [])],
        "n_agent_steps": timing.get("n_agent_steps"), "tools": tools,
        "time_s": {k: timing.get(k) for k in ("total_wall_s", "agent_llm_s", "tool_s", "tool_llm_s")},
        "llm_calls": {"agent": sum(1 for e in events if e["type"] == "llm_call" and e.get("caller") == "agent"),
                      "tool": timing.get("tool_llm_calls")},
        "audit": {k: audit.get(k) for k in ("n_calls", "matched", "foreign_tasks", "protocol_violations",
                                            "sampler_mismatches", "unmatched_calls")},
    }
    sm = snapshots(events, "schema_matching_results")
    if sm:
        i0, s0 = score_sm(sm_records(blobs, sm[0], rd.name), data, STUDY)
        i1, s1 = score_sm(sm_records(blobs, sm[-1], rd.name), data, STUDY)
        out["sm"] = {"initial_top1": int(s0["top1_correct"]), "final_top1": int(s1["top1_correct"]),
                     "n_annotated": int(s1["n_annotated"]), "final_mrr": round(s1["mrr"], 3)}
        m = i0.merge(i1, on="source_column", suffixes=("_0", "_1"))
        counts = {}
        for r in m.itertuples():
            kind = sm_change(r.annotated_1, r.correct_0, r.correct_1, r.pred_0, r.pred_1)
            counts[kind] = counts.get(kind, 0) + 1
            if kind != "unchanged":
                sm_rows.append({"run": label, "source_column": r.source_column, "initial": r.pred_0,
                                "final": r.pred_1, "gold": r.gold_1, "change": kind})
        out["sm"]["changes"] = counts
    vm = snapshots(events, "value_matching_results")
    if vm and any(e["type"] == "tool_call" and e.get("tool") == "match_values" and not e.get("error")
                  for e in events):
        d0, s0 = score_vm(vm_records(blobs, vm[0], rd.name), data, STUDY, mode="end_to_end")
        d1, s1 = score_vm(vm_records(blobs, vm[-1], rd.name), data, STUDY, mode="end_to_end")
        final_sm = {r["source_column"]: r for r in sm_records(blobs, sm[-1], rd.name)} if sm else {}
        sm_ok = dict(zip(i1.source_column, i1.correct)) if sm else {}

        def why_not(r):
            if r.category != "categorical":
                return f"{r.category}: not asked (task asks for categorical attributes)"
            if sm_ok.get(r.source_attribute):
                return "categorical: column mapped to another GT target"
            return "categorical: column mapped to a wrong target"

        na = d1[~d1.attempted]
        att = d1[d1.attempted & (d1.category == "categorical")]
        m = d0.merge(d1, on="item_id", suffixes=("_0", "_1"))
        out["vm"] = {
            "after_match_values": vm_cat(s0), "final": vm_cat(s1),
            "all_items_micro": round(s1["all_but_no_legal_target"]["micro"], 3),
            "categorical_attempted": {"n": int(len(att)), "correct": int(att.correct.sum())},
            "changed": [{"source": r.source_attribute_1, "value": r.key_1, "before": r.pred_0, "after": r.pred_1,
                         "gold": r.gold_1, "effect": "fixed" if r.correct_1 and not r.correct_0 else
                         "broken" if r.correct_0 and not r.correct_1 else "no effect"}
                        for r in m[m.pred_0 != m.pred_1].itertuples()],
            "wrong": [{"source": r.source_attribute, "target": r.target_attribute, "value": r.key,
                       "pred": r.pred, "gold": r.gold} for r in att[~att.correct].itertuples()],
            "not_attempted": {k: int(v) for k, v in na.apply(why_not, axis=1).value_counts().items()},
            "sm_targets_used": sorted({r["targets"][0] for r in final_sm.values() if r["targets"]} &
                                      set(d1[d1.attempted].target_attribute)),
        }
    return out


def compare_pair(a: Path, b: Path):
    def calls(rd, proc):
        return [(e.get("caller"), e["request_sha256"], (e.get("decoding") or {}).get("seed"), e["output_sha256"],
                 e.get("logprobs_sha256")) for e in read_events(rd) if e["type"] == "llm_call" and e["proc"] == proc]
    res = {"runs": [a.name, b.name]}
    for proc in ("driver", "mcp"):
        x, y = calls(a, proc), calls(b, proc)
        res[proc] = {"n": [len(x), len(y)]} | {
            f"same_{k}": sum(p[i] == q[i] for p, q in zip(x, y))
            for i, k in ((1, "request"), (2, "seed"), (3, "output"), (4, "logprobs"))}
    strip = ("run_id", "ts", "ts_ns", "pid", "seq")
    preds = [[{k: v for k, v in json.loads(l).items() if k not in strip} for l in open(d / "predictions.jsonl")]
             for d in (a, b)]
    res["predictions_identical"] = preds[0] == preds[1]
    res["claims_identical"] = (a / "claims.md").read_bytes() == (b / "claims.md").read_bytes()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(Path.home() / "hbench_raw"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    raw, out = Path(a.raw).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data, sm_rows = Data(), []
    dirs = {label: find_run(raw, sfx) for label, sfx in RUNS.items()}
    runs = {label: analyze(label, rd, data, sm_rows) for label, rd in dirs.items()}
    (out / "runs.json").write_text(json.dumps(runs, indent=1, default=str) + "\n")
    with open(out / "sm_changes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["run", "source_column", "initial", "final", "gold", "change"], lineterminator="\n")
        w.writeheader()
        w.writerows(sm_rows)
    pair = compare_pair(dirs[PAIR[0]], dirs[PAIR[1]])
    (out / "acceptance_pair.json").write_text(json.dumps(pair, indent=1) + "\n")
    for label, r in runs.items():
        sm = r.get("sm", {})
        print(f"{label:22s} {r['outcome'] or '-':15s} steps={r['n_agent_steps']} "
              f"SM {sm.get('initial_top1')}->{sm.get('final_top1')}/{sm.get('n_annotated')} "
              f"VM {r.get('vm', {}).get('final', '-')}")
    print("pair:", json.dumps({k: v for k, v in pair.items() if k != "runs"}))


if __name__ == "__main__":
    main()
