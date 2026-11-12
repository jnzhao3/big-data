"""Command line: python -m hbench <command> (run from the repo root, or with hbench installed in the env)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import SCORING_VERSION
from .data import SM_GT_VARIANTS, Data
from .manifest import MANIFEST_PATH, build_manifest, verify_manifest
from .predictions import read_predictions, validate, write_predictions


def _write_new(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        sys.exit(f"{path} exists; pass --force to replace it")
    path.write_text(text)


def cmd_manifest(a) -> int:
    data = Data(a.root)
    out = Path(a.manifest) if a.manifest else MANIFEST_PATH
    if a.action == "build":
        _write_new(out, json.dumps(build_manifest(data.root), indent=1, sort_keys=True) + "\n", a.force)
        print(f"wrote {out}")
        return 0
    problems = verify_manifest(data.root, out)
    print("\n".join(problems) if problems else f"ok: data matches {out}")
    return 1 if problems else 0


def cmd_preds(a) -> int:
    if a.action == "validate":
        recs = read_predictions(a.input)
        data = Data(a.root)
        errors, warnings = validate(recs, data.target_columns if data.exists() else None)
        for w in warnings:
            print("warning:", w)
        for e in errors:
            print("error:", e)
        print(f"{len(recs)} records, {len(errors)} errors, {len(warnings)} warnings")
        return 1 if errors else 0
    if a.action == "convert":
        if a.format == "clark-match":
            from .adapters.clark_match import from_clark_match_csv
            recs = from_clark_match_csv(a.input, a.run_id, a.study)
        elif a.format == "magneto-txt":
            from .adapters.magneto import from_magneto_txt
            recs = from_magneto_txt(Path(a.input).read_text(), a.run_id or Path(a.input).stem, a.study)
        elif a.format in ("bdikit-sm-csv", "bdikit-vm-csv"):
            from .adapters import bdikit
            df = pd.read_csv(a.input)
            fn = bdikit.from_bdikit_schema_df if a.format == "bdikit-sm-csv" else bdikit.from_bdikit_value_df
            recs = fn(df, a.run_id or Path(a.input).stem, a.study, a.output_type)
        else:
            sys.exit(f"unknown format {a.format}")
        print(f"wrote {len(recs)} records to {write_predictions(a.out, recs)}")
        return 0
    return 2


def score_file(pred_path: Path, data: Data, gts, vm_mode: str):
    """Score every (run, study) in a predictions file. Returns {gt: (items DataFrame, summaries list)}."""
    from .score_sm import score_sm
    from .score_vm import score_vm

    recs = read_predictions(pred_path)
    cond_id = _run_condition_id(pred_path.parent)
    errors, _ = validate(recs)
    if errors:
        raise ValueError(f"{pred_path}: {len(errors)} invalid records, e.g. {errors[0]}")
    groups: dict[tuple, list] = {}
    for r in recs:
        groups.setdefault((r["run_id"], r["study"], r["kind"]), []).append(r)
    out = {}
    for gt in gts:
        frames, summaries = [], []
        for (run_id, study, kind), rs in sorted(groups.items()):
            if kind == "sm":
                items, s = score_sm(rs, data, study, gt)
            elif gt == gts[0]:  # VM has one ground truth; score it once
                items, s = score_vm(rs, data, study, vm_mode)
            else:
                continue
            items.insert(0, "run_id", run_id)
            items.insert(1, "kind", kind)
            items.insert(2, "condition_id", cond_id)
            frames.append(items)
            summaries.append({"run_id": run_id, **s})
        out[gt] = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), summaries)
    return out


def _run_condition_id(run_dir: Path):
    """condition_id from the run header when the predictions sit in a run folder (so stability can group)."""
    from .runlog.writer import read_events
    if not any(run_dir.glob("events.*.jsonl")):
        return None
    header = next((e for e in read_events(run_dir) if e["type"] == "run_header"), {})
    return header.get("condition_id")


def cmd_score(a) -> int:
    data = Data(a.root)
    pred_path = Path(a.predictions)
    out_dir = Path(a.out_dir) if a.out_dir else pred_path.parent
    gts = [a.gt] + [g for g in a.also if g != a.gt]
    for gt, (items, summaries) in score_file(pred_path, data, gts, a.vm_mode).items():
        _write_new(out_dir / f"scores.{gt}.json", json.dumps(
            {"scoring_version": SCORING_VERSION, "predictions": str(pred_path), "summaries": summaries},
            indent=1, default=str) + "\n", a.force)
        _write_new(out_dir / f"items.{gt}.csv", items.to_csv(index=False), a.force)
        for s in summaries:
            if s["kind"] == "sm":
                print(f"[{gt}] {s['run_id']} {s['study']} SM top1={s['top1_correct']}/{s['n_annotated']} "
                      f"MRR={s['mrr']:.3f} closed-world={s['closed_world_correct']}/{s['n_columns']} "
                      f"hallucinated={s['hallucinated']}")
            else:
                cats = {k: f"{v['micro']:.3f}" for k, v in s.items() if isinstance(v, dict) and "micro" in v}
                print(f"[{s['gt']}] {s['run_id']} {s['study']} VM micro by category: {cats}")
    return 0


def _read_items(path) -> pd.DataFrame:
    """An items table. Tables written by the experiment scripts have no condition_id column (only `hbench score`
    adds one), so take it from the run header of the folder the table sits in; "" when there is none."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "condition_id" not in df:
        df["condition_id"] = _run_condition_id(Path(path).parent) or ""
    df["seed_independent"] = _run_seed_independent(Path(path).parent)
    df["run_dir"] = str(Path(path).parent)
    return df


def _run_seed_independent(run_dir: Path):
    """Were the run's sampled calls independent (hbench.seeds.check_seeds)? None without a run log here."""
    from .runlog.writer import read_events
    from .seeds import check_seeds
    if not any(run_dir.glob("events.*.jsonl")):
        return None
    return check_seeds(read_events(run_dir))["independent"]


def cmd_stability(a) -> int:
    from .stability import stability_report

    long = pd.concat([_read_items(p) for p in a.items], ignore_index=True)
    long["correct"] = long["correct"].map(lambda x: None if x == "" else x == "True")
    long["margin"] = pd.to_numeric(long["margin"], errors="coerce") if "margin" in long else None
    cols = ["run_id", "item_id", "pred", "correct", "margin"]
    if "p_path" in long and "greedy_path" in long:  # item tables of runs at T>0 (evidence v2)
        long["p_path"] = pd.to_numeric(long["p_path"], errors="coerce")
        long["greedy_path"] = long["greedy_path"].map({"True": True, "False": False})
        cols += ["p_path", "greedy_path"] + (["prompt_sha256"] if "prompt_sha256" in long else [])
    reference = pd.read_csv(a.reference, dtype=str, keep_default_na=False) if a.reference else None
    report = {}
    for (kind, study, cid), g in long.groupby(["kind", "study", "condition_id"], dropna=False):
        key = f"{kind}:{study}" + (f":{cid}" if cid else "")
        # False if any run's calls shared seeds, or two runs sampled the same prompt with the same seed (copies,
        # not replicates); True only if every run was checked
        flags = set(g["seed_independent"])
        independent = False if False in flags else True if flags == {True} else None
        if independent is not False and g["run_dir"].nunique() > 1:
            from .runlog.writer import read_events
            from .seeds import check_replicates
            across = check_replicates({d: read_events(d) for d in g["run_dir"].unique()
                                       if any(Path(d).glob("events.*.jsonl"))})["independent"]
            independent = False if across is False else independent
        report[key] = stability_report(g[cols], reference=reference, endpoint=a.endpoint,
                                       seed_independent=independent)
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        _write_new(Path(a.out), text + "\n", a.force)
    print(text)
    return 0


def cmd_engine(a) -> int:
    from .engine import ENGINE_LOCK, check, engine_snapshot, lock_from_snapshot

    snap = engine_snapshot(a.models)
    if a.action == "snapshot":
        print(json.dumps(snap, indent=1, default=str))
        return 0
    if a.action == "lock":
        lock = lock_from_snapshot(snap)
        if ENGINE_LOCK.exists():  # recorded from audits, not from the snapshot
            old = json.loads(ENGINE_LOCK.read_text())
            if old.get("sampler_defaults"):
                lock["sampler_defaults"] = old["sampler_defaults"]
        if a.write:
            _write_new(ENGINE_LOCK, json.dumps(lock, indent=1, sort_keys=True) + "\n", a.force)
            print(f"wrote {ENGINE_LOCK}")
        else:
            print(json.dumps(lock, indent=1, sort_keys=True))
        return 0
    drift = check(snap)
    print("\n".join(drift) if drift else "ok: engine matches env/engine-lock.json")
    return 1 if drift else 0


def cmd_audit(a) -> int:
    from .engine import audit_problems, audit_run

    summary = audit_run(a.run_dir)
    print(json.dumps(summary, indent=1))
    probs = audit_problems(summary)
    for x in probs:
        print("problem:", x)
    return 1 if probs else 0


def _parse_set(items) -> dict:
    """--set key=value pairs; values are read as JSON (numbers), else kept as strings."""
    out = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep or not k:
            sys.exit(f"--set takes key=value, not {item!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _parse_seeds(text: str | None) -> list:
    """"1-15" or "1,2,3" (or a mix); [None] when no seeds are given."""
    if not text:
        return [None]
    out = []
    for part in text.split(","):
        lo, sep, hi = part.partition("-")
        out += list(range(int(lo), int(hi) + 1)) if sep else [int(lo)]
    return out


def cmd_replay(a) -> int:
    from .design import make_condition
    from .engine.lineage import _audit_rows
    from .replay import (_TRANSPORT, RUN_PROTOCOL, api_base, client_for, compare, load_calls, replay_call,
                         replay_seeds)
    from .runlog.session import start_run
    from .runlog.writer import read_events

    calls, blobs = load_calls(a.run_dir)
    idx = [int(x) for x in a.calls.split(",")] if a.calls else list(range(len(calls)))
    bad = [i for i in idx if not 0 <= i < len(calls)]
    if bad:
        sys.exit(f"call indices {bad} out of range: the run has {len(calls)} replayable calls")
    if "cold_start" in a.protocol and idx != list(range(len(idx))):
        sys.exit("cold_start replays a prefix of the run: indices must be 0..k (a gap would change the lineage)")
    all_calls = calls
    calls = [calls[i] for i in idx]
    if not calls:
        sys.exit("no replayable llm_call events in that run")
    header = next((e for e in read_events(a.run_dir) if e["type"] == "run_header"), {})
    src_cond = header.get("condition") or {}
    study = (src_cond.get("data") or {}).get("study") or (header.get("task") or {}).get("study") or "unknown"
    model = blobs.get(calls[0]["request_raw"]).get("model")
    overrides, seeds = _parse_set(a.set), _parse_seeds(a.seeds)
    set_seed = overrides.pop("seed", None)  # --set seed=N is a base like --seeds N: never one seed on every call
    if set_seed is not None:
        if seeds != [None]:
            sys.exit("give the seed with --seeds or --set seed, not both")
        seeds = [int(set_seed)]
    if "cold_start" in a.protocol and a.repeat > 1 and seeds != [None]:
        sys.exit("cold_start with --seeds runs one cold replay per seed; drop --repeat")
    rows = []
    for protocol, seed, rep_run in [(p, s, r) for p in a.protocol for s in seeds
                                    for r in (range(a.repeat) if p == "cold_start" else [None])]:
        # each seed, and under cold_start each repeat, is one replicate of the replay condition: its own run, so
        # every cold_start repeat really starts cold. The replicate's seed is a base: each sampled call gets its
        # own seed (hbench.seeds), since llama-server reseeds per request and a shared seed correlates samples
        call_seeds = replay_seeds(all_calls, blobs, seed, overrides)
        src_name = Path(a.run_dir).name
        src_dec = calls[0].get("decoding") or {}
        cond = make_condition(
            method={"name": "replay", "protocol": protocol, "source_run": src_name,
                    **({"overrides": overrides} if overrides else {})},
            model={"name": model},
            decoding={"temperature": overrides.get("temperature", src_dec.get("temperature")),
                      **{k: v for k, v in overrides.items() if k not in ("temperature", "seed")},
                      **({"seed": "per_replicate"} if seed is not None else {})},
            engine={"transport": _TRANSPORT.get(calls[0].get("transport"), calls[0].get("transport")),
                    "protocol": RUN_PROTOCOL.get(protocol, "natural")},
            data={"study": study}, replicate=seed if seed is not None else (rep_run or 0))
        log = start_run(Path(a.out), "replay", cond, task={"kind": "replay", "source_run": str(a.run_dir)},
                        models=[model], ollama_base=api_base(calls[0]))
        done = []
        try:
            for rep in (range(a.repeat) if rep_run is None else [rep_run]):
                for i, ev in zip(idx, calls):
                    cs, base = call_seeds[i]
                    ov = {**overrides, **({"seed": cs} if cs is not None else {})}
                    res = replay_call(ev, blobs, client_for(ev, log), protocol, step=i, overrides=ov or None,
                                      seed_base=base)
                    done.append((rep, i, ev, res.span_id, cs))
        finally:
            log.close()
        by_span = {e.get("span_id"): e for e in read_events(log.run_dir)}
        both = _Blobs(blobs, log.blobs)
        lin_src, lin_rep = _audit_rows(Path(a.run_dir)), _audit_rows(log.run_dir)
        mine = [{"protocol": protocol, "seed": seed, "call_seed": cs, "repeat": rep, "source_call": i, **row}
                for rep, i, ev, span, cs in done
                for row in compare(ev, by_span[span], both, lin_src.get(ev["span_id"]), lin_rep.get(span))]
        rows += mine
        pd.DataFrame(mine).to_csv(log.run_dir / "replay_compare.csv", index=False)
        print(f"[{protocol}{'' if seed is None else f' seed {seed}'}] run {log.run_dir}")
    df = pd.DataFrame(rows)
    # calls without logprobs or without a decision leave these columns empty, so aggregate whatever is there
    cols = [c for c in ("same_text", "same_tokens", "bitwise_same_logprobs", "decision_flipped", "lineage_same")
            if c in df]
    df[cols] = df[cols].astype(float)
    agg = {**{c: "mean" for c in cols},
           **({"max_abs_dlogprob_chosen": "max"} if "max_abs_dlogprob_chosen" in df else {}),
           **({"d_margin": lambda x: x.abs().max()} if "d_margin" in df else {})}
    print(df.groupby("protocol").agg(agg).to_string())
    if "p_path_source" in df:  # resampling: observed decision changes against what the source's logprobs predict
        for protocol, g in df.groupby("protocol"):
            p = pd.to_numeric(g["p_path_source"], errors="coerce")
            print(f"[{protocol}] sampled decisions changed: {int(g['decision_flipped'].sum())} of {len(g)}; "
                  f"expected at most {float((1 - p).sum()):.2f} (sum of 1 - P over {int(p.notna().sum())} with P)")
    return 0


class _Blobs:
    """Read blobs from either store (source run and replay run may live under different roots)."""
    def __init__(self, *stores):
        self.stores = stores

    def get(self, ref):
        for st in self.stores:
            try:
                return st.get(ref)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(ref)


def cmd_lineage(a) -> int:
    from .engine import lineage

    runs = lineage.run_dirs(a.paths)
    if not runs:
        sys.exit("no run folders found")
    if a.action == "rebuild":
        if not a.out_dir:
            sys.exit("lineage rebuild needs --out-dir (the run folders are not modified)")
        out = Path(a.out_dir)
        for rd in runs:
            s = lineage.rebuild(rd, out / f"{rd.name}.lineage.json")
            print(f"{rd.name}: {json.dumps(s)}")
        return 0
    res = lineage.check(runs, a.lineage_dir)
    print(json.dumps(res, indent=1))
    return 1 if res["violations"] else 0


def cmd_llm(a) -> int:
    from .llm.record import rederive
    from .runlog.writer import BlobStore, read_events

    run_dir = Path(a.run_dir)
    blobs = BlobStore(run_dir.parent / "_blobs")
    rows = [rederive(e, blobs) for e in read_events(run_dir) if e["type"] == "llm_call"]
    out = run_dir / "llm_calls.rederived.jsonl"
    _write_new(out, "".join(json.dumps(r, default=str) + "\n" for r in rows), a.force)
    print(f"wrote {len(rows)} rederived calls to {out}")
    return 0


def cmd_log(a) -> int:
    from .runlog.check import check_run

    res = check_run(a.run_dir)
    print(json.dumps(res, indent=1))
    return 1 if res["problems"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hbench")
    ap.add_argument("--root", default=None, help="data root (default: $HBENCH_DATA or <repo>/datasets)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="build or verify datasets/MANIFEST.json")
    m.add_argument("action", choices=["build", "verify"])
    m.add_argument("--manifest", default=None)
    m.add_argument("--force", action="store_true")
    m.set_defaults(func=cmd_manifest)

    p = sub.add_parser("preds", help="validate or convert prediction files")
    p.add_argument("action", choices=["validate", "convert"])
    p.add_argument("input")
    p.add_argument("--format", choices=["clark-match", "magneto-txt", "bdikit-sm-csv", "bdikit-vm-csv"])
    p.add_argument("--run-id", default=None)
    p.add_argument("--study", default="clark")
    p.add_argument("--output-type", default="top1", choices=["top1", "ranked"])
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_preds)

    s = sub.add_parser("score", help="score a predictions.jsonl")
    s.add_argument("predictions")
    s.add_argument("--gt", default="gdc-sm", choices=sorted(SM_GT_VARIANTS))
    s.add_argument("--also", nargs="*", default=[], choices=sorted(SM_GT_VARIANTS))
    s.add_argument("--vm-mode", default="oracle", choices=["oracle", "end_to_end"])
    s.add_argument("--out-dir", default=None)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_score)

    t = sub.add_parser("stability", help="RQ1 stability metrics from several runs' items CSVs")
    t.add_argument("items", nargs="+")
    t.add_argument("--reference", default=None, help="items CSV of a T=0 run of the same prompts, for the sampling check")
    t.add_argument("--endpoint", default=None, choices=["/api/chat", "/v1/chat/completions"],
                   help="whose default sampler the runs used (with --reference)")
    t.add_argument("--out", default=None)
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_stability)

    e = sub.add_parser("engine", help="engine snapshot, lock file, drift check")
    e.add_argument("action", choices=["snapshot", "lock", "check"])
    e.add_argument("--models", nargs="+", default=["qwen2.5:14b"])
    e.add_argument("--write", action="store_true", help="lock: write env/engine-lock.json")
    e.add_argument("--force", action="store_true")
    e.set_defaults(func=cmd_engine)

    au = sub.add_parser("audit", help="join a run's LLM calls with the Ollama server log")
    au.add_argument("run_dir")
    au.set_defaults(func=cmd_audit)

    rp = sub.add_parser("replay", help="replay a run's LLM calls under engine protocols and compare")
    rp.add_argument("run_dir")
    rp.add_argument("--protocol", nargs="+", default=["as_is"],
                    choices=["as_is", "cold_start", "cold", "resend", "after_unrelated"])
    rp.add_argument("--calls", default=None, help="comma-separated call indices (default: all)")
    rp.add_argument("--repeat", type=int, default=1)
    rp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a sampler setting (temperature, seed, top_p, top_k, min_p on /api/chat)")
    rp.add_argument("--seeds", default=None, help="e.g. 1-15: one replay run per seed, replicates of one condition")
    rp.add_argument("--out", required=True, help="raw-results root for the replay runs")
    rp.set_defaults(func=cmd_replay)

    ln = sub.add_parser("lineage", help="check that T=0 calls with the same input and lineage are bitwise identical, "
                                        "or rebuild the lineage of runs audited before E0 v3")
    ln.add_argument("action", choices=["check", "rebuild"])
    ln.add_argument("paths", nargs="+", help="run folders or raw-results roots")
    ln.add_argument("--lineage-dir", default=None, help="check: read <run>.lineage.json files written by rebuild")
    ln.add_argument("--out-dir", default=None, help="rebuild: where to write <run>.lineage.json")
    ln.set_defaults(func=cmd_lineage)

    ll = sub.add_parser("llm", help="re-derive normalized llm_call fields from stored raw payloads")
    ll.add_argument("action", choices=["rederive"])
    ll.add_argument("run_dir")
    ll.add_argument("--force", action="store_true")
    ll.set_defaults(func=cmd_llm)

    lg = sub.add_parser("log", help="check a run directory")
    lg.add_argument("action", choices=["validate"])
    lg.add_argument("run_dir")
    lg.set_defaults(func=cmd_log)

    a = ap.parse_args(argv)
    if a.cmd == "preds" and a.action == "convert" and not (a.format and a.out):
        ap.error("preds convert needs --format and --out")
    return a.func(a)
