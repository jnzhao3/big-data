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
            frames.append(items)
            summaries.append({"run_id": run_id, **s})
        out[gt] = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), summaries)
    return out


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


def cmd_stability(a) -> int:
    from .stability import stability_report

    long = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in a.items], ignore_index=True)
    long["correct"] = long["correct"].map(lambda x: None if x == "" else x == "True")
    report = {}
    for (kind, study), g in long.groupby(["kind", "study"]):
        report[f"{kind}:{study}"] = stability_report(g[["run_id", "item_id", "pred", "correct"]])
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        _write_new(Path(a.out), text + "\n", a.force)
    print(text)
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
    t.add_argument("--out", default=None)
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_stability)

    lg = sub.add_parser("log", help="check a run directory")
    lg.add_argument("action", choices=["validate"])
    lg.add_argument("run_dir")
    lg.set_defaults(func=cmd_log)

    a = ap.parse_args(argv)
    if a.cmd == "preds" and a.action == "convert" and not (a.format and a.out):
        ap.error("preds convert needs --format and --out")
    return a.func(a)
