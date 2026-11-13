"""E1 Stage 1 driver (single-turn part); see README.md.

  1a  margin survey: clark_match full + shortlist on 10 studies, qwen2.5:14b, T=0, seed 1, natural protocol
  1b  two-lineage enumeration: hbench replay --protocol resend on every call with decision margin < 0.5 (or no
      margin), plus a seeded 5% sample of the rest
  1c  transport: clark shortlist with --engine-protocol cold through /api/chat and through /v1

Engine check before and after; every run folder is validated at the end. The back-to-back follow-up (1e) is
the stage1e plan of run_plan.py.

Usage: python -I stage1.py <out_root>      (outside the repo: raw logs carry Qwen's Chinese candidate tokens)
       [--model M] [--studies clark,krug] [--modes full,shortlist] [--think] [--skip-1c]

The defaults are the Stage 1 run of 2026-11-03 (qwen2.5:14b, ten studies, both modes, no thinking). The qwen3.8
27b pilot passes --model qwen3.8-27b-bdikit --studies clark --think --skip-1c.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

MODEL = "qwen2.5:14b"
STUDIES = ["cao", "clark", "dou", "gillette", "huang", "krug", "mcdermott", "satpathy", "vasaikar", "wang"]
_ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
_ap.add_argument("out_root")
_ap.add_argument("--model", default=MODEL)
_ap.add_argument("--studies", default=",".join(STUDIES))
_ap.add_argument("--modes", default="full,shortlist", help="1a modes, in run order (full first keeps its prefix cached)")
_ap.add_argument("--think", action="store_true", help="let a thinking model think (clark_match --think)")
_ap.add_argument("--skip-1c", action="store_true", help="skip the transport comparison")
ARGS = _ap.parse_args()
REPO, OUT = Path(__file__).resolve().parents[2], Path(ARGS.out_root).resolve()
PY = sys.executable
CLARK = str(REPO / "experiments/qwen_local/clark_match.py")
MODEL, STUDIES, THINK = ARGS.model, ARGS.studies.split(","), ["--think"] if ARGS.think else []
MARGIN_MAX = 0.5
SAMPLE_REST = 0.05
LOG = OUT / "driver_log.jsonl"
sys.path.insert(0, str(REPO))
from hbench.runlog.writer import read_events  # noqa: E402


def step(name, cmd, cwd=REPO):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    rec = {"step": name, "cmd": cmd, "rc": p.returncode, "secs": round(time.time() - t0, 1),
           "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-3000:]}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] {name}: rc={p.returncode} {rec['secs']}s", flush=True)
    if p.returncode != 0:
        print(p.stderr[-1500:], flush=True)
    return p


def run_dirs(root: Path):
    return sorted(p.parent for p in root.glob("**/events.driver.jsonl"))


def select_calls(run_dir: Path) -> list[int]:
    """Indices (in hbench replay's order: llm_call events with a raw request) of calls whose decision margin is
    below MARGIN_MAX or unknown, plus a seeded SAMPLE_REST share of the rest."""
    calls = [e for e in read_events(run_dir) if e["type"] == "llm_call" and e.get("request_raw")]
    idx_of = {e["span_id"]: i for i, e in enumerate(calls)}
    preds = [json.loads(line) for line in open(run_dir / "predictions.jsonl")]
    low, rest = [], []
    for p in preds:
        ev = p.get("evidence") or {}
        i = idx_of.get(ev.get("span_id"))
        if i is None:
            continue
        (low if ev.get("margin") is None or ev["margin"] < MARGIN_MAX else rest).append(i)
    rng = random.Random(f"e1-stage1-{run_dir.name}")
    k = round(SAMPLE_REST * len(rest))
    return sorted(low + (rng.sample(rest, k) if k else []))


if __name__ == "__main__":
    if OUT == REPO or REPO in OUT.parents:
        sys.exit("out_root must be outside the repo")
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    p = step("engine_check_before", [PY, "-m", "hbench", "engine", "check", "--models", MODEL])
    if p.returncode != 0:
        sys.exit("engine drift before Stage 1; not running")

    # 1a: all full-mode runs first (shared prefix stays cached), then shortlist
    for mode in ARGS.modes.split(","):
        for s in STUDIES:
            step(f"1a:{mode}:{s}", [PY, "-I", CLARK, "--model", MODEL, "--mode", mode, "--study", s, "--run", "1",
                                    "--temperature", "0", "--seed", "1", *THINK, "--out-dir", str(OUT / "1a"),
                                    "--raw-dir", str(OUT / "1a")])

    # 1b: resend replays of fragile calls
    selection = {}
    for rd in run_dirs(OUT / "1a"):
        if not (rd / "predictions.jsonl").exists():
            continue
        idx = select_calls(rd)
        selection[rd.name] = idx
        if idx:
            step(f"1b:{rd.name}", [PY, "-m", "hbench", "replay", str(rd), "--calls", ",".join(map(str, idx)),
                                   "--protocol", "resend", "--out", str(OUT / "1b")])
    (OUT / "1b_selection.json").write_text(json.dumps(selection, indent=1))

    # 1c: cold-vs-cold through both transports
    for run, transport in () if ARGS.skip_1c else (("2", "ollama"), ("3", "openai_v1")):
        step(f"1c:{transport}", [PY, "-I", CLARK, "--model", MODEL, "--mode", "shortlist", "--study", "clark",
                                 "--run", run, "--temperature", "0", "--seed", "1", *THINK, "--transport", transport,
                                 "--engine-protocol", "cold", "--out-dir", str(OUT / "1c"), "--raw-dir", str(OUT / "1c")])

    step("engine_check_after", [PY, "-m", "hbench", "engine", "check", "--models", MODEL])
    problems = {}
    for rd in run_dirs(OUT):
        p = subprocess.run([PY, "-m", "hbench", "log", "validate", str(rd)], cwd=REPO, capture_output=True, text=True)
        try:
            res = json.loads(p.stdout)
        except json.JSONDecodeError:
            res = {"problems": [f"validate output unreadable: {p.stdout[-300:]} {p.stderr[-300:]}"]}
        if res.get("problems"):
            problems[str(rd.relative_to(OUT))] = res["problems"]
    (OUT / "validate_summary.json").write_text(json.dumps(
        {"n_runs": len(run_dirs(OUT)), "runs_with_problems": problems}, indent=1))
    print(f"validated {len(run_dirs(OUT))} runs, {len(problems)} with problems; total "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
