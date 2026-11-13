"""E4 run plans for E1's run_plan.py, plus the checks and the morning summary a night needs.

    python -s make_plan.py night N1 --out n1.json [--studies clark,mcdermott,huang]
    python -s make_plan.py cells clark:Z1 clark:Z2 clark:D:1:2 clark:T:3 clark:A:5 clark:P-row:1 --out x.json
    python -s make_plan.py prepare n1.json      # create the shared T=0 tool cache a plan uses (idempotent)
    python -s make_plan.py compare --root R clark:Z1 clark:Z2   # check step: bitwise equal runs, or exit 1
    python -s make_plan.py summary --root R     # morning summary: summary.md and summary.json under R

A plan is engine_check, one `cmd` step per run of experiments/e3_agent/run_agent.py, and engine_check again.
When a plan holds both Z1 and Z2 of a study, a compare step follows Z2 and stops the plan if they differ.

Every run gets the same absolute --raw-dir: the staged input path <raw-dir>/_inputs/e3/<study>.csv is part of the
task prompt, so another raw dir would change the prompt, every per-call seed and the tokens, and break the seed
grid. Never point it under the run_plan root.

Cells are study:arm[:args]:
    Z1, Z2, Z     T=0 on both, seed 1. Z1 fills the shared T=0 tool cache (--bdikit-cache shared); Z2 and Z use
                  their own cache. Z2 must equal Z1 bit for bit.
    D:a:t         agent and tools sampled (--sample-t / --sample-top-p), agent seed a, tool seed t
                  (--tool-seed only off the diagonal, so a diagonal cell is the same run as before the flag)
    T:k           agent T=0, tools sampled, --seed k
    A:k           agent sampled, tools T=0, --seed k, reading and extending the shared T=0 tool cache
    P-hash:h      as Z, --hashseed random (h only labels the repeat)
    P-prompt      as Z, --prompt-variant no_json_format
    P-row:r       as Z, --row-order r
The study list and the sampling of the sampled arms are inputs; the nights are doc appendix A with the studies
filled in by role.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))  # this checkout's hbench
MODEL = "qwen3.8-27b-bdikit-nb1024"
RAW_DIR = Path.home() / "hbench_raw"
T0_CACHE = RAW_DIR / "e4_t0cache"
SHARED = Path.home() / ".cache" / "bdikit"  # models/ and embeddings/ (deterministic, shared by every run)
MAGNETO_BASE_KEYS = ("mpnet", "roberta", "e5", "arctic", "minilm")  # hbench.runlog.mcp_logged
PLAN_VERSION = 1
DEFAULT_STUDIES = ("clark", "mcdermott", "huang")


# ------------------------------------------------------------------ cells

def parse_cell(spec: str) -> dict:
    study, arm, *args = spec.split(":")
    ints = [int(x) for x in args]
    if arm in ("Z", "Z1", "Z2", "P-prompt") and not ints:
        return {"study": study, "arm": arm}
    if arm == "D" and len(ints) == 2:
        return {"study": study, "arm": arm, "a": ints[0], "t": ints[1]}
    if arm in ("T", "A", "P-hash", "P-row") and len(ints) == 1:
        return {"study": study, "arm": arm, "k": ints[0]}
    raise ValueError(f"bad cell {spec!r} (see the module docstring)")


def cell_label(c: dict) -> str:
    arm = c["arm"]
    if arm == "D":
        return f"D{c['a']}x{c['t']}"
    if arm in ("T", "A", "P-hash", "P-row"):
        return f"{arm}{c['k']}"
    return arm


def run_step(c: dict, raw_dir: Path, cache: Path, sample_t: float, sample_p: float, model: str,
             turn_timeout: int) -> dict:
    """The run_plan cmd step of one cell."""
    arm = c["arm"]
    greedy, sampled = ("0", "1"), (repr(sample_t), repr(sample_p))
    extra: list[str] = []
    if arm in ("Z", "Z1", "Z2") or arm.startswith("P-"):
        agent, tools, seed, block, rep = greedy, greedy, 1, "Z" if arm.startswith("Z") else arm, 1
        if arm == "Z2":
            rep = 2
        if arm == "Z1":
            extra += ["--bdikit-cache", "shared", "--bdikit-shared", str(cache)]
        elif arm == "P-hash":
            extra += ["--hashseed", "random"]
            rep = c["k"]
        elif arm == "P-prompt":
            extra += ["--prompt-variant", "no_json_format"]
        elif arm == "P-row":
            extra += ["--row-order", str(c["k"])]
            rep = c["k"]
    elif arm == "D":
        agent, tools, seed, block, rep = sampled, sampled, c["a"], "D", 10 * c["a"] + c["t"]
        if c["t"] != c["a"]:
            extra += ["--tool-seed", str(c["t"])]
    elif arm == "T":
        agent, tools, seed, block, rep = greedy, sampled, c["k"], "T", c["k"]
    elif arm == "A":
        agent, tools, seed, block, rep = sampled, greedy, c["k"], "A", c["k"]
        extra += ["--bdikit-cache", "shared", "--bdikit-shared", str(cache)]
    else:
        raise ValueError(arm)
    argv = ["{python}", "-s", "{repo}/experiments/e3_agent/run_agent.py", "--study", c["study"], "--task", "sm",
            "--model", model, "--temperature", agent[0], "--top-p", agent[1], "--seed", str(seed),
            "--tool-temperature", tools[0], "--tool-top-p", tools[1], "--engine-protocol", "cold_start",
            "--turn-timeout", str(turn_timeout), "--block", block, "--replicate", str(rep),
            "--raw-dir", str(raw_dir), *extra]
    return {"name": f"{c['study']}:{cell_label(c)}", "kind": "cmd", "cell": c, "argv": argv}


def compare_step(study: str) -> dict:
    return {"name": f"check:{study}:Z1=Z2", "kind": "cmd", "stop_on_fail": True,
            "argv": ["{python}", "-s", "{repo}/experiments/e4_agent_repeats/make_plan.py", "compare", "--root", "{root}",
                     f"{study}:Z1", f"{study}:Z2"]}


def make_plan(cells: list[dict], name: str, raw_dir: Path = RAW_DIR, cache: Path = T0_CACHE,
              sample_t: float = 1.0, sample_p: float = 1.0, model: str = MODEL, turn_timeout: int = 10800) -> dict:
    raw_dir, cache = Path(raw_dir).expanduser().resolve(), Path(cache).expanduser().resolve()
    bad = [k for k in MAGNETO_BASE_KEYS if k in str(cache)]
    if bad:
        raise ValueError(f"cache dir {cache} contains {bad}: Magneto picks its base model by path substring")
    steps = [{"name": "engine_check_before", "kind": "engine_check", "model": model}]
    for c in cells:
        steps.append(run_step(c, raw_dir, cache, sample_t, sample_p, model, turn_timeout))
        if c["arm"] == "Z2" and any(x["study"] == c["study"] and x["arm"] == "Z1" for x in cells):
            steps.append(compare_step(c["study"]))
    steps.append({"name": "engine_check_after", "kind": "engine_check", "model": model})
    names = [s["name"] for s in steps]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate cells: {dup}")
    uses_cache = any(c["arm"] in ("Z1", "A") for c in cells)
    return {"plan_version": PLAN_VERSION, "name": name, "model": model, "raw_dir": str(raw_dir),
            "t0_cache": str(cache) if uses_cache else None, "sampling": {"temperature": sample_t, "top_p": sample_p},
            "turn_timeout": turn_timeout, "cells": cells, "steps": steps}


# ------------------------------------------------------------------ nights (doc appendix A)

def _transversal(k: int) -> list[tuple[int, int]]:
    """k=0 is the diagonal; transversal k holds the 4 cells (a, (a-1+k) mod 4 + 1): distinct rows and columns."""
    return [(a, (a - 1 + k) % 4 + 1) for a in range(1, 5)]


def night_cells(night: str, studies=DEFAULT_STUDIES) -> list[str]:
    s1, s2, s3 = studies

    def D(s, cells):
        return [f"{s}:D:{a}:{t}" for a, t in cells]
    nights = {
        "N1": [f"{s1}:Z1", f"{s1}:Z2", f"{s2}:Z", f"{s3}:Z"]
              + [c for k in (1, 2) for s in (s1, s2, s3) for c in D(s, [(k, k)])]
              + [c for k in (3, 4) for s in (s1, s2) for c in D(s, [(k, k)])],
        "N2": D(s1, _transversal(1)) + D(s3, [(3, 3)]) + D(s1, _transversal(2)) + D(s3, [(4, 4)])
              + D(s1, _transversal(3)) + [f"{s1}:T:{k}" for k in (1, 2)],
        "N3": D(s2, _transversal(1)) + [f"{s1}:A:{k}" for k in (1, 2, 3, 4)] + D(s2, _transversal(2))
              + [f"{s1}:A:{k}" for k in (5, 6, 7, 8)] + D(s2, _transversal(3)),
        "N4": D(s3, _transversal(1)) + [f"{s1}:T:3"] + D(s3, _transversal(2)) + [f"{s1}:T:{k}" for k in (4, 5)],
        "N5": D(s3, _transversal(3)) + [f"{s1}:T:{k}" for k in (6, 7, 8)]
              + [f"{s1}:P-hash:{h}" for h in (1, 2)] + [f"{s1}:P-prompt"] + [f"{s1}:P-row:{r}" for r in (1, 2, 3)],
    }
    if night not in nights:
        raise ValueError(f"unknown night {night!r}; one of {sorted(nights)}")
    return nights[night]


# ------------------------------------------------------------------ night helpers

def check_data(plan: dict) -> str:
    """Every study of the plan has its source CSV and SM ground truth under the data root (HBENCH_DATA, else the
    checkout's datasets/, which a pinned run checkout does not have). Raises SystemExit otherwise."""
    from hbench.data import Data, default_root
    root, data = default_root(), Data()
    missing = []
    for study in sorted({c["study"] for c in plan["cells"]}):
        try:
            if not data.source_path(study).exists():
                raise FileNotFoundError(data.source_path(study))
            data.sm_gold(study, "gdc-sm")
        except Exception as e:  # noqa: BLE001  (any failure to read a study's files)
            missing.append(f"{study}: {type(e).__name__}: {e}")
    if missing:
        raise SystemExit(f"data root {root} is missing files (set HBENCH_DATA to the main checkout's datasets/):\n  "
                         + "\n  ".join(missing))
    return f"data root {root}: ok for {sorted({c['study'] for c in plan['cells']})}"


def prepare(plan: dict) -> str:
    """Check the data root, then create the shared T=0 tool cache (models/ and embeddings/ linked to the shared
    ones, an empty runs/)."""
    msg = check_data(plan) if plan.get("cells") else "no runs in this plan"
    if not plan.get("t0_cache"):
        return msg + "; no shared T=0 cache in this plan"
    cache = Path(plan["t0_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "runs").mkdir(exist_ok=True)
    for sub in ("models", "embeddings"):
        link, target = cache / sub, SHARED / sub
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise SystemExit(f"{link} points to {link.resolve()}, not {target}")
        elif link.exists():
            raise SystemExit(f"{link} exists and is not a link to {target}")
        else:
            link.symlink_to(target, target_is_directory=True)
    n = len(list((cache / "runs").iterdir()))
    return msg + f"; {cache}: ready, {n} cached results in runs/" + (
        " (Z1 must start from an empty runs/ to be a real check)" if n and any(
            c["arm"] == "Z1" for c in plan["cells"]) else "")


def executions(root: Path) -> dict:
    """The last execution.jsonl record of every step."""
    out = {}
    f = root / "execution.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            out[rec["name"]] = rec
    return out


def run_dir_of(rec: dict | None) -> Path | None:
    """run_agent.py prints its run dir as the last line of its output."""
    if not rec:
        return None
    lines = [x for x in (rec.get("out") or "").splitlines() if x.strip()]
    if lines:
        p = Path(lines[-1].strip())
        if (p / "events.driver.jsonl").exists():
            return p
    return None


def _events(rd: Path, proc: str) -> list[dict]:
    return [json.loads(x) for x in (rd / f"events.{proc}.jsonl").read_text().splitlines() if x.strip()]


def call_keys(rd: Path) -> dict:
    keys = ("caller", "request_sha256", "output_sha256", "logprobs_sha256")
    return {proc: [tuple([e.get(k) for k in keys] + [(e.get("decoding") or {}).get("seed")])
                   for e in _events(rd, proc) if e["type"] == "llm_call"] for proc in ("driver", "mcp")}


def compare_runs(a: Path, b: Path) -> dict:
    """Per-call request, seed, output and logprobs hashes of two runs, and their predictions."""
    ka, kb = call_keys(a), call_keys(b)
    res = {"runs": [str(a), str(b)], "equal": True}
    for proc in ("driver", "mcp"):
        x, y = ka[proc], kb[proc]
        first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
        res[proc] = {"n": [len(x), len(y)], "first_difference": first}
        if len(x) != len(y) or first is not None:
            res["equal"] = False
    strip = ("run_id", "ts", "ts_ns", "pid", "seq")
    preds = [[{k: v for k, v in json.loads(line).items() if k not in strip}
              for line in (d / "predictions.jsonl").read_text().splitlines() if line.strip()] for d in (a, b)]
    res["predictions_equal"] = preds[0] == preds[1]
    res["equal"] = res["equal"] and res["predictions_equal"]
    return res


def validate(rd: Path) -> list:
    p = subprocess.run([sys.executable, "-s", "-m", "hbench", "log", "validate", str(rd)], cwd=REPO,
                       capture_output=True, text=True)
    try:
        return json.loads(p.stdout).get("problems") or []
    except json.JSONDecodeError:
        return [f"unreadable validate output (rc {p.returncode}): {(p.stdout + p.stderr)[-300:]}"]


def summarize(root: Path, run_validate: bool = True) -> dict:
    plan = json.loads((root / "plan.json").read_text())["plan"]
    done = executions(root)
    rows = []
    for step in plan["steps"]:
        rec = done.get(step["name"])
        row = {"name": step["name"], "kind": step["kind"], "status": rec["status"] if rec else "not run",
               "secs": rec.get("secs") if rec else None}
        rd = run_dir_of(rec) if step.get("cell") else None
        if step.get("cell") and rec:
            row["run_dir"] = str(rd) if rd else None
            if rd:
                end = next((e for e in _events(rd, "driver") if e["type"] == "run_end"), {})
                audit = rd / "engine_audit.json"
                summ = json.loads(audit.read_text())["summary"] if audit.exists() else {}
                row.update(outcome=end.get("outcome"), run_status=end.get("status"), followups=end.get("followups"),
                           wall_min=round(end["wall_s"] / 60, 1) if end.get("wall_s") else None,
                           sm=[(s.get("top1_correct"), s.get("n_annotated")) for s in end.get("scores") or []
                               if s.get("kind") == "sm"],
                           audit=f"{summ.get('matched')}/{summ.get('n_calls')} matched, "
                                 f"{summ.get('foreign_tasks')} foreign" if summ else "no audit")
                if run_validate:
                    probs = validate(rd)
                    row["validate"] = "ok" if not probs else f"{len(probs)} problems: {str(probs)[:300]}"
            else:
                row["error"] = (rec.get("err") or rec.get("error") or "")[-300:]
        elif rec and rec["status"] != "ok":
            row["error"] = (rec.get("out") or rec.get("err") or rec.get("error") or "")[-300:].strip()
        rows.append(row)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = {"plan": plan["name"], "root": str(root), "counts": counts, "steps": rows}
    (root / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [f"# {plan['name']}: {counts}", "", "| step | status | min | outcome | SM | audit | validate |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        mins = r.get("wall_min") or (round(r["secs"] / 60, 1) if r.get("secs") else "")
        sm = " ".join(f"{a}/{b}" for a, b in r.get("sm") or [])
        lines.append(f"| {r['name']} | {r['status']} | {mins} | {r.get('outcome') or ''} | {sm} | "
                     f"{r.get('audit') or ''} | {r.get('validate') or r.get('error') or ''} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return out


# ------------------------------------------------------------------ cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("night", "cells"):
        p = sub.add_parser(name)
        if name == "night":
            p.add_argument("night", help="N1..N5 (doc appendix A)")
            p.add_argument("--studies", default=",".join(DEFAULT_STUDIES), help="the three studies, by role")
        else:
            p.add_argument("cells", nargs="+", help="study:arm[:args]")
            p.add_argument("--name", default="e4_cells")
        p.add_argument("--out", required=True)
        p.add_argument("--raw-dir", default=str(RAW_DIR), help="the same for every E4 run (it is in the prompt)")
        p.add_argument("--t0-cache", default=str(T0_CACHE), help="shared T=0 tool cache of Z1 and the A arm")
        p.add_argument("--sample-t", type=float, default=1.0, help="temperature of the sampled side")
        p.add_argument("--sample-top-p", type=float, default=1.0, help="top_p of the sampled side")
        p.add_argument("--model", default=MODEL)
        p.add_argument("--turn-timeout", type=int, default=10800)
    p = sub.add_parser("prepare")
    p.add_argument("plan_file")
    p = sub.add_parser("compare")
    p.add_argument("--root", required=True)
    p.add_argument("steps", nargs=2, help="two step names of the plan run under --root")
    p = sub.add_parser("summary")
    p.add_argument("--root", required=True)
    p.add_argument("--no-validate", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd in ("night", "cells"):
        if a.cmd == "night":
            studies = tuple(a.studies.split(","))
            if len(studies) != 3:
                ap.error("--studies takes three studies (roles 1, 2, 3 of appendix A)")
            specs, name = night_cells(a.night, studies), f"e4_{a.night}"
        else:
            specs, name = a.cells, a.name
        plan = make_plan([parse_cell(s) for s in specs], name, a.raw_dir, a.t0_cache, a.sample_t, a.sample_top_p,
                         a.model, a.turn_timeout)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
        n_runs = sum(1 for s in plan["steps"] if s.get("cell"))
        print(f"{a.out}: {len(plan['steps'])} steps, {n_runs} runs")
        return 0
    if a.cmd == "prepare":
        print(prepare(json.loads(Path(a.plan_file).read_text())))
        return 0
    if a.cmd == "compare":
        done = executions(Path(a.root))
        dirs = [run_dir_of(done.get(s)) for s in a.steps]
        if None in dirs:
            print(json.dumps({"equal": False, "error": f"no run dir for {[s for s, d in zip(a.steps, dirs) if not d]}"}))
            return 1
        res = compare_runs(*dirs)
        print(json.dumps(res))
        return 0 if res["equal"] else 1
    out = summarize(Path(a.root), run_validate=not a.no_validate)
    print((Path(a.root) / "summary.md").read_text())
    return 0 if all(r["status"] == "ok" for r in out["steps"]) else 1


if __name__ == "__main__":
    sys.exit(main())
