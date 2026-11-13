"""E1 run-plan executor: write a plan of steps, execute it in order, record what was planned and what ran.

Execution order is itself an experimental factor for the engine: which computation lineage a call gets depends on
the prompt cache, i.e. on what ran before (back-to-back repeats move to a single-token lineage; an unload resets
it). So the order is designed and written down before anything runs, and the record says exactly which runs each
step produced, when, and how it ended.

    python -I run_plan.py make stage2 --out stage2.json [--columns clark:10,krug:10] [--repeats 3]
    python -I run_plan.py make stage1e --out stage1e.json
    python -I run_plan.py make stage3 --out stage3.json --stage1 <Stage 1 results> [--seeds 12,10,3,6]
    python -I run_plan.py run stage2.json --root <out root>      # outside the repo: raw logs contain Chinese tokens
    python -I run_plan.py run stage2.json --root <out root> --resume   # skip steps that already finished ok

The root receives plan.json (the plan as run, with its sha256), execution.jsonl (one record per step: kind,
times, status, the run folders it created, a tail of its output) and the run folders under raw/. A failed
engine_check or unload stops the plan, and so does a failed step marked "stop_on_fail": true (a check step).
Step kinds:

    engine_check   hbench engine check; also refuses to go on when another model is loaded (a foreign client)
    unload         unload the model: the next call starts from an empty slot and RAM prompt cache
    tool_loop      one tool_loop.py episode, in process
    cmd            a command (argv with {python} {repo} {root} {here} placeholders), run from the repo root
    replay         hbench replay of the run an earlier step produced, under the given protocols
    select         write a Stage 3 resampling selection (resample.py select_chat / select_v1) under the root
    resample       one resampling pass (resample.py run_pass) over a selection or its P6 subset
    validate       hbench log validate on every run under the root (validate_summary.json)
    lineage_check  hbench lineage check on every run under the root (lineage_check.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from hbench.data import Data, default_root  # noqa: E402

MODEL = "qwen2.5:14b"
PLAN_VERSION = 1
PY = sys.executable


# ------------------------------------------------------------------ plans

def pick_columns(spec: str, data: Data) -> list[tuple[str, str]]:
    """"clark:10,krug:10" -> the first 10 source columns of each study, in table order; "clark:Case_ID+Age" ->
    those columns (they must exist)."""
    out = []
    for part in spec.split(","):
        study, _, n = part.partition(":")
        cols = data.source_columns(study)
        if n and not n.isdigit():
            names = n.split("+")
            missing = [c for c in names if c not in cols]
            if missing:
                raise ValueError(f"not source columns of {study}: {missing}")
            out += [(study, c) for c in names]
        else:
            out += [(study, c) for c in (cols[: int(n)] if n else cols)]
    return out


def episode(name, study, column, replicate, block, protocol, **sampling) -> dict:
    """A tool_loop step; sampling: transport, temperature, seed, model, think (tool_loop.py's defaults otherwise)."""
    return {"name": name, "kind": "tool_loop", "study": study, "column": column, "replicate": replicate,
            "block": block, "protocol": protocol, "raw": f"raw/{block}", **sampling}


STAGE2_BLOCKS = ("unload_each", "back_to_back", "interleaved")


def make_stage2(columns: list[tuple[str, str]], repeats: int = 3, model: str = MODEL, think: bool | None = None,
                blocks=STAGE2_BLOCKS, single_turn: bool = True) -> dict:
    """E1 Stage 2 (design doc section 5):
      2a  each column's episode repeated `repeats` times in three orders, each block starting from an unload:
          unload_each (every episode starts cold), back_to_back (A A A B B B ...), interleaved (A B ... A B ...)
      2b  replay fidelity: every unload_each episode of replicate 1 replayed per call cold, and whole from cold
      st  single-turn check: clark shortlist three times back to back (is run 3 equal to run 2?)
    Margins of the tool decisions (2c) come from the 2a runs themselves. A model other than the default is
    passed to every episode and clark run; think (None: the model's default) to both as well. The 27b pilot
    keeps two blocks and drops the single-turn check (the Stage 1 driver covers single turns)."""
    sampling = {**({"model": model} if model != MODEL else {}), **({"think": think} if think is not None else {})}
    steps = [{"name": "engine_check_before", "kind": "engine_check", "model": model}]
    all_blocks = {
        "unload_each": [(s, c, r) for s, c in columns for r in range(1, repeats + 1)],
        "back_to_back": [(s, c, r) for s, c in columns for r in range(1, repeats + 1)],
        "interleaved": [(s, c, r) for r in range(1, repeats + 1) for s, c in columns],
    }
    for block in blocks:
        seq = all_blocks[block]
        steps.append({"name": f"{block}:unload", "kind": "unload", "model": model})
        proto = "cold_start" if block == "unload_each" else "natural"
        steps += [episode(f"{block}:{s}:{c}:r{r}", s, c, r, block, proto, **sampling) for s, c, r in seq]
    if single_turn:
        steps.append({"name": "single_turn:unload", "kind": "unload", "model": model})
        steps += [clark_run(f"single_turn:clark_shortlist:r{run}", "shortlist", run, "raw/single_turn", model,
                            think=bool(think)) for run in (6, 7, 8)]
    for s, c in columns:
        steps.append({"name": f"replay:{s}:{c}", "kind": "replay", "of": f"unload_each:{s}:{c}:r1",
                      "protocols": ["cold", "cold_start"], "raw": "raw/replay"})
    steps += [{"name": "engine_check_after", "kind": "engine_check", "model": model},
              {"name": "validate", "kind": "validate"}, {"name": "lineage_check", "kind": "lineage_check"}]
    return {"plan_version": PLAN_VERSION, "name": "e1_stage2", "model": model, "think": think, "columns": columns,
            "repeats": repeats, "blocks": list(blocks), "steps": steps}


def clark_run(name: str, mode: str, run: int, raw: str, model: str = MODEL, think: bool = False) -> dict:
    return {"name": name, "kind": "cmd", "raw": raw,
            "argv": ["{python}", "-I", "{repo}/experiments/qwen_local/clark_match.py", "--model", model, "--mode", mode,
                     "--study", "clark", "--run", str(run), "--temperature", "0", "--seed", "1",
                     *(["--think"] if think else []), "--out-dir", f"{{root}}/{raw}_csv", "--raw-dir", f"{{root}}/{raw}"]}


def make_stage1e(columns=None, repeats: int = 2, model: str = MODEL) -> dict:
    """E1 Stage 1e: clark shortlist, then clark full, each run `repeats` times back to back from an unload
    (natural protocol, T=0). Short dissimilar prompts move every call of a repeat onto the single-token lineage
    through host-RAM prompt-cache restores; the long shared prefix of full mode does not."""
    steps = [{"name": "engine_check_before", "kind": "engine_check", "model": model},
             {"name": "unload", "kind": "unload", "model": model}]
    steps += [clark_run(f"{mode}:r{run}", mode, run, "raw/1e", model)
              for mode in ("shortlist", "full") for run in range(4, 4 + repeats)]
    steps += [{"name": "engine_check_after", "kind": "engine_check", "model": model},
              {"name": "validate", "kind": "validate"}, {"name": "lineage_check", "kind": "lineage_check"}]
    return {"plan_version": PLAN_VERSION, "name": "e1_stage1e", "model": model, "repeats": repeats, "steps": steps}


V1 = "openai_v1"
DEFAULT_T = {"ollama": 0.8, "openai_v1": 1.0}  # the endpoints' server defaults (env/engine-lock.json)
ENDPOINTS = {"ollama": "/api/chat", V1: "/v1/chat/completions"}


def default_temperatures(model: str, show=None) -> dict:
    """Each endpoint's default temperature for this model: its Modelfile settings over the server's, except what
    the endpoint sends itself (/v1 always sends temperature 1.0). qwen2.5:14b sets none (0.8 and 1.0);
    qwen3.8-27b-bdikit sets temperature 1, so /api/chat runs it at 1.0. show: model -> /api/show (for tests)."""
    from hbench import sampler
    if show is None:
        from hbench.runlog import ollama
        show = ollama.show
    info = show(model)
    if "error" in info:
        raise RuntimeError(f"cannot read {model}'s Modelfile settings: {info['error']}")
    ms = sampler.model_sampler(info.get("parameters"))
    out = {t: sampler.defaults(ep, model=ms).get("temp") for t, ep in ENDPOINTS.items()}
    if any(v is None for v in out.values()):
        raise RuntimeError(f"no sampler defaults for {model}: {out}")
    return out


def make_stage3(columns: list[tuple[str, str]], stage1: str, chat_seeds: int = 12, v1_seeds: int = 10,
                p6_seeds: int = 3, agent_seeds: int = 6, model: str = MODEL, think: bool | None = None,
                chat_by: str = "path", n_p6: int | None = None, repeat_subset: str = "all",
                temperatures: dict | None = None) -> dict:
    """E1 Stage 3 (design doc section 12): sampling at the endpoints' default temperatures.
      3c-ref  every column's tool-loop episode at T=0 on /v1 (the agent's route), from an unload; then each
              replayed per call cold, which gives batched-lineage greedy sources for the /v1 resampling
      3b      calibration (P7): resample the selected calls at the default temperature, one cold_start pass per
              seed. /api/chat: clark decisions from Stage 1, stratified by 1 - p_path, plus p_path = 1 controls
              (chat_by="value": every Stage 1 call, for a thinking model whose p_path is near 0 everywhere).
              /v1: the final call of every reference episode plus earlier calls spread over p_path.
      3a      seed x lineage (P6): passes 1..p6_seeds repeated (same seed, same lineage: bitwise equal?) on every
              selected call (repeat_subset="all") or on the P6 subset ("p6"), and the P6 subset resent (same seed,
              single-token lineage)
      3c      whole episodes at the /v1 default temperature, seeds 1..agent_seeds, each from an unload; then
              seed 1 again (same seed, same lineage: is an agent run at T>0 reproducible with a seed?)
    Every T>0 call gets its own seed derived from the replicate's seed as a base (hbench.seeds; resample.call_seed for resampled calls): with
    one seed for all calls, llama-server would sample every call with the same random draws.
    The default temperatures are the model's (default_temperatures()); model and think go to every episode."""
    temps = temperatures or (DEFAULT_T if model == MODEL else default_temperatures(model))
    sampling = {**({"model": model} if model != MODEL else {}), **({"think": think} if think is not None else {})}
    sel_opts = {k: v for k, v in (("by", None if chat_by == "path" else chat_by), ("n_p6", n_p6)) if v is not None}
    steps = [{"name": "engine_check_before", "kind": "engine_check", "model": model}]
    steps += [episode(f"v1_ref:{s}:{c}", s, c, 1, "v1_ref", "cold_start", transport=V1, temperature=0, seed=1,
                      **sampling) for s, c in columns]
    steps += [{"name": f"v1_ref_replay:{s}:{c}", "kind": "replay", "of": f"v1_ref:{s}:{c}", "protocols": ["cold"],
               "raw": "raw/v1_ref_replay"} for s, c in columns]
    steps += [{"name": "select:chat", "kind": "select", "what": "chat", "stage1": stage1,
               "out": "selections/chat.json", **({"options": sel_opts} if sel_opts else {})},
              {"name": "select:v1", "kind": "select", "what": "v1", "replays_of": "v1_ref_replay:",
               "out": "selections/v1.json", **({"options": {"n_p6": n_p6}} if n_p6 is not None else {})}]
    for ep, n_cal, temp in (("chat", chat_seeds, temps["ollama"]), ("v1", v1_seeds, temps[V1])):
        sel, ov = f"selections/{ep}.json", {"temperature": temp}
        steps += [resample(f"{ep}:cal:s{k}", sel, "all", "cold_start", ov, k, f"{ep}_cal") for k in range(1, n_cal + 1)]
        steps += [resample(f"{ep}:repeat:s{k}", sel, repeat_subset, "cold_start", ov, k, f"{ep}_repeat")
                  for k in range(1, p6_seeds + 1)]
        steps.append({"name": f"{ep}:resend:unload", "kind": "unload", "model": model})
        steps += [resample(f"{ep}:resend:s{k}", sel, "p6", "resend", ov, k, f"{ep}_resend")
                  for k in range(1, p6_seeds + 1)]
    steps += [episode(f"v1_sampled:{s}:{c}:s{k}", s, c, k, "v1_sampled", "cold_start", transport=V1,
                      temperature=temps[V1], seed=k, **sampling)
              for k in range(1, agent_seeds + 1) for s, c in columns]
    steps += [episode(f"v1_sampled_repeat:{s}:{c}:s1", s, c, 1, "v1_sampled_repeat", "cold_start", transport=V1,
                      temperature=temps[V1], seed=1, **sampling) for s, c in columns]
    steps += [{"name": "engine_check_after", "kind": "engine_check", "model": model},
              {"name": "validate", "kind": "validate"}, {"name": "lineage_check", "kind": "lineage_check"}]
    return {"plan_version": PLAN_VERSION, "name": "e1_stage3", "model": model, "think": think, "columns": columns,
            "stage1": stage1, "temperatures": temps, "chat_by": chat_by, "repeat_subset": repeat_subset,
            "seeds": {"chat": chat_seeds, "v1": v1_seeds, "p6": p6_seeds, "agent": agent_seeds}, "steps": steps}


def resample(name, selection, subset, protocol, overrides, seed, block) -> dict:
    return {"name": name, "kind": "resample", "selection": selection, "subset": subset, "protocol": protocol,
            "overrides": overrides, "seed": seed, "block": block, "raw": f"raw/{block}"}


PLANS = {"stage1e": make_stage1e, "stage2": make_stage2, "stage3": make_stage3}


# ------------------------------------------------------------------ execution

def run_dirs(root: Path) -> set[str]:
    return {str(p.parent.relative_to(root)) for p in root.glob("raw/**/events.driver.jsonl")}


def fill(arg: str, root: Path) -> str:
    return arg.format(python=PY, repo=REPO, root=root, here=HERE)


def sh(argv, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=timeout)


def loaded_models() -> list[str]:
    with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=10) as r:
        return [m.get("name") for m in json.load(r).get("models", [])]


def do_step(step: dict, root: Path, done: dict, data: Data) -> dict:
    kind = step["kind"]
    if kind == "engine_check":
        tagged = lambda m: m if ":" in m else m + ":latest"  # noqa: E731  (ps lists "name:latest")
        others = [m for m in loaded_models() if tagged(m) != tagged(step["model"])]
        if others:
            return {"status": "failed", "error": f"other models are loaded ({', '.join(others)}): another client is "
                                                 "using the server, so its requests would interleave with ours"}
        p = sh([PY, "-m", "hbench", "engine", "check", "--models", step["model"]])
        return {"status": "ok" if p.returncode == 0 else "failed", "rc": p.returncode, "out": p.stdout[-2000:]}
    if kind == "unload":
        from hbench.design import ollama_lease
        from hbench.llm import LLMClient
        with ollama_lease(f"run_plan:{step['name']}", wait=60):
            ok = LLMClient(None, step["model"]).unload()
        return {"status": "ok" if ok else "failed", "confirmed": ok}
    if kind == "tool_loop":
        import tool_loop
        s = tool_loop.run_episode(step["study"], step["column"], replicate=step["replicate"], block=step["block"],
                                  order=step["order"], protocol=step["protocol"], raw_dir=root / step["raw"],
                                  data=data, **{k: step[k] for k in ("transport", "temperature", "seed", "model", "think")
                                                if k in step})
        return {"status": "ok", "loop_status": s["status"], "final": s["final"], "n_steps": s["n_steps"]}
    if kind == "cmd":
        p = sh([fill(a, root) for a in step["argv"]])
        return {"status": "ok" if p.returncode == 0 else "failed", "rc": p.returncode, "out": p.stdout[-2000:],
                "err": p.stderr[-2000:]}
    if kind == "replay":
        src = done.get(step["of"])
        if not src or src.get("status") != "ok" or len(src.get("runs") or []) != 1:
            return {"status": "skipped", "error": f"step {step['of']!r} did not produce exactly one run"}
        p = sh([PY, "-m", "hbench", "replay", str(root / src["runs"][0]), "--protocol", *step["protocols"],
                "--out", str(root / step["raw"])])
        return {"status": "ok" if p.returncode == 0 else "failed", "rc": p.returncode, "out": p.stdout[-2000:],
                "err": p.stderr[-2000:]}
    if kind == "select":
        import resample as rs
        opts = step.get("options") or {}
        if step["what"] == "chat":
            sel = rs.select_chat(Path(step["stage1"]), **opts)
        else:
            runs = [root / r for n, rec in done.items() if n.startswith(step["replays_of"])
                    and rec.get("status") == "ok" for r in rec.get("runs") or []]
            sel = rs.select_v1(sorted(runs), **opts)
        out = root / step["out"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(sel, indent=1))
        return {"status": "ok" if sel["sources"] else "failed", "n_sources": len(sel["sources"]),
                "n_p6": len(sel["p6"]), "sha": sel["sha"]}
    if kind == "resample":
        import resample as rs
        sel = json.loads((root / step["selection"]).read_text())
        sources = sel["sources"] if step["subset"] == "all" else sel[step["subset"]]
        res = rs.run_pass(sources, protocol=step["protocol"], overrides=step["overrides"], seed=step["seed"],
                          raw_dir=root / step["raw"], label=step["name"], block=step["block"], order=step["order"])
        return {"status": "ok", **{k: v for k, v in res.items() if k != "run_dir"}}
    if kind == "validate":
        problems = {}
        for rd in sorted(run_dirs(root)):
            p = sh([PY, "-m", "hbench", "log", "validate", str(root / rd)])
            try:
                res = json.loads(p.stdout)
            except json.JSONDecodeError:
                res = {"problems": [f"unreadable output: {p.stdout[-300:]} {p.stderr[-300:]}"]}
            if res.get("problems"):
                problems[rd] = res["problems"]
        (root / "validate_summary.json").write_text(json.dumps({"n_runs": len(run_dirs(root)),
                                                                "runs_with_problems": problems}, indent=1))
        return {"status": "ok" if not problems else "failed", "n_runs": len(run_dirs(root)),
                "n_with_problems": len(problems)}
    if kind == "lineage_check":
        p = sh([PY, "-m", "hbench", "lineage", "check", *[str(root / rd) for rd in sorted(run_dirs(root))]])
        (root / "lineage_check.json").write_text(p.stdout)
        return {"status": "ok" if p.returncode == 0 else "failed", "rc": p.returncode}
    raise ValueError(f"unknown step kind {kind!r}")


def execute(plan: dict, root: Path, resume: bool = False) -> int:
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan, indent=1)
    sha = hashlib.sha256(text.encode()).hexdigest()
    plan_file, exec_file = root / "plan.json", root / "execution.jsonl"
    if plan_file.exists():
        prior = json.loads(plan_file.read_text())
        if prior.get("sha256") != sha:
            sys.exit(f"{plan_file} holds a different plan; use another --root")
        if not resume:
            sys.exit(f"{root} already ran this plan; pass --resume to continue it")
    else:
        plan_file.write_text(json.dumps({"sha256": sha, "plan": plan}, indent=1))
    done = {}
    if exec_file.exists():
        for line in exec_file.read_text().splitlines():
            rec = json.loads(line)
            done[rec["name"]] = rec  # the last record of a step wins
    data = Data(default_root())
    names = [s["name"] for s in plan["steps"]]
    if len(set(names)) != len(names):
        sys.exit("step names must be unique")
    for i, step in enumerate(plan["steps"]):
        if resume and (done.get(step["name"]) or {}).get("status") == "ok":
            continue
        step = {**step, "order": i}
        before, t0 = run_dirs(root), time.time()
        try:
            res = do_step(step, root, done, data)
        except Exception as e:
            res = {"status": "failed", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        rec = {"order": i, "name": step["name"], "kind": step["kind"], "start": t0, "secs": round(time.time() - t0, 1),
               "runs": sorted(run_dirs(root) - before), **res}
        done[step["name"]] = rec
        with open(exec_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {i:4d} {step['name']}: {rec['status']} {rec['secs']}s", flush=True)
        if rec["status"] == "failed" and (step["kind"] in ("engine_check", "unload") or step.get("stop_on_fail")):
            print(f"stopping: {rec.get('error') or rec.get('out', '')[-500:]}", flush=True)
            return 1
    failed = [n for n, r in done.items() if r.get("status") != "ok"]
    print(f"done: {len(plan['steps'])} steps, {len(failed)} not ok" + (f": {failed[:10]}" if failed else ""))
    return 0 if not failed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="write a plan")
    mk.add_argument("plan", choices=sorted(PLANS))
    mk.add_argument("--out", required=True)
    mk.add_argument("--columns", default="clark:10,krug:10", help="study:n,... (the first n source columns)")
    mk.add_argument("--repeats", type=int, default=None, help="default: 3 for stage2, 2 for stage1e")
    mk.add_argument("--model", default=MODEL)
    mk.add_argument("--think", choices=["on", "off"], default=None, help="stage2/3: thinking (unset = model default)")
    mk.add_argument("--blocks", default=None, help="stage2: blocks to keep, e.g. unload_each,back_to_back")
    mk.add_argument("--no-single-turn", action="store_true", help="stage2: drop the clark single-turn check")
    mk.add_argument("--stage1", default=None, help="stage3: the Stage 1 results folder (holds 1a/raw)")
    mk.add_argument("--seeds", default=None, help="stage3: chat,v1,p6,agent seed counts (default 12,10,3,6)")
    mk.add_argument("--chat-by", choices=["path", "value"], default="path",
                    help="stage3: /api/chat selection by p_path bins (14b) or every Stage 1 call (thinking model)")
    mk.add_argument("--p6", type=int, default=None, help="stage3: size of the P6 subsets (default 20 chat, 10 v1)")
    mk.add_argument("--repeat-subset", choices=["all", "p6"], default="all",
                    help="stage3: calls the same-seed repeat passes send")
    rn = sub.add_parser("run", help="execute a plan")
    rn.add_argument("plan_file")
    rn.add_argument("--root", required=True, help="output root, outside the repo")
    rn.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        cols = pick_columns(a.columns, Data(default_root())) if a.plan in ("stage2", "stage3") else None
        if a.plan == "stage3":
            if not a.stage1:
                sys.exit("stage3 needs --stage1 <Stage 1 results folder>")
            n = dict(zip(("chat_seeds", "v1_seeds", "p6_seeds", "agent_seeds"),
                         map(int, a.seeds.split(",")))) if a.seeds else {}
            plan = make_stage3(cols, str(Path(a.stage1).resolve()), **n, model=a.model,
                               think=None if a.think is None else a.think == "on", chat_by=a.chat_by, n_p6=a.p6,
                               repeat_subset=a.repeat_subset)
        elif a.plan == "stage2":
            plan = make_stage2(cols, **({"repeats": a.repeats} if a.repeats else {}), model=a.model,
                               think=None if a.think is None else a.think == "on",
                               blocks=tuple(a.blocks.split(",")) if a.blocks else STAGE2_BLOCKS,
                               single_turn=not a.no_single_turn)
        else:
            plan = PLANS[a.plan](cols, **({"repeats": a.repeats} if a.repeats else {}), model=a.model)
        Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
        print(f"{a.out}: {len(plan['steps'])} steps")
        return 0
    root = Path(a.root).resolve()
    if root == REPO or REPO in root.parents:
        sys.exit("--root must be outside the repo: raw logs carry Qwen's Chinese candidate tokens")
    return execute(json.loads(Path(a.plan_file).read_text()), root, resume=a.resume)


if __name__ == "__main__":
    sys.exit(main())
