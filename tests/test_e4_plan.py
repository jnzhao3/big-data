"""E4 run tooling without Ollama: the plan generator, run_plan's stop on a failed check step, the Z1=Z2 compare
and the morning summary."""

import json
import sys
from pathlib import Path

import pytest

from hbench.data import REPO_ROOT

E4 = REPO_ROOT / "experiments" / "e4_agent_repeats"
sys.path.insert(0, str(E4))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "e1_noise_floor"))
import make_plan as mp  # noqa: E402
import run_plan  # noqa: E402


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_night1_follows_appendix_a_with_shared_settings(tmp_path):
    plan = mp.make_plan([mp.parse_cell(c) for c in mp.night_cells("N1")], "e4_N1", raw_dir=tmp_path / "raw",
                        cache=tmp_path / "e4_t0cache")
    names = [s["name"] for s in plan["steps"]]
    assert names == ["engine_check_before", "clark:Z1", "clark:Z2", "check:clark:Z1=Z2", "mcdermott:Z", "huang:Z",
                     "clark:D1x1", "mcdermott:D1x1", "huang:D1x1", "clark:D2x2", "mcdermott:D2x2", "huang:D2x2",
                     "clark:D3x3", "mcdermott:D3x3", "clark:D4x4", "mcdermott:D4x4", "engine_check_after"]
    runs = [s for s in plan["steps"] if s.get("cell")]
    assert len(runs) == 14
    raw = str((tmp_path / "raw").resolve())
    for s in runs:
        a = s["argv"]
        assert a[2] == "{repo}/experiments/e3_agent/run_agent.py" and "{root}" not in " ".join(a)
        assert (_arg(a, "--raw-dir"), _arg(a, "--model"), _arg(a, "--task")) == (raw, mp.MODEL, "sm")
        assert (_arg(a, "--turn-timeout"), _arg(a, "--engine-protocol")) == ("10800", "cold_start")
        for f in ("--temperature", "--top-p", "--tool-temperature", "--tool-top-p", "--seed", "--block", "--replicate"):
            assert _arg(a, f) is not None, f
        assert "--tool-seed" not in a                                    # night 1 is all diagonal
    z1, z2 = runs[0]["argv"], runs[1]["argv"]
    assert _arg(z1, "--bdikit-cache") == "shared" and _arg(z1, "--bdikit-shared") == plan["t0_cache"]
    assert "--bdikit-cache" not in z2 and _arg(z1, "--replicate") == "1" and _arg(z2, "--replicate") == "2"
    check = plan["steps"][3]
    assert check["stop_on_fail"] and check["argv"][-2:] == ["clark:Z1", "clark:Z2"]


def test_arms_set_sampling_seeds_and_extras():
    def argv(spec, **kw):
        return mp.make_plan([mp.parse_cell(spec)], "x", **kw)["steps"][1]["argv"]
    d = argv("clark:D:2:3")
    assert [_arg(d, f) for f in ("--temperature", "--top-p", "--tool-temperature", "--tool-top-p", "--seed",
                                 "--tool-seed", "--block", "--replicate")] == ["1.0", "1.0", "1.0", "1.0", "2", "3",
                                                                               "D", "23"]
    assert "--tool-seed" not in argv("clark:D:2:2")
    t = argv("clark:T:4")
    assert [_arg(t, f) for f in ("--temperature", "--tool-temperature", "--tool-top-p", "--seed")] == \
        ["0", "1.0", "1.0", "4"]
    a = argv("clark:A:5")
    assert [_arg(a, f) for f in ("--temperature", "--tool-temperature", "--seed", "--bdikit-cache")] == \
        ["1.0", "0", "5", "shared"]
    assert _arg(argv("clark:P-hash:2"), "--hashseed") == "random"
    assert _arg(argv("clark:P-prompt"), "--prompt-variant") == "no_json_format"
    r = argv("clark:P-row:3")
    assert _arg(r, "--row-order") == "3" and _arg(r, "--temperature") == "0" and _arg(r, "--seed") == "1"
    q = argv("clark:D:1:1", sample_t=0.6, sample_p=0.95)                 # the sampled side is an input
    assert (_arg(q, "--temperature"), _arg(q, "--top-p")) == ("0.6", "0.95")
    assert "--bdikit-cache" not in argv("clark:Z") and "--bdikit-cache" not in d


def test_nights_and_study_roles():
    n2 = mp.night_cells("N2", ("clark", "mcdermott", "wang"))
    assert n2[:4] == ["clark:D:1:2", "clark:D:2:3", "clark:D:3:4", "clark:D:4:1"] and n2[4] == "wang:D:3:3"
    total = sum(len(mp.night_cells(n)) for n in ("N1", "N2", "N3", "N4", "N5"))
    assert total == 74                                                   # doc appendix A
    every = [c for n in ("N1", "N2", "N3", "N4", "N5") for c in mp.night_cells(n) if ":D:" in c]
    assert len(every) == len(set(every))                                 # no cell twice
    for s in ("clark", "mcdermott", "huang"):                            # the full 4 x 4 grid per study
        assert {c for c in every if c.startswith(s + ":")} == {f"{s}:D:{a}:{t}" for a in range(1, 5)
                                                               for t in range(1, 5)}
    with pytest.raises(ValueError, match="duplicate"):
        mp.make_plan([mp.parse_cell("clark:Z"), mp.parse_cell("clark:Z")], "x")
    with pytest.raises(ValueError, match="Magneto"):
        mp.make_plan([mp.parse_cell("clark:Z1")], "x", cache=Path("/tmp/mpnet_cache"))
    with pytest.raises(ValueError):
        mp.parse_cell("clark:D:1")


def test_prepare_links_the_shared_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "SHARED", tmp_path / "shared")
    plan = mp.make_plan([mp.parse_cell("clark:Z1")], "x", cache=tmp_path / "e4_t0cache")
    assert "0 cached" in mp.prepare(plan) and "0 cached" in mp.prepare(plan)   # idempotent
    for sub in ("models", "embeddings"):
        assert (tmp_path / "e4_t0cache" / sub).resolve() == (tmp_path / "shared" / sub).resolve()
    (tmp_path / "e4_t0cache" / "runs" / "old").write_text("x")
    assert "must start from an empty" in mp.prepare(plan)
    assert "no shared T=0 cache" in mp.prepare(mp.make_plan([mp.parse_cell("clark:D:1:1")], "x"))


def test_prepare_refuses_a_data_root_without_the_studies(tmp_path, monkeypatch, data):
    plan = mp.make_plan([mp.parse_cell("clark:D:1:1"), mp.parse_cell("huang:Z")], "x")
    assert "ok for ['clark', 'huang']" in mp.check_data(plan)
    monkeypatch.setenv("HBENCH_DATA", str(tmp_path / "nodata"))             # a pinned checkout without datasets/
    with pytest.raises(SystemExit, match="HBENCH_DATA"):
        mp.prepare(plan)


def test_out_parent_is_created(tmp_path):
    out = tmp_path / "plans" / "deep" / "n1.json"
    assert mp.main(["night", "N1", "--out", str(out)]) == 0 and json.loads(out.read_text())["name"] == "e4_N1"


def test_run_plan_stops_at_a_failed_check_step(tmp_path, data):
    ok = {"kind": "cmd", "argv": ["{python}", "-c", "pass"]}
    bad = {"kind": "cmd", "argv": ["{python}", "-c", "import sys; sys.exit(1)"]}
    plan = {"name": "t", "steps": [{"name": "a", **bad}, {"name": "b", **ok}, {"name": "check", **bad, "stop_on_fail": True},
                                   {"name": "c", **ok}]}
    assert run_plan.execute(plan, tmp_path / "root") == 1
    recs = [json.loads(x) for x in (tmp_path / "root" / "execution.jsonl").read_text().splitlines()]
    assert [(r["name"], r["status"]) for r in recs] == [("a", "failed"), ("b", "ok"), ("check", "failed")]


def _fake_run(d: Path, calls, preds, outcome="final_table"):
    d.mkdir(parents=True)
    ev = [{"type": "run_header", "proc": "driver"}] + [
        {"type": "llm_call", "proc": "driver", "caller": "agent", "request_sha256": r, "output_sha256": o,
         "logprobs_sha256": o, "decoding": {"seed": 7}} for r, o in calls] + [
        {"type": "run_end", "proc": "driver", "status": "ok", "outcome": outcome, "followups": 0, "wall_s": 600,
         "scores": [{"kind": "sm", "top1_correct": 12, "n_annotated": 14}]}]
    (d / "events.driver.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    (d / "events.mcp.jsonl").write_text(json.dumps({"type": "llm_call", "proc": "mcp", "caller": "tool:x",
                                                    "request_sha256": "q", "output_sha256": "w"}) + "\n")
    (d / "predictions.jsonl").write_text("".join(json.dumps({"run_id": d.name, "kind": "sm", "t": p}) + "\n"
                                                 for p in preds))
    (d / "engine_audit.json").write_text(json.dumps({"summary": {"matched": 2, "n_calls": 2, "foreign_tasks": 0}}))
    return d


def test_compare_and_summary(tmp_path, capsys):
    root = tmp_path / "root"
    root.mkdir()
    r1 = _fake_run(tmp_path / "raw" / "z1", [("a", "b"), ("c", "d")], ["x"])
    r2 = _fake_run(tmp_path / "raw" / "z2", [("a", "b"), ("c", "d")], ["x"])
    r3 = _fake_run(tmp_path / "raw" / "z3", [("a", "b"), ("c", "e")], ["x"])
    plan = mp.make_plan([mp.parse_cell(c) for c in ("clark:Z1", "clark:Z2", "clark:Z")], "t")
    (root / "plan.json").write_text(json.dumps({"sha256": "x", "plan": plan}))
    recs = [{"name": "engine_check_before", "status": "ok", "secs": 1},
            {"name": "clark:Z1", "status": "ok", "secs": 600, "out": f"noise\n{r1}\n"},
            {"name": "clark:Z2", "status": "ok", "secs": 600, "out": f"{r2}\n"},
            {"name": "clark:Z", "status": "ok", "secs": 600, "out": f"{r3}\n"}]
    (root / "execution.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert mp.main(["compare", "--root", str(root), "clark:Z1", "clark:Z2"]) == 0
    assert mp.main(["compare", "--root", str(root), "clark:Z1", "clark:Z"]) == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["driver"]["first_difference"] == 1
    assert mp.main(["compare", "--root", str(root), "clark:Z1", "nothing"]) == 1
    out = mp.summarize(root, run_validate=False)
    by = {r["name"]: r for r in out["steps"]}
    assert by["clark:Z1"]["outcome"] == "final_table" and by["clark:Z1"]["wall_min"] == 10.0
    assert by["clark:Z1"]["run_dir"] == str(r1) and by["clark:Z1"]["sm"] == [(12, 14)]
    assert by["check:clark:Z1=Z2"]["status"] == "not run" and out["counts"]["not run"] == 2
    assert "| clark:Z2 | ok | 10.0 | final_table | 12/14 | 2/2 matched, 0 foreign |" in (root / "summary.md").read_text()


def test_night_script_resumes_and_summarizes():
    text = (E4 / "night.sh").read_text()
    for needle in ("caffeinate -i -s", "api/version", "i4Tools", "run_plan.py\" run", "--resume", "summary --root",
                   "prepare", "HBENCH_DATA", "--git-common-dir", "not starting the night"):
        assert needle in text, needle
