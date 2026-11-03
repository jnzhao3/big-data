import json
from pathlib import Path

import pytest

from hbench.engine import lineage
from hbench.engine.audit import parse_requests

FX = Path(__file__).parent / "fixtures"
LOG = (FX / "ollama_server_lineage.log").read_text()
CALLS = json.loads((FX / "probe3_lineage_calls.json").read_text())["calls"]


def _probe3_records():
    """Lineage of each of probe 3's 12 calls, reconstructed from the server log alone."""
    records = lineage.reconstruct(parse_requests(LOG)["stream"])
    assert len(records) == 13  # task 4 is the streaming request, made outside hbench between the two runs
    return [(c, records[c["window_task_index"]]) for c in CALLS]


def test_probe3_lineages_from_the_real_log():
    got = _probe3_records()
    assert [r["canonical"] for _, r in got] == [c["lineage"] for c, _ in got]
    assert all(not r["uncertain"] for _, r in got) and got[0][1]["reload_before"]
    live2 = got[1][1]  # live /api/chat turn 2: 16 generated tokens of turn 1 reused
    assert (live2["gen_reused"], live2["inherited_single"]) == (16, 0) and live2["prefix_origins"] == [0]
    as_is2 = got[-1][1]  # same counts on the face of it, but it inherited the single-token recompute of position 236
    assert (as_is2["gen_reused"], as_is2["inherited_single"]) == (16, 1)


def test_same_lineage_iff_bitwise_on_probe3():
    got = _probe3_records()
    for i, (a, ra) in enumerate(got):
        for b, rb in got[i + 1:]:
            if a["step"] == b["step"] and a["n_tokens"] == b["n_tokens"]:  # the same request
                assert (ra["fingerprint"] == rb["fingerprint"]) == (a["logprobs_sha256"] == b["logprobs_sha256"])


def _task(n, cached, gen):
    return {"kind": "task", "task": {"n_tokens": n, "cached": cached, "eval_tokens": gen, "stop_n_tokens": n + gen - 1}}


def test_small_last_chunk_is_kept_separate():
    cold, partial = lineage.reconstruct([{"kind": "runner_start"}, _task(1030, 0, 5), _task(1030, 50, 5)])
    assert cold["canonical"] == "B1024 b6"     # the last chunk of a cold prompt over 1024 tokens is only 6 tokens
    assert partial["canonical"] == "B1030"     # cached 50 of a large chunk, then one chunk of 980
    assert cold["fingerprint"] != partial["fingerprint"]


def test_ram_restore_and_its_ambiguity():
    stream = [{"kind": "runner_start"}, _task(100, 0, 3),                    # slot: B100 G2
              {"kind": "save", "length": 102}, _task(80, 0, 3),               # another prompt
              {"kind": "save", "length": 82},
              {"kind": "candidates", "items": [[102, 100, "0.980", "1.000"], [82, 10, "0.122", "0.100"]]},
              {"kind": "restore", "f_keep": "0.980", "f_sim": "1.000"}, _task(100, 99, 3)]
    rec = lineage.reconstruct(stream)[-1]
    assert rec["canonical"] == "B99 S1" and rec["prefix_origins"] == [0] and not rec["uncertain"]
    stream[5] = {"kind": "candidates", "items": [[102, 100, "0.980", "1.000"], [102, 100, "0.980", "1.000"]]}
    assert "ambiguous restore" in lineage.reconstruct(stream)[-1]["uncertain"]


def test_unknown_start_and_several_runners():
    rec = lineage.reconstruct([_task(100, 40, 3)])[0]
    assert rec["canonical"] == "?40 B60" and rec["uncertain"] == ["unknown start"]
    recs = lineage.reconstruct([{"kind": "runner_start"}, _task(100, 0, 3),
                                {"kind": "runners_loaded", "count": 2}, _task(100, 99, 3)])
    assert not recs[0]["uncertain"] and "several runners" in recs[1]["uncertain"]


def test_check_reports_same_lineage_with_different_logprobs(tmp_path):
    from hbench.runlog import RunLog
    from hbench.runlog.check import check_run
    log = RunLog.create(tmp_path / "raw", "e1", "t", "clark", run_id="r")
    log.emit("run_header", engine=None)
    spans = [log.emit("llm_call", prompt_sha256="p1", input_sha256="i1", logprobs_sha256=h,
                      decoding={"temperature": 0}) for h in ("aa", "bb")]
    log.close()
    rows = [{"span_id": s, "lineage": {"fingerprint": "f1", "canonical": "B10", "uncertain": []}} for s in spans]
    (log.run_dir / "engine_audit.json").write_text(json.dumps({"summary": {}, "calls": rows}))
    got = lineage.check([log.run_dir])
    assert len(got["violations"]) == 1 and got["n_groups"] == 1
    assert any(p.startswith("lineage: same lineage, different logprobs") for p in check_run(log.run_dir)["problems"])


def _audited_run(tmp_path):
    """The probe 3 window as one run: its 12 calls, audited against the fixture log."""
    import shutil
    from hbench.engine.audit import audit_run
    from hbench.runlog import RunLog
    shutil.copy(FX / "ollama_server_lineage.log", tmp_path / "server.log")
    log = RunLog.create(tmp_path / "raw", "e1", "t", "clark", run_id="r")
    log.emit("run_header", engine={"server_log": {"path": str(tmp_path / "server.log"), "offset": 0}})
    # the streaming request between the /v1 run and the cold replay was made outside hbench; probe 3 found its
    # logprobs identical to the /v1 turn 1 it repeated, so it enters this run as that call again
    for c in CALLS[:4] + [CALLS[2]] + CALLS[4:]:
        log.emit("llm_call", api_base="http://localhost:11434", transport=c["transport"], decoding={"temperature": 0},
                 response_raw={"blob": "sha256:00", "bytes": 1}, engine={"prompt_tokens": c["n_tokens"]},
                 input_sha256=f'{c["step"]}', logprobs_sha256=c["logprobs_sha256"])
    log.close()
    return log.run_dir, audit_run(log.run_dir)


def test_audit_writes_each_calls_lineage(tmp_path):
    run_dir, summary = _audited_run(tmp_path)
    rows = json.loads((run_dir / "engine_audit.json").read_text())["calls"]
    want = [c["lineage"] for c in CALLS[:4] + [CALLS[2]] + CALLS[4:]]
    assert [r["lineage"]["canonical"] for r in rows] == want and summary["lineage_uncertain"] == []
    assert rows[1]["lineage"]["prefix_origins"] == ["driver:2"]  # live turn 2 reused what turn 1 computed


def test_cli_lineage_rebuild_and_check(tmp_path):
    from hbench.cli import main
    run_dir, _ = _audited_run(tmp_path)
    out = tmp_path / "lineage"
    out.mkdir()
    assert main(["lineage", "rebuild", str(run_dir), "--out-dir", str(out)]) == 0
    rebuilt = json.loads((out / "r.lineage.json").read_text())
    assert len(rebuilt["calls"]) == 13 and rebuilt["calls"][0]["lineage"]["canonical"] == "B237"
    assert main(["lineage", "check", str(run_dir), "--lineage-dir", str(out)]) == 0


def test_a_restored_prompt_leaves_the_cache():
    # the server moves a restored entry into the slot ("cache state: N-1 prompts" after it), so the next lookup
    # lists one entry fewer; E1 Stage 1e shortlist r5 restored 16 prompts in a row this way
    stream = [{"kind": "runner_start"}, _task(100, 0, 3), {"kind": "save", "length": 102},
              _task(80, 0, 3), {"kind": "save", "length": 82},
              _task(90, 0, 3), {"kind": "save", "length": 92},                       # cache: A102, B82, C92
              {"kind": "candidates", "items": [[102, 100, "0.980", "1.000"], [82, 5, "0.061", "0.050"],
                                               [92, 5, "0.054", "0.050"]]},
              {"kind": "restore", "f_keep": "0.980", "f_sim": "1.000"}, _task(100, 99, 3),  # A leaves the cache
              {"kind": "save", "length": 102},                                        # cache: B82, C92, A'102
              {"kind": "candidates", "items": [[82, 80, "0.976", "1.000"], [92, 5, "0.054", "0.062"],
                                               [102, 5, "0.049", "0.062"]]},
              {"kind": "restore", "f_keep": "0.976", "f_sim": "1.000"}, _task(80, 79, 3)]
    last = lineage.reconstruct(stream)[-1]
    assert last["uncertain"] == [] and last["canonical"] == "B79 S1" and last["prefix_origins"] == [1]


def test_uncertainty_from_a_restore_is_inherited():
    stream = [{"kind": "runner_start"}, _task(100, 0, 3), {"kind": "save", "length": 102},
              _task(80, 0, 3), {"kind": "save", "length": 82},
              {"kind": "candidates", "items": [[102, 100, "0.980", "1.000"], [82, 100, "0.980", "1.000"]]},
              {"kind": "restore", "f_keep": "0.980", "f_sim": "1.000"},   # two candidates match: ambiguous
              _task(100, 99, 3), _task(110, 102, 3), {"kind": "runner_start"}, _task(50, 0, 3)]
    a, b, fresh = lineage.reconstruct(stream)[2:]
    assert "ambiguous restore" in a["uncertain"]
    assert b["uncertain"] == ["inherits an uncertain lineage"]  # the next turn reuses the guessed positions
    assert fresh["uncertain"] == []                             # a runner start clears it


def test_a_task_without_counts_is_uncertain():
    rec = lineage.reconstruct([{"kind": "runner_start"}, {"kind": "task", "task": {
        "n_tokens": None, "cached": None, "eval_tokens": None, "stop_n_tokens": None}}])[0]
    assert rec["uncertain"] == ["no token counts in the server log"]


def test_a_skipped_save_is_not_cached():
    from hbench.engine.audit import parse_requests
    text = ("srv   prompt_save:  - saving prompt with length 82, total state size = 1 MiB\n"
            "srv         alloc:  - prompt is already in the cache, skipping\n")
    assert [e["kind"] for e in parse_requests(text)["stream"]] == ["save", "save_skipped"]
    stream = [{"kind": "runner_start"}, _task(100, 0, 3), {"kind": "save", "length": 102}, _task(80, 0, 3),
              {"kind": "save", "length": 82}, {"kind": "save_skipped"},
              {"kind": "candidates", "items": [[102, 100, "0.980", "1.000"]]},
              {"kind": "restore", "f_keep": "0.980", "f_sim": "1.000"}, _task(100, 99, 3)]
    assert lineage.reconstruct(stream)[-1]["uncertain"] == []
    stream[6] = {"kind": "candidates", "items": [[103, 100, "0.980", "1.000"]]}  # same count, other length
    assert "cache listing out of sync" in lineage.reconstruct(stream)[-1]["uncertain"]


def test_check_falls_back_to_the_audit_and_reports_coverage(tmp_path):
    from hbench.runlog import RunLog
    log = RunLog.create(tmp_path / "raw", "e1", "t", "clark", run_id="r")
    log.emit("run_header", engine=None)
    spans = [log.emit("llm_call", input_sha256="i1", logprobs_sha256=h, decoding={"temperature": 0})
             for h in ("aa", "bb", "cc")]
    log.close()
    rows = [{"span_id": s, "lineage": {"fingerprint": "f1", "canonical": "B10", "uncertain": u}}
            for s, u in zip(spans, [[], [], ["unknown start"]])]
    (log.run_dir / "engine_audit.json").write_text(json.dumps({"summary": {}, "calls": rows}))
    empty = tmp_path / "lineage"
    empty.mkdir()
    got = lineage.check([log.run_dir], lineage_dir=empty)  # no <run>.lineage.json there: use the audit
    assert len(got["violations"]) == 1
    assert got["coverage"]["r"] == {"t0_calls": 3, "checked": 2, "uncertain": 1, "no_lineage": 0}
