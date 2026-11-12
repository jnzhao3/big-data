"""A run started from a dirty tree keeps the code it used: the diff against its commit and the untracked sources."""

import subprocess

from hbench.runlog import header as hdr
from hbench.runlog.writer import RunLog


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return repo


def test_snapshot_holds_the_diff_and_untracked_sources(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(hdr, "REPO_ROOT", repo)
    assert hdr.code_snapshot() is None                                   # clean tree: the commit says it all
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new_script.py").write_text("print('hi')\n")
    (repo / "data.csv").write_text("a,b\n")
    (repo / "api_token.json").write_text("{}")
    snap = hdr.code_snapshot()
    assert "+x = 2" in snap["diff"] and snap["untracked"] == {"new_script.py": "print('hi')\n"}
    assert sorted(snap["skipped"]) == ["api_token.json", "data.csv"] and len(snap["git_commit"]) == 40


def test_log_code_restores_the_tree(tmp_path, monkeypatch):
    from hbench.cli import main
    from hbench.runlog.check import check_run
    repo = _repo(tmp_path)
    monkeypatch.setattr(hdr, "REPO_ROOT", repo)
    (repo / "a.py").write_text("x = 2\n" + "# padding\n" * 600)          # large enough to go to the blob store
    (repo / "new_script.py").write_text("print('hi')\n")
    log = RunLog.create(tmp_path / "raw", "e0", "t", "clark", run_id="dirty")
    log.emit("run_header", condition={}, code=hdr.code_info(), code_snapshot=hdr.code_snapshot())
    log.close()
    assert check_run(log.run_dir)["code"]["recoverable"] is True
    out = tmp_path / "restored"
    assert main(["log", "code", str(log.run_dir), "--out", str(out)]) == 0
    clone = tmp_path / "clone"
    _git(repo, "worktree", "add", "-q", "--detach", str(clone), "HEAD")
    _git(clone, "apply", str(out / "code.patch"))
    assert (clone / "a.py").read_text() == (repo / "a.py").read_text()
    assert (out / "untracked" / "new_script.py").read_text() == "print('hi')\n"


def test_old_dirty_runs_are_marked_unrecoverable(tmp_path):
    from hbench.runlog.check import check_run
    log = RunLog.create(tmp_path / "raw", "e0", "t", "clark", run_id="old")
    log.emit("run_header", condition={}, code={"git_commit": "abc", "git_dirty": True, "git_diff_sha256": "f1"})
    log.close()
    assert check_run(log.run_dir)["code"] == {"git_commit": "abc", "git_dirty": True, "recoverable": False}
