"""Ground-truth isolation and the SQL guard."""

import json

import pytest

from conftest import GT_SENTINELS
from cvm.sqlguard import SqlGuard, build_db
from cvm.study import StudyView
from cvm.tools import METHODS, MatcherTools, ToolError
from hbench.data import Data


@pytest.fixture
def no_gt_items(monkeypatch):
    """The server must never build items from the ground truth."""
    def boom(*a, **k):
        raise AssertionError("ground-truth item builder called")
    monkeypatch.setattr(Data, "vm_items", boom)
    monkeypatch.setattr(Data, "vm_rows", boom)


def _every_output(t: MatcherTools) -> str:
    """Call every tool with a spread of arguments; return all results and error messages as one string.
    No argument contains a sentinel, so any sentinel in the output came from the ground truth."""
    out = []

    def call(fn, *a, **k):
        try:
            out.append(json.dumps(fn(*a, **k), default=str))
        except (ToolError, ValueError) as e:
            out.append(str(e))

    call(t.describe_study)
    for c in ("Case", "Race", "Age", "NoSuchColumn"):
        call(t.profile_column, c)
        call(t.list_values, c, 0, 1000)
        call(t.sample_rows, 1000, 0, c)
    for a in ("race", "age_at_diagnosis", "gender", "missing_attr"):
        call(t.describe_attribute, a)
        call(t.list_permissible, a, 0, 1000)
        for m in METHODS:
            call(t.search_permissible, a, "GTONLY", m)
            call(t.search_permissible, a, "10957", m)
    for m in METHODS:
        call(t.search_attributes, "age", m)
    call(t.submit_value_matches, [
        {"source_attribute": "Race", "target_attribute": "race", "source_value": "GTONLY",
         "target_value": "white", "confidence": 0.5},
        {"source_attribute": "Age", "target_attribute": "age_at_diagnosis", "source_value": "30",
         "target_value": "10957", "confidence": 0.5}])
    call(t.get_progress)
    if t.sql is not None:
        for q in ("SELECT * FROM source", "SELECT * FROM attributes", "SELECT * FROM permissible_values",
                  "SELECT * FROM sqlite_master", "SELECT name FROM pragma_table_list"):
            call(t.query, q)
    return "\n".join(out)


@pytest.mark.parametrize("mode", ["oracle", "end_to_end"])
def test_no_tool_reveals_ground_truth(data, tmp_path, no_gt_items, mode):
    view = StudyView(data, "clark", mode)
    sql = SqlGuard(build_db(tmp_path / "m.db", view), cap=1000)
    from cvm.store import SubmissionStore
    t = MatcherTools(view, SubmissionStore(tmp_path / "s.jsonl"), cap=1000, sql=sql)
    if mode == "end_to_end":
        t.map_attribute("Race", "race")
        t.map_attribute("Age", "age_at_diagnosis")
    text = _every_output(t)
    for s in GT_SENTINELS:
        assert s not in text, f"ground-truth value {s!r} leaked"
    # the SQL database holds exactly the source table and the schema, no ground-truth table
    import sqlite3
    con = sqlite3.connect(tmp_path / "m.db")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert tables == {"source", "attributes", "permissible_values"}


def test_sql_guard(data, tmp_path):
    view = StudyView(data, "clark", "oracle")
    g = SqlGuard(build_db(tmp_path / "m.db", view), cap=3)
    r = g.query('SELECT "Race", count(*) FROM source GROUP BY "Race" ORDER BY 2 DESC')
    assert r["columns"] == ["Race", "count(*)"] and r["returned"] == 3 and r["truncated"] is True
    assert g.query("SELECT value FROM permissible_values WHERE attribute='race'")["returned"] == 3
    assert g.query("SELECT count(*) FROM source")["rows"] == [[6]]
    for bad in ("DELETE FROM source", "DROP TABLE source", "INSERT INTO source (row_id) VALUES (1)",
                "UPDATE source SET Race='x'", "CREATE TABLE t(x)", "PRAGMA table_info(source)",
                "ATTACH DATABASE 'x.db' AS x", "SELECT * FROM sqlite_master", "SELECT 1; SELECT 2",
                "SELECT load_extension('x')", ""):
        with pytest.raises(ValueError):
            g.query(bad)


def test_sql_disabled_by_default(make_tools):
    with pytest.raises(ToolError, match="disabled"):
        make_tools().query("SELECT 1")
