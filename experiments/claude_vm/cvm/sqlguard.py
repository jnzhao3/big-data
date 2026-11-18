"""The optional read-only SQL tool: a SQLite database of the source table and the GDC schema only.

The ground truth is never written into the database. The connection is opened read-only (URI mode=ro), and an
authorizer allows only SELECT over the three tables; everything else (ATTACH, PRAGMA, writes, temp objects) is
denied by SQLite itself, not by string checks. Single statements only, and at most `cap` rows come back.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

TABLES = ("source", "attributes", "permissible_values")

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", None)
if _RECURSIVE is not None:
    _ALLOWED_ACTIONS.add(_RECURSIVE)
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_db(path: Path, view) -> Path:
    """Write the database for one StudyView. Source cells are stored as text, exactly as in source.csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        cols = view.columns
        con.execute(f"CREATE TABLE source (row_id INTEGER, {', '.join(_q(c) + ' TEXT' for c in cols)})")
        con.executemany(f"INSERT INTO source VALUES ({', '.join('?' * (len(cols) + 1))})",
                        [(i, *row) for i, row in enumerate(view.table.itertuples(index=False, name=None))])
        con.execute("CREATE TABLE attributes (name TEXT PRIMARY KEY, description TEXT, enumerated INTEGER, "
                    "n_permissible INTEGER)")
        con.execute("CREATE TABLE permissible_values (attribute TEXT, value TEXT, description TEXT)")
        for name, entry in view.schema.items():
            vd = entry.get("value_data") or {}
            con.execute("INSERT INTO attributes VALUES (?, ?, ?, ?)",
                        (name, entry.get("column_description") or "", int(bool(vd)), len(vd)))
            con.executemany("INSERT INTO permissible_values VALUES (?, ?, ?)",
                            [(name, v, d or "") for v, d in vd.items()])
        con.commit()
    finally:
        con.close()
    return path


def _authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_READ:
        # dbname is None for some reads, e.g. count(*); ATTACH is denied, so no other database can be named
        return sqlite3.SQLITE_OK if arg1 in TABLES and dbname in ("main", None) else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (arg2 or "").lower() in _DENIED_FUNCTIONS else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class SqlGuard:
    def __init__(self, db_path: Path, cap: int):
        self.db_path = Path(db_path)
        self.cap = cap

    def query(self, sql: str) -> dict:
        sql = (sql or "").strip()
        if not sql:
            raise ValueError("empty query")
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            con.set_authorizer(_authorizer)
            try:
                cur = con.execute(sql)  # sqlite3 refuses more than one statement here
            except sqlite3.ProgrammingError as e:
                raise ValueError(f"only a single statement is allowed ({e})") from None
            except sqlite3.DatabaseError as e:
                raise ValueError(f"query rejected: {e}") from None
            if cur.description is None:
                raise ValueError("only SELECT queries are allowed")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(self.cap + 1)
        finally:
            con.close()
        truncated = len(rows) > self.cap
        return {"columns": cols, "rows": [list(r) for r in rows[: self.cap]], "returned": min(len(rows), self.cap),
                "truncated": truncated, "row_cap": self.cap}
