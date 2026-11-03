"""Append-only run logs.

Layout under a raw-results root (e.g. experiments/<exp>/results/raw/, which git ignores):

  runs_index.jsonl                       one line when each run starts and ends
  _blobs/sha256/ab/<sha256>.json.gz      content-addressed payloads, shared by all runs, written once
  <run_id>/events.<proc>.jsonl           one file per process, one JSON event per line
  <run_id>/predictions.jsonl, scores...  run outputs

Nothing is ever overwritten: run directories and event files are created exclusively, every event is
flushed and fsynced, and readers tolerate a truncated last line after a crash.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..predictions import jsonable

LOG_SCHEMA = "hbench.log/1"
INLINE_LIMIT = 4096  # bytes of canonical JSON; larger field values go to the blob store
EVENT_TYPES = ("run_header", "llm_call", "tool_call", "agent_step", "state_snapshot", "prediction_batch", "score",
               "error", "note", "run_end")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", str(s)).strip("_") or "x"


def make_run_id(exp: str, cond: str, study: str, stamp: str | None = None, rand: str | None = None) -> str:
    """<exp>-<cond>-<study>-<UTC yyyymmddThhmmssZ>-<4 hex>. Factor levels belong in the run header;
    the id only has to be unique and readable."""
    return f"{_slug(exp)}-{_slug(cond)}-{_slug(study)}-{stamp or utc_stamp()}-{rand or secrets.token_hex(2)}"


def canonical_json(obj) -> bytes:
    return json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha256_json(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


class BlobStore:
    """Write-once store of JSON payloads keyed by the sha256 of their canonical JSON."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, h: str) -> Path:
        return self.root / "sha256" / h[:2] / f"{h}.json.gz"

    def put(self, obj) -> dict:
        data = canonical_json(obj)
        h = hashlib.sha256(data).hexdigest()
        p = self._path(h)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
            with gzip.open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)  # same content under the same name, so a race is harmless
        return {"blob": f"sha256:{h}", "bytes": len(data)}

    def get(self, ref):
        h = (ref["blob"] if isinstance(ref, dict) else ref).split(":", 1)[-1]
        with gzip.open(self._path(h), "rb") as f:
            return json.loads(f.read())


def is_blob_ref(v) -> bool:
    return isinstance(v, dict) and set(v) == {"blob", "bytes"} and str(v["blob"]).startswith("sha256:")


class RunLog:
    """Event writer for one process of one run. Thread-safe (LiteLLM calls success callbacks from a pool)."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None, proc: str = "driver",
                 blob_root: str | Path | None = None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.proc = _slug(proc)
        self.blobs = BlobStore(blob_root if blob_root is not None else self.run_dir.parent / "_blobs")
        self.path = self.run_dir / f"events.{self.proc}.jsonl"
        self._f = open(self.path, "x", encoding="utf-8")
        self._seq = 0
        self._lock = threading.Lock()
        self.pid = os.getpid()
        self.closed = False
        self.counts: dict[str, int] = {}

    @classmethod
    def create(cls, root: str | Path, exp: str, cond: str, study: str, proc: str = "driver",
               run_id: str | None = None) -> "RunLog":
        """New run directory under root (refuses an existing one) and its runs_index.jsonl entry."""
        root = Path(root)
        run_id = run_id or make_run_id(exp, cond, study)
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        log = cls(run_dir, run_id, proc)
        log._index({"event": "start", "exp": exp, "cond": cond, "study": study})
        return log

    @classmethod
    def attach(cls, run_dir: str | Path, proc: str) -> "RunLog":
        """Another process of an existing run (e.g. the MCP server) writes its own events file."""
        return cls(run_dir, Path(run_dir).name, proc)

    def _index(self, entry: dict) -> None:
        line = json.dumps({"run_id": self.run_id, "ts": datetime.now(timezone.utc).isoformat(), **entry})
        fd = os.open(self.run_dir.parent / "runs_index.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (line + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def emit(self, type: str, *, parent_id: str | None = None, step: int | None = None, inline=(),
             **fields) -> str:
        """Write one event and return its span_id ("<proc>:<seq>"). Field values whose canonical JSON exceeds
        INLINE_LIMIT bytes are stored as blobs and replaced by {"blob": "sha256:...", "bytes": n}."""
        if self.closed:
            raise RuntimeError(f"run log {self.path} is closed")
        if type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {type!r}")
        body = {}
        for k, v in fields.items():
            v = jsonable(v)
            if k not in inline and not is_blob_ref(v) and len(canonical_json(v)) > INLINE_LIMIT:
                v = self.blobs.put(v)
            body[k] = v
        with self._lock:
            self._seq += 1
            span_id = f"{self.proc}:{self._seq}"
            ts_ns = time.time_ns()
            rec = {"v": LOG_SCHEMA, "run_id": self.run_id, "proc": self.proc, "pid": self.pid, "seq": self._seq,
                   "ts": datetime.fromtimestamp(ts_ns / 1e9, timezone.utc).isoformat(), "ts_ns": ts_ns,
                   "type": type, "span_id": span_id}
            if parent_id is not None:
                rec["parent_id"] = parent_id
            if step is not None:
                rec["step"] = step
            rec.update(body)
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._f.flush()
            os.fsync(self._f.fileno())
            self.counts[type] = self.counts.get(type, 0) + 1
        return span_id

    def write_file(self, name: str, text: str) -> Path:
        """A new file in the run directory (exclusive create)."""
        p = self.run_dir / name
        with open(p, "x", encoding="utf-8") as f:
            f.write(text)
        return p

    def close(self, status: str = "ok", **summary) -> None:
        if self.closed:
            return
        self.emit("run_end", status=status, event_counts=dict(self.counts), **summary)
        self._f.close()
        self.closed = True
        if self.proc == "driver":
            self._index({"event": "end", "status": status})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(status="error" if exc else "ok", **({"error": f"{exc_type.__name__}: {exc}"} if exc else {}))
        return False


def read_events(run_dir: str | Path) -> list[dict]:
    """All events of a run, merged across processes and ordered by time. A truncated final line (crash
    while writing) is skipped; any other unreadable line raises."""
    events = []
    for p in sorted(Path(run_dir).glob("events.*.jsonl")):
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    continue
                raise
    return sorted(events, key=lambda e: (e["ts_ns"], e["proc"], e["seq"]))
