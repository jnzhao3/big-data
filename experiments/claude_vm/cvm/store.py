"""Submissions and schema decisions of one session, as an append-only JSONL log.

Every accepted submission, NONE decision and attribute (un)mapping is one line, so the whole history is kept.
The current state is the last line per key: re-submitting an item replaces the earlier answer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .study import REPO_ROOT  # noqa: F401
from hbench.normalize import canon_value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def item_key(source_attribute: str, target_attribute: str, source_value) -> tuple[str, str, str]:
    return (source_attribute, target_attribute, canon_value(source_value))


class SubmissionStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------ writes

    def add_match(self, source_attribute, target_attribute, source_value, target_value, confidence, rationale):
        self._append({"ts": now_iso(), "event": "match", "source_attribute": source_attribute,
                      "target_attribute": target_attribute, "source_value": canon_value(source_value),
                      "target_value": target_value, "confidence": confidence, "rationale": rationale})

    def set_mapping(self, source_attribute: str, target_attribute: str, mapped: bool):
        self._append({"ts": now_iso(), "event": "map" if mapped else "unmap",
                      "source_attribute": source_attribute, "target_attribute": target_attribute})

    # ------------------------------------------------------------------ state

    def matches(self) -> dict[tuple, dict]:
        """Item key -> its latest submission (target_value None means an explicit NONE)."""
        state = {}
        for e in self.events():
            if e.get("event") == "match":
                state[item_key(e["source_attribute"], e["target_attribute"], e["source_value"])] = e
        return state

    def mapped_pairs(self) -> list[tuple[str, str]]:
        """End-to-end mode: the attribute pairs mapped at the end, in the order first mapped."""
        state: dict[tuple, bool] = {}
        for e in self.events():
            if e.get("event") in ("map", "unmap"):
                state[(e["source_attribute"], e["target_attribute"])] = e["event"] == "map"
        return [p for p, on in state.items() if on]
