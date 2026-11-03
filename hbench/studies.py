"""Canonical study ids and the spellings used by the different benchmark files."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Study:
    id: str           # canonical id used in item ids, run ids and headers: "gillette"
    name: str         # GDC-SM / GDC-VM file stem: "Gillette"
    gdc_new_dir: str  # folder under GDC_new: "gilette" (misspelled upstream)


_NAMES = ["Cao", "Clark", "Dou", "Gillette", "Huang", "Krug", "McDermott", "Satpathy", "Vasaikar", "Wang"]
STUDIES: dict[str, Study] = {
    n.lower(): Study(n.lower(), n, "gilette" if n == "Gillette" else n.lower()) for n in _NAMES
}
_ALIASES = {"gilette": "gillette", "gillete": "gillette", "gilete": "gillette"}


def study_id(name: str) -> str:
    """'Gillette', 'gilette', ' GILLETTE ' or 'gillete' -> 'gillette'. Unknown names raise KeyError."""
    key = re.sub(r"[\s_\-]+", "", str(name)).casefold()
    key = _ALIASES.get(key, key)
    if key not in STUDIES:
        raise KeyError(f"unknown study {name!r}; known: {sorted(STUDIES)}")
    return key


def get_study(name: str) -> Study:
    return STUDIES[study_id(name)]
