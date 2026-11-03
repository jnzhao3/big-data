"""Fingerprints of the benchmark files every score depends on (datasets/MANIFEST.json).

The data itself is not in git, so the manifest is what tells two machines (or two months) apart.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .studies import STUDIES

MANIFEST_VERSION = 1
# Tracked in git (the data next to it is not); verified against whatever data root is in use.
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "datasets" / "MANIFEST.json"
SOURCES = {
    "GDC_new": "VIDA-NYU/data-harmonization-benchmark main@3207e37, datasets/GDC_new.tar.gz",
    "gdc_sm": "Zenodo 14963588 (GDC-SM v1.0, 2025-04)",
    "gdc_vm": "Zenodo 18557064 (GDC-VM, 2026-02-09)",
}


def manifest_files() -> list[str]:
    files = []
    for s in STUDIES.values():
        files += [f"GDC_new/{s.gdc_new_dir}/source.csv", f"GDC_new/{s.gdc_new_dir}/target.csv",
                  f"GDC_new/{s.gdc_new_dir}/groundtruth.csv",
                  f"gdc_sm/extracted/data/ground-truth/{s.name}.csv", f"gdc_vm/{s.name}.csv"]
    files += ["gdc_sm/extracted/data/target-tables/gdc_unique_columns_concat_values.csv", "gdc_sm/gdc-sm-data.zip",
              "gdc_vm/gdc_schema.json", "gdc_vm/gdc_schema_flattened.csv"]
    return files


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha256(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def build_manifest(root: str | Path) -> dict:
    root = Path(root)
    files = {rel: {"bytes": (root / rel).stat().st_size, "md5": md5_file(root / rel)} for rel in manifest_files()}
    return {"manifest_version": MANIFEST_VERSION, "sources": SOURCES, "files": files,
            "manifest_sha256": manifest_sha256(files)}


def verify_manifest(root: str | Path, manifest: dict | str | Path) -> list[str]:
    """Problems found (missing or changed files); an empty list means the data matches the manifest."""
    root = Path(root)
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    problems = []
    for rel, want in manifest["files"].items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif p.stat().st_size != want["bytes"] or md5_file(p) != want["md5"]:
            problems.append(f"changed: {rel}")
    return problems
