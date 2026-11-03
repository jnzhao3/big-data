from pathlib import Path

import pytest

from hbench.data import REPO_ROOT, Data

RESULTS = REPO_ROOT / "experiments" / "qwen_local" / "results"


@pytest.fixture(scope="session")
def data():
    """Benchmark data ($HBENCH_DATA or <repo>/datasets). Tests that need it are skipped when it is absent."""
    d = Data()
    if not d.exists():
        pytest.skip(f"benchmark data not found at {d.root}; set HBENCH_DATA (see datasets/README.md)")
    return d


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return RESULTS
