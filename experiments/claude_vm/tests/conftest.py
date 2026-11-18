"""Fixtures: a tiny dataset in hbench's layout (study 'clark'), small enough to reason about by hand.

Source columns Race and Age. Ground truth (GDC-VM layout) covers:
  - a case variant:        WHITE -> white, White -> white
  - a two-member gold set: '' (empty cell) -> {Unknown, not reported}
  - an empty gold set:     Klingon -> (no GDC value; only NONE is correct)
  - an age conversion:     Age in years -> days
  - ground-truth-only sentinels that no tool may ever reveal:
      a source value that is not in the table (GTONLYSENTINEL), and the age gold numbers (10957.5, ...)
"""

import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

from cvm import REPO_ROOT  # noqa: E402,F401
from cvm.store import SubmissionStore  # noqa: E402
from cvm.study import StudyView  # noqa: E402
from cvm.tools import MatcherTools  # noqa: E402
from hbench.data import Data  # noqa: E402

SOURCE = """Case,Race,Age
C1,White,30
C2,WHITE,41
C3,Asian,55
C4,,30
C5,Klingon,41
C6,White,55
"""

GT = """source_attribute,target_attribute,source_value,target_value
Race,race,White,white
Race,race,WHITE,white
Race,race,Asian,asian
Race,race,,Unknown
Race,race,,not reported
Race,race,Klingon,
Race,race,GTONLYSENTINEL,white
Age,age_at_diagnosis,30,10957.5
Age,age_at_diagnosis,41,14975.25
Age,age_at_diagnosis,55,20088.75
"""

SCHEMA = {
    "race": {"column_description": "Self-reported race.",
             "value_data": {"white": "White race", "asian": "Asian race", "Unknown": "", "not reported": ""}},
    "age_at_diagnosis": {"column_description": "Age at the time of diagnosis expressed in number of days since birth.",
                         "value_data": {}},
    "gender": {"column_description": "Gender of the patient.", "value_data": {"female": "", "male": ""}},
}

GT_SENTINELS = ["GTONLYSENTINEL", "10957.5", "14975.25", "20088.75"]


@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "datasets"
    (root / "GDC_new" / "clark").mkdir(parents=True)
    (root / "GDC_new" / "clark" / "source.csv").write_text(SOURCE)
    (root / "gdc_vm").mkdir()
    (root / "gdc_vm" / "Clark.csv").write_text(GT)
    (root / "gdc_vm" / "gdc_schema.json").write_text(json.dumps(SCHEMA))
    return root


@pytest.fixture
def data(data_root) -> Data:
    return Data(data_root)


@pytest.fixture
def make_tools(data, tmp_path):
    def make(mode="oracle", cap=50, sql=None):
        view = StudyView(data, "clark", mode)
        store = SubmissionStore(tmp_path / f"subs_{mode}.jsonl")
        return MatcherTools(view, store, cap=cap, sql=sql)
    return make
