"""Model-written code: the restricted interpreter, the code tools, and ground-truth isolation."""

import json

import pytest

from conftest import GT_SENTINELS
from cvm.codebox import CodeError, CodeStore, execute, validate
from cvm.store import SubmissionStore
from cvm.study import StudyView
from cvm.tools import MatcherTools, ToolError

AGES = '''
import math
def run(inputs):
    out = {}
    for y in values("Age"):
        out[y] = math.floor(float(y) * 365.25 + 0.5)
    print("converted", len(out))
    return out
'''


@pytest.fixture
def code_tools(data, tmp_path):
    def make(search_only=False):
        view = StudyView(data, "clark", "oracle")
        return MatcherTools(view, SubmissionStore(tmp_path / "s.jsonl"), cap=2, search_only=search_only,
                            code=CodeStore(tmp_path / "code"))
    return make


@pytest.mark.parametrize("code", [
    "import os\ndef run(i): return 1",
    "import subprocess\ndef run(i): return 1",
    "from pathlib import Path\ndef run(i): return 1",
    "from math import *\ndef run(i): return 1",
    "def run(i): return open('/etc/hostname').read()",
    "def run(i): return eval('1')",
    "def run(i): return ().__class__",
    "def run(i): return getattr(1, 'real')",
    "def run(i): return '{0.__class__}'.format(1)",
    "def run(i): return 'a__b'",
    "def run(i): return type(1)",
    "def run(i):\n    g = (x for x in [1])\n    return g.gi_frame",
    "def run(_i): return 1",
    "def go(i): return 1",
    "def run(i) return 1",
])
def test_validator_rejects(code):
    with pytest.raises(CodeError):
        validate(code)


def test_execution_limits_and_proxies():
    assert execute("import statistics\ndef run(i): return statistics.sys", None, {})["ok"] is False
    loop = execute("def run(i):\n    while True:\n        pass", None, {})
    assert loop["ok"] is False and ("limit" in loop["error"] or "timed out" in loop["error"])
    assert execute("def run(i): return 'x' * (2 ** 31)", None, {})["ok"] is False
    r = execute("import re\ndef run(i): return re.sub('a', 'b', i)", "banana", {})
    assert r == {"ok": True, "result": "bbnbnb", "stdout": "", "error": None}


def test_code_tools(code_tools):
    t = code_tools()
    assert t.save_code("ages", AGES, "years to days")["version"] == 1
    assert t.save_code("ages", AGES.replace("0.5", "0.4999"), "v2")["version"] == 2
    r = t.run_code("ages", columns=["Age"], version=1)
    assert r["total"] == 3 and r["returned"] == 2 and r["next_offset"] == 2 and r["stdout"] == "converted 3\n"
    rest = t.run_code("ages", columns=["Age"], version=1, offset=2)
    assert {**r["result"], **rest["result"]} == {"30": 10958, "41": 14975, "55": 20089}
    assert t.list_code()["functions"] == [{"name": "ages", "versions": 2, "latest_description": "v2"}]
    t.save_code("perm", "def run(i): return permissible('race')", "")
    first = t.run_code("perm", attributes=["race"], limit=10)        # capped at the row cap (2)
    second = t.run_code("perm", attributes=["race"], offset=first["next_offset"])
    assert first["total"] == 4 and first["returned"] == 2
    assert first["result"] + second["result"] == ["white", "asian", "Unknown", "not reported"]
    t.save_code("boom", "def run(i): return values('Race')", "")
    with pytest.raises(ToolError, match="not requested"):
        t.run_code("boom")
    with pytest.raises(ToolError):
        t.save_code("bad", "import os\ndef run(i): return 1")
    with pytest.raises(ToolError):
        t.save_code("1bad", AGES)
    with pytest.raises(ToolError, match="no saved code"):
        t.run_code("missing")


def test_code_tools_respect_modes(code_tools, make_tools):
    t = code_tools(search_only=True)
    t.save_code("perm", "def run(i): return permissible('race')", "")
    with pytest.raises(ToolError, match="search-only"):
        t.run_code("perm", attributes=["race"])
    with pytest.raises(ToolError, match="disabled"):
        make_tools().save_code("x", AGES)


def test_code_cannot_reach_ground_truth(code_tools, data_root):
    """Code tries every route we know to the ground-truth file; no sentinel may come back."""
    t = code_tools()
    gt = str(data_root / "gdc_vm" / "Clark.csv")
    attempts = {
        "open": f"def run(i): return open({gt!r}).read()",
        "import os": f"import os\ndef run(i): return os.listdir({gt!r})",
        "re module": f"import re\ndef run(i): return str(re.compile({gt!r}))",
        "statistics.sys": "import statistics\ndef run(i): return str(statistics.sys.modules)",
        "everything requested": "def run(i): return [values('Race'), values('Age'), permissible('race')]",
    }
    out = []
    for name, code in attempts.items():
        try:
            t.save_code(name.replace(" ", "_").replace(".", "_"), code)
            out.append(json.dumps(t.run_code(name.replace(" ", "_").replace(".", "_"), columns=["Race", "Age"],
                                             attributes=["race"], limit=2)))
        except ToolError as e:
            out.append(str(e))
    text = "\n".join(out)
    for s in GT_SENTINELS:
        assert s not in text
