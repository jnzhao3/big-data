"""Shared code of the Claude Code value-matching harness (experiments/claude_vm).

Modules that the hook scripts and the viewer import (paths, timeline) use the standard library only; the
others use hbench and the repo environment.
"""

import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = HARNESS_DIR.parents[1]

# Always use this checkout's hbench, as the other experiment scripts do.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
