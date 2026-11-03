#!/usr/bin/env bash
# Build the shared harmonization environment: BDI-Kit 0.11.0 stack (incl. Magneto via
# magneto-python 0.2.0), LangGraph/MCP agent stack, and test tools, all at locked versions.
#
# Usage: bash setup_env.sh [venv_dir]      default venv_dir: ~/venvs/harmonize
# Requires Python 3.12 (macOS: brew install python@3.12). Override the interpreter with PYTHON=...
set -euo pipefail

VENV="${1:-$HOME/venvs/harmonize}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3.12}"

case "$VENV" in
  *" "*) echo "venv path must not contain spaces (console-script shebangs and agent file paths break): $VENV" >&2; exit 1 ;;
esac
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), f"need Python 3.12, got {sys.version}"'

"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --require-virtualenv "pip==26.3"
"$VENV/bin/python" -m pip install --require-virtualenv -r "$HERE/requirements-lock.txt"
"$VENV/bin/python" -m pip check

"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
import pathlib
import site
import sys

# litellm 1.82.7 and 1.82.8 were compromised on PyPI; bdi-kit 0.11.0 pins 1.82.6
assert md.version("litellm") == "1.82.6", md.version("litellm")
pth = [p for d in site.getsitepackages() for p in pathlib.Path(d).glob("*.pth")]
bad = [p for p in pth if "litellm" in p.name.lower()]
assert not bad, f"unexpected .pth files: {bad}"
print("ok: python", sys.version.split()[0], "| bdi-kit", md.version("bdi-kit"), "| litellm", md.version("litellm"),
      "| magneto-python", md.version("magneto-python"), "| .pth files:", sorted(p.name for p in pth))
EOF
