"""Model-written code: validation, storage and sandboxed execution (option 3: restricted Python, no OS sandbox).

The model saves functions `def run(inputs): ...` and runs them on data it passes in, or on data the server
supplies (source column values, permissible values). Defences, in layers:

1. Static check (AST), before anything runs: imports only of math, re, statistics, unicodedata; no name or
   attribute starting with "_"; no string constant containing "__"; no frame/code attributes; no `.format` /
   `.format_map` (format strings can reach attributes); no dangerous builtins (open, eval, exec, getattr, ...).
2. A child process: `python -I -S`, an empty environment, an empty temporary working directory, CPU, memory,
   file-size and wall-clock limits.
3. Inside it, the code runs with a whitelist of builtins and with proxies of the allowed modules that expose only
   their public, non-module attributes (statistics would otherwise expose sys).

This is not a security boundary in the strict sense: restricted Python has a history of escapes, and the child
runs as the same user. It keeps an honest model from reaching files, the network or the ground truth by
accident or by casual attempt. An OS sandbox (bubblewrap) would be the stronger option.
"""

from __future__ import annotations

import ast
import json
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = Path(__file__).with_name("codebox_runner.py")
ALLOWED_MODULES = ("math", "re", "statistics", "unicodedata")
DENIED_NAMES = {"open", "eval", "exec", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
                "dir", "input", "breakpoint", "help", "memoryview", "object", "type", "super", "classmethod",
                "staticmethod", "property", "exit", "quit", "license", "credits", "copyright", "id", "hash",
                "__import__", "__builtins__", "__loader__", "__spec__"}
DENIED_ATTRS = {"format", "format_map", "mro", "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code",
                "cr_await", "ag_frame", "ag_code", "ag_await", "f_globals", "f_locals", "f_builtins", "f_back",
                "f_code", "tb_frame", "tb_next", "co_code", "func_globals", "func_code", "im_func", "im_self"}
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
MAX_CODE_CHARS = 20_000
TIMEOUT_S = 10
CPU_S = 5
MEM_BYTES = 768 * 1024 * 1024
MAX_OUTPUT_BYTES = 2_000_000


class CodeError(ValueError):
    pass


def validate(code: str) -> None:
    """Raise CodeError if the code uses anything outside the allowed subset, or has no run(inputs)."""
    if not isinstance(code, str) or not code.strip():
        raise CodeError("code is empty")
    if len(code) > MAX_CODE_CHARS:
        raise CodeError(f"code is longer than {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeError(f"syntax error: {e.msg} (line {e.lineno})") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_MODULES:
                    raise CodeError(f"import of {a.name!r} is not allowed; allowed: {', '.join(ALLOWED_MODULES)}")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_MODULES or node.level:
                raise CodeError(f"import from {node.module!r} is not allowed; allowed: {', '.join(ALLOWED_MODULES)}")
            for a in node.names:
                if a.name == "*" or a.name.startswith("_"):
                    raise CodeError("star imports and private names are not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in DENIED_NAMES:
                raise CodeError(f"name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in DENIED_ATTRS:
                raise CodeError(f"attribute {node.attr!r} is not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                raise CodeError(f"name {node.name!r} is not allowed")
        elif isinstance(node, ast.arg):
            if node.arg.startswith("_"):
                raise CodeError(f"argument name {node.arg!r} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise CodeError("string constants containing '__' are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            pass
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" and len(n.args.args) == 1 for n in tree.body):
        raise CodeError("define a top-level function run(inputs)")


def _limits():  # runs in the child before exec
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_S, CPU_S))
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes


def execute(code: str, inputs, data: dict) -> dict:
    """Run validated code in a child process. data: {"values": {column: [...]}, "permissible": {attr: [...]}}.
    Returns {"ok": bool, "result": ..., "stdout": str, "error": str | None}."""
    validate(code)
    payload = json.dumps({"code": code, "inputs": inputs, "data": data, "modules": ALLOWED_MODULES})
    with tempfile.TemporaryDirectory(prefix="cvm_code_") as cwd:
        try:
            p = subprocess.run([sys.executable, "-I", "-S", str(RUNNER)], input=payload, capture_output=True,
                               text=True, cwd=cwd, env={}, timeout=TIMEOUT_S, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return {"ok": False, "result": None, "stdout": "", "error": f"timed out after {TIMEOUT_S} s"}
    if len(p.stdout) > MAX_OUTPUT_BYTES:
        return {"ok": False, "result": None, "stdout": "", "error": "output larger than 2 MB"}
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        err = (p.stderr or "").strip().splitlines()[-1:] or [f"exit code {p.returncode}"]
        if p.returncode < 0 or "MemoryError" in (p.stderr or ""):
            err = [f"stopped by a resource limit ({err[0]})"]
        return {"ok": False, "result": None, "stdout": "", "error": err[0]}
    return out


class CodeStore:
    """Saved functions of one session: <run_dir>/code/<name>_v<k>.py plus index.json."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.index_path = self.dir / "index.json"

    def index(self) -> dict:
        return json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def save(self, name: str, code: str, description: str, ts: str) -> dict:
        if not NAME_RE.match(name or ""):
            raise CodeError("name must start with a letter and use letters, digits and _ (max 64)")
        validate(code)
        self.dir.mkdir(parents=True, exist_ok=True)
        idx = self.index()
        versions = idx.setdefault(name, [])
        k = len(versions) + 1
        path = self.dir / f"{name}_v{k}.py"
        path.write_text(code)
        versions.append({"version": k, "file": path.name, "description": description or "", "saved_at": ts})
        self.index_path.write_text(json.dumps(idx, indent=1) + "\n")
        return {"name": name, "version": k}

    def load(self, name: str, version: int | None = None) -> tuple[str, int]:
        versions = self.index().get(name)
        if not versions:
            raise CodeError(f"no saved code named {name!r}; list_code shows what is saved")
        v = versions[-1] if version is None else next((x for x in versions if x["version"] == version), None)
        if v is None:
            raise CodeError(f"{name!r} has no version {version}")
        return (self.dir / v["file"]).read_text(), v["version"]
