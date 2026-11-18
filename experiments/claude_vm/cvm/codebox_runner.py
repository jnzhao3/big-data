"""Child-process side of codebox: runs one validated function. Started as `python -I -S codebox_runner.py`
with an empty environment, in an empty temporary directory, under resource limits. Reads the job as JSON on
stdin and writes {"ok", "result", "stdout", "error"} as JSON on stdout. Standard library only."""

import importlib
import json
import sys
import types


def _proxy(mod):
    """A namespace with the module's public, non-module attributes only (statistics would expose sys)."""
    ns = types.SimpleNamespace()
    for k in dir(mod):
        if k.startswith("_"):
            continue
        v = getattr(mod, k)
        if isinstance(v, types.ModuleType):
            continue
        setattr(ns, k, v)
    return ns


def main():
    job = json.loads(sys.stdin.read())
    allowed = {m: _proxy(importlib.import_module(m)) for m in job["modules"]}
    printed = []

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in allowed or level:
            raise ImportError(f"import of {name!r} is not allowed")
        return allowed[name]

    def safe_print(*args, sep=" ", end="\n"):
        if sum(len(x) for x in printed) < 20000:
            printed.append(sep.join(str(a) for a in args) + end)

    data = job.get("data") or {}

    def values(column):
        """Distinct source values of a column the call requested (columns=[...]); '' is the empty cell."""
        if column not in (data.get("values") or {}):
            raise KeyError(f"column {column!r} was not requested in run_code(columns=[...])")
        return list(data["values"][column])

    def permissible(attribute):
        """Permissible values of an attribute the call requested (attributes=[...])."""
        if attribute not in (data.get("permissible") or {}):
            raise KeyError(f"attribute {attribute!r} was not requested in run_code(attributes=[...])")
        return list(data["permissible"][attribute])

    safe = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in (
        "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int",
        "isinstance", "len", "list", "map", "max", "min", "ord", "pow", "range", "repr", "reversed", "round",
        "set", "slice", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "KeyError",
        "IndexError", "TypeError", "ZeroDivisionError", "ArithmeticError", "StopIteration", "True", "False",
        "None") if n not in ("True", "False", "None")}
    safe.update({"print": safe_print, "__import__": safe_import})
    env = {"__builtins__": safe, "values": values, "permissible": permissible}
    try:
        exec(compile(job["code"], "<model code>", "exec"), env)
        result = env["run"](job.get("inputs"))
        out = {"ok": True, "result": result, "stdout": "".join(printed), "error": None}
        text = json.dumps(out, ensure_ascii=False, default=str)
    except BaseException as e:  # noqa: BLE001 (report every failure to the model)
        text = json.dumps({"ok": False, "result": None, "stdout": "".join(printed),
                           "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False, default=str)
    sys.stdout.write(text)


main()
