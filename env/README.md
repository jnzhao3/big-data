# Shared Python environment

Everyone on the team uses this one environment for every experiment: E0 scoring, `clark_match`, Magneto, and the BDI-Kit tools and agent. That way, differences between runs come from the methods, not from package versions.

## Install

The environment needs Python 3.12. On macOS, install it with `brew install python@3.12`.

```bash
bash env/setup_env.sh            # creates ~/venvs/harmonize
bash env/setup_env.sh <dir>      # or a different location; the path must not contain spaces
```

The script does six things:
1. creates a venv;
2. installs the exact versions in `requirements-lock.txt`;
3. installs this repo's `hbench` package in editable mode, so it follows your checkout;
4. runs `pip check`;
5. confirms that `litellm` is 1.82.6;
6. checks that no unexpected `.pth` file was installed.

Run scripts with `~/venvs/harmonize/bin/python <script>`.

If the environment already exists, add `hbench` without rebuilding by running this from the repo root:

```bash
~/venvs/harmonize/bin/python -m pip install --no-deps --no-build-isolation -e .
```

`hbench` is the only package outside the lock, and the scripts still work without it, because they fall back to the repo copy.

## Rules

- **Do not upgrade or add single packages in this environment.** In particular, never move `litellm` off 1.82.6: versions 1.82.7 and 1.82.8 were compromised on PyPI, and bdi-kit 0.11.0 pins 1.82.6. To change any version, update `requirements-lock.txt` for everyone, rebuild, and rerun the checks below.
- **Do not run with `python -I` if you need `PYTHONHASHSEED`.** `-I` ignores every `PYTHON*` environment variable. Use `python -s` instead.
- **Magneto is the PyPI package `magneto-python` 0.2.0, which bdi-kit installs.** You do not need a separate clone or venv: with the default `mpnet` model it gives the same results as the GitHub version.
- **If your own pipeline needs packages that conflict with the lock, use a separate venv.** You still have to write predictions in the shared format. The scorer only needs `pandas` and `numpy`.
- **The lock was generated on macOS arm64.** It has not been tested on Linux or HPC yet; report any install problems there.

## Checks done on 2026-11-02

| Check | Result |
|---|---|
| `pip check` | clean |
| Rebuild with `setup_env.sh` | the rebuilt environment has a `pip freeze` identical to the original |
| Magneto on clark without an LLM | identical per column to earlier runs |
| Magneto on clark with the qwen2.5:14b reranker at T=0, run twice | identical per column to earlier runs |
| BDI-Kit `tfidf` and `edit_distance` on all GDC-VM attribute pairs | match content unchanged |
| BDI-Kit MCP server | starts and lists 10 tools |
