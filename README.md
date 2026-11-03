# big-data

NYU Big Data final project, Group 9.

## Done so far (Freya, as of 2026-11-02)

| What | Where |
|---|---|
| Benchmark data: which GDC files to use, their versions, and pitfalls | [`datasets/README.md`](datasets/README.md) |
| One shared, locked Python environment for every experiment | [`env/`](env/README.md) |
| Local Qwen test: `qwen2.5:14b` on Ollama works for column matching and tool calling, and as Magneto's reranker | [`experiments/qwen_local/`](experiments/qwen_local/README.md) |
| `hbench`: one scorer, prediction format and run log for both tracks (under review) | [PR #1](https://github.com/jnzhao3/big-data/pull/1) |

## What everyone should follow

### Environment

```bash
bash env/setup_env.sh                  # creates ~/venvs/harmonize; needs Python 3.12 (brew install python@3.12)
~/venvs/harmonize/bin/python <script>
```

- Do not upgrade or add single packages. Never move `litellm` off 1.82.6: 1.82.7 and 1.82.8 were compromised on PyPI. Version changes go into `env/requirements-lock.txt` for everyone.
- If your pipeline needs conflicting packages, use a separate venv, but still write predictions in the format below.
- The lock was built on macOS arm64. Linux and HPC are untested.

### Data

- The data is not in git. Download it into `datasets/` with the commands in [`datasets/README.md`](datasets/README.md).
- Column-level ground truth: GDC-SM (Zenodo 14963588, 259 pairs). Value-level: GDC-VM (Zenodo 18557064). Source tables: `GDC_new/<study>/source.csv` from [VIDA-NYU/data-harmonization-benchmark](https://github.com/VIDA-NYU/data-harmonization-benchmark).
- Do not score against `GDC_new/<study>/groundtruth.csv`. It is an older ground truth without the 1:n pairs.

### Predictions and run logs (proposed in [PR #1](https://github.com/jnzhao3/big-data/pull/1), please review)

Write one decision per line to `predictions.jsonl`, so every method and both tracks are scored by the same code:

```json
{"schema": "hbench.pred/1", "run_id": "...", "kind": "sm", "study": "clark", "source_column": "Tumor_Site", "output_type": "ranked", "targets": ["site_of_resection_or_biopsy", "primary_site"], "shown": "*", "status": "ok"}
```

- `targets: []` means "no match". `shown` lists the candidates the method saw (`"*"` for all 736 GDC columns).
- Value matching uses `"kind": "vm"` with `source_attribute`, `target_attribute` and `source_value`. All fields and scoring rules are in `hbench/README.md`.
- Check, then score: `python -m hbench preds validate predictions.jsonl`, then `python -m hbench score predictions.jsonl --gt gdc-sm`.
- Raw run logs go under `experiments/<exp>/results/raw/`, which git ignores. Commit only small summaries. Give every run a new `run_id` and never overwrite an earlier run.

### LLM runs

- Set the temperature explicitly in every call; Ollama's default is not 0.
- BDI-Kit's LLM matchers default to the cloud model `openai/gpt-4o-mini`. For local runs, point LiteLLM at Ollama.
- BDI-Kit caches match results in `~/.cache/bdikit/runs/` and returns them for repeated calls. Pass `use_cache=False`, or empty that folder before each run.
