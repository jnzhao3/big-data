# big-data

Final project for the NYU Big Data course (Group 9).

## Progress

As of 2026-11-02:

| Component | Location | Owner |
|---|---|---|
| Benchmark data selection and documentation | [`datasets/`](datasets/README.md) | Freya |
| Shared Python environment | [`env/`](env/README.md) | Freya |
| Feasibility study of local LLMs via Ollama (recommended: `qwen2.5:14b`) | [`experiments/qwen_local/`](experiments/qwen_local/README.md) | Freya |
| Shared scoring, prediction format and run logging (`hbench`) | [PR #1](https://github.com/jnzhao3/big-data/pull/1) (under review) | Freya |

## Conventions

All experiments follow these conventions so that results are comparable.

### Environment

Use the shared environment in `env/` (Python 3.12):

```bash
bash env/setup_env.sh                  # creates ~/venvs/harmonize
~/venvs/harmonize/bin/python <script>
```

- Do not upgrade or add individual packages; change versions only in `env/requirements-lock.txt`.
- Keep `litellm` at 1.82.6; versions 1.82.7 and 1.82.8 were compromised on PyPI.
- Pipelines with conflicting dependencies may use a separate environment but must write predictions in the shared format.
- The lock file has been tested on macOS (arm64) only.

### Data

Data is not tracked in git; see [`datasets/README.md`](datasets/README.md) for download instructions.

| Purpose | Source |
|---|---|
| Column-level ground truth | GDC-SM v1.0 (Zenodo 14963588), 259 pairs |
| Value-level ground truth | GDC-VM (Zenodo 18557064) |
| Source tables | `GDC_new/<study>/source.csv` from [VIDA-NYU/data-harmonization-benchmark](https://github.com/VIDA-NYU/data-harmonization-benchmark) |

Do not score against `GDC_new/<study>/groundtruth.csv`; GDC-SM supersedes it.

### Predictions and run logs

Each run writes `predictions.jsonl`, one decision per line, in the `hbench.pred/1` format (under review in [PR #1](https://github.com/jnzhao3/big-data/pull/1)):

```json
{"schema": "hbench.pred/1", "run_id": "...", "kind": "sm", "study": "clark", "source_column": "Tumor_Site", "output_type": "ranked", "targets": ["site_of_resection_or_biopsy", "primary_site"], "shown": "*", "status": "ok"}
```

- `targets: []` denotes no match. `shown` lists the candidates presented to the method; `"*"` denotes all 736 GDC columns.
- Value-matching records use `"kind": "vm"` with `source_attribute`, `target_attribute` and `source_value`. See `hbench/README.md` for the full specification.
- Raw logs go to `experiments/<experiment>/results/raw/`, which is not tracked; commit only summaries. Give each run a unique `run_id` and never overwrite earlier runs.

To validate and score a run:

```bash
python -m hbench preds validate predictions.jsonl
python -m hbench score predictions.jsonl --gt gdc-sm
```

### LLM calls

- Set the temperature explicitly in every call; the Ollama default is not zero.
- BDI-Kit's LLM matchers default to the cloud model `openai/gpt-4o-mini`; specify an Ollama model for local runs.
- BDI-Kit returns cached results for repeated calls. Pass `use_cache=False` or clear `~/.cache/bdikit/runs/` before each run.
