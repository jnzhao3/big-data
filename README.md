# big-data

Final project for the NYU Big Data course (Group 9). We study how reproducible LLM-based data-harmonization methods and agents are when they run on a local model. The task is to match the columns and values of ten CPTAC clinical tables to the GDC data model. We measure how much repeated runs differ, and why.

The task has two parts, and every result in the repo belongs to one of them:

- **Schema matching (SM):** which GDC column a source column corresponds to, e.g. `Tumor_Site` → `primary_site`. Ground truth: GDC-SM (259 column pairs). Records have `"kind": "sm"`.
- **Value matching (VM):** for a matched column pair, which GDC value each source value corresponds to, e.g. in the `Race` → `race` pair, `White` → `white`. Ground truth: GDC-VM (3,904 values). Records have `"kind": "vm"`.

## Start here

1. **Ollama and the model.** The project is pinned to Ollama 0.40.0, run as the Homebrew service (`env/engine-lock.json`):

   ```bash
   brew install ollama && brew pin ollama      # pin, so Homebrew never upgrades it silently
   brew services start ollama
   ollama --version                           # 0.40.0
   ollama pull qwen2.5:14b                    # about 9 GB
   ```

   If Homebrew gives you a version other than 0.40.0, tell Freya before running experiments. `log validate` reports every run on another engine version as a problem. Everything so far was tested on a MacBook Pro M1 Max with 64 GB.

   These are the Linux versions of the commands:

   ```bash
   curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.0 sh
  sudo systemctl edit ollama    # add the two lines below under [Service]
  #   Environment="OLLAMA_FLASH_ATTENTION=1"
  #   Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
  ollama pull qwen2.5:14b
  ```

2. **Python environment:** `bash env/setup_env.sh` creates `~/venvs/harmonize` (Python 3.12). See [`env/README.md`](env/README.md).

3. **Data:** download it into `datasets/` as [`datasets/README.md`](datasets/README.md) describes ("Re-fetching").

4. **Check the setup**, from the repo root:

   ```bash
   PY=~/venvs/harmonize/bin/python
   $PY -m hbench manifest verify                      # your data matches datasets/MANIFEST.json
   $PY -m hbench engine check --models qwen2.5:14b    # your Ollama matches env/engine-lock.json
   $PY -m pytest -q                                   # all tests pass
   ```

5. **A first run** (about one minute). It should report `"acc_gold_cols": "9/14"` and write its run folder under `~/hbench_raw/`:

   ```bash
   $PY -I experiments/qwen_local/clark_match.py --model qwen2.5:14b --mode shortlist --study clark --run 1 \
       --temperature 0 --seed 1 --engine-protocol cold_start --out-dir ~/hbench_raw/csv
   $PY -m hbench log validate ~/hbench_raw/<run_id>   # the run_id is in the printed summary
   ```

6. **Your own method:** follow the Quick start in [`hbench/README.md`](hbench/README.md).

## What is here

| Folder | What it is | Task |
|---|---|---|
| [`datasets/`](datasets/README.md) | Which benchmark files to use, how to get them, and their pitfalls. The data itself is not in git | SM, VM |
| [`env/`](env/README.md) | The shared Python environment and the pinned inference engine | |
| [`hbench/`](hbench/README.md) | The shared library: prediction format, scoring, run logging and checks. `tests/` holds its tests | SM, VM |
| [`experiments/qwen_local/`](experiments/qwen_local/README.md) | Two methods on local Qwen, `clark_match` and Magneto, with the feasibility study's results | SM |
| [`experiments/e1_noise_floor/`](experiments/e1_noise_floor/README.md) | E1: the engine's noise floor. Stage 1 results, the synthetic tool loop, the run-plan executor and the analyses | SM |
| `notebooks/` | Exploration notebooks and draft scripts | |

The planned experiments, which the run logs are built for:

| | Experiment | Task |
|---|---|---|
| E1 | Engine noise floor: how much the inference engine alone changes outputs | SM |
| E2 | Magneto repeated runs | SM |
| E3 | The BDI-Kit agent end to end (schema and value matching), timed | SM, VM |
| E4 | Repeated agent runs and a decomposition of their variance | SM, VM |
| E5 | Where two runs diverge, and replaying a run from that point | SM, VM |
| E6 | Interventions that stabilize runs (temperature, seeds, voting, caching, ordering) | SM, VM |
| E7 | What the agent says versus what it does | SM, VM |

## Rules for every run

These keep results comparable across runs, methods and people.

- **Environment.** Use `~/venvs/harmonize`.
  - Never add or upgrade packages in it; change versions only in `env/requirements-lock.txt`.
  - Keep `litellm` at 1.82.6: versions 1.82.7 and 1.82.8 were compromised on PyPI.
- **Commands.** Run everything from the repo root.
  - Scripts: `~/venvs/harmonize/bin/python -I <script>`.
  - hbench commands: `~/venvs/harmonize/bin/python -m hbench ...`.
  - BDI-Kit runs that need `PYTHONHASHSEED` use `-s` instead of `-I` (see `env/README.md`).
- **Engine.** Use Ollama 0.40.0 as the Homebrew service, and let nothing else use it while a run is going.
  - hbench's lease keeps two of our runs from overlapping.
  - Any other client shows up in the run's audit as foreign requests, and `log validate` fails the run.
- **Data.** Score against GDC-SM and GDC-VM with `python -m hbench score`, never against `GDC_new/<study>/groundtruth.csv`.
  - `--gt` chooses the schema-matching ground truth only. Value-matching records (`"kind": "vm"`) are always scored against GDC-VM; choose `--vm-mode oracle` if the method was given the attribute pairs, `--vm-mode end_to_end` if it chose them itself (default `oracle`).
  - The output files are still named after `--gt` (`scores.gdc-sm.json`, `items.gdc-sm.csv`), even when they hold value-matching scores.
- **LLM calls.**
  - Set the temperature explicitly: the Ollama default is not zero and differs by endpoint.
  - Log every call, with `hbench.llm.LLMClient` or LiteLLM's `openai/` route plus `hbench.runlog.litellm_cb`. The `ollama_chat/` route cannot return logprobs.
  - Open runs with `start_run`, and check each one with `python -m hbench log validate <run_dir>`.
  - At T>0, state one seed per run (`"per_replicate"` for repeats). hbench sends each sampled call its own seed derived from it, on every route above; `log validate` fails a run whose calls shared one seed (see `hbench/README.md`).
- **Repeated runs.** Use `engine.protocol: "cold_start"`, so every repeat starts from the same engine state. See the Quick start.
- **Outputs.**
  - Each run writes `predictions.jsonl` in the shared format (`hbench/README.md`).
  - Raw run logs go to `$HBENCH_RAW`, by default `~/hbench_raw`, never into the repo. Qwen's logprob candidates in the logs and in `predictions.jsonl` can contain Chinese tokens.
  - Commit summary tables only, and never overwrite an earlier run.
- **BDI-Kit.**
  - Its LLM matchers default to the cloud model `openai/gpt-4o-mini`; point them at Ollama for local runs.
  - It returns cached results for repeated calls: pass `use_cache=False` or clear `~/.cache/bdikit/runs/` before each run.
  - A runner script for the BDI-Kit agent is still in progress and not in the repo yet.
