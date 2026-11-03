# Local Qwen methods: clark_match and Magneto

Two schema-matching methods that run on local Qwen through Ollama. The feasibility study that chose the model is in [`RESULTS.md`](RESULTS.md). It chose `qwen2.5:14b`: about as accurate as the 27B model, 3-6x faster, and tool calling works.

Setup (Ollama, the model, the environment and the data) is in the repo [README](../../README.md). Run everything from the repo root, with `PY=~/venvs/harmonize/bin/python`.

## clark_match.py

For each source column of a study, the model picks one GDC target column and answers in JSON. Two candidate modes:
- `shortlist` (default): 20 candidates, by column-name similarity and value overlap, shown with sample values;
- `full`: all 736 target column names, without sample values.

```bash
$PY -I experiments/qwen_local/clark_match.py --model qwen2.5:14b --mode shortlist --study clark --run 1 \
    --temperature 0 --seed 1 --engine-protocol cold_start --out-dir ~/hbench_raw/csv
```

- `--run N` numbers the repeats of one condition. With `--engine-protocol cold_start`, every repeat starts from the same engine state.
- `--study` takes any of the ten studies. `--gt old` scores against the old ground truth instead of GDC-SM.
- `--transport openai_v1` sends the same requests through `/v1`.
- Outputs:
  - `<tag>.csv` and a line in `summary.jsonl` go to `--out-dir`. An existing `<tag>.csv` is never overwritten.
  - The run folder goes to `--raw-dir` (default `$HBENCH_RAW` or `~/hbench_raw`). It holds the run log, `predictions.jsonl` with each decision's evidence, the scores, and `engine_audit.json`.
- Keep `--out-dir` outside `results/`, which holds the study's committed outputs.
- `--rescore <folder> --out-dir <new folder>` rescores the CSV of an earlier run in `<folder>` without calling the model.

## magneto_ollama.py

Magneto (VIDA-NYU) with its LLM reranker on local Qwen. The reranker goes through LiteLLM's `openai/` route, which returns logprobs.

```bash
$PY -I experiments/qwen_local/magneto_ollama.py                # reranker: openai/qwen2.5:14b at T=0
$PY -I experiments/qwen_local/magneto_ollama.py --no-llm       # embeddings and the bipartite reranker only
```

- `--engine-protocol cold_start` for repeats; `--study`, `--run` and `--seed` work as for clark_match.
- The first run downloads Magneto's embedding model, `sentence-transformers/all-mpnet-base-v2`, into the Hugging Face cache.

## Expected results

On the pinned engine (Ollama 0.40.0, set up as in the repo README), these runs on clark should reproduce:

| Run | GDC-SM top-1 | MRR | Time on an M1 Max |
|---|---|---|---|
| clark_match shortlist, T=0, seed 1 | 9/14 | | about 70 s |
| Magneto `--no-llm` | 7/14 | 0.591 | about 15 s |
| Magneto with the qwen2.5:14b reranker, T=0 | 9/14 | 0.708 | about 400 s |

## Other files

| File | Purpose |
|---|---|
| `run_all.sh` | Reproduces the study's clark runs: 2 models x 2 modes x 2 runs, back to back, at the model's default temperature. For new experiments, set `--temperature` and `--engine-protocol cold_start` instead |
| `bench_speed.sh`, `test_openai_tools.py` | The study's speed and tool-calling checks |
| `results/` | The study's outputs; `results/gdc_sm/` holds them rescored against GDC-SM. See `RESULTS.md` |

## For agent experiments

From the study's tool-calling check:
- **Temperature.** Use temperature 0 unless run-to-run variation is what you measure. At Ollama's default temperature, qwen2.5:14b occasionally starts a reply in Thai or Chinese.
- **Tool-call history.** Before sending an assistant message that has tool calls back in the history, drop its text. Ollama's qwen2.5 template renders only the text of such a message, so the tool call would disappear from the history.
