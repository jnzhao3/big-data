# Llama on local Ollama: schema-matching sweeps

Scripts that run the two existing schema-matching methods, `clark_match` and Magneto (both in [`../qwen_local/`](../qwen_local/README.md)), with `llama3.1:8b` on all ten GDC studies. Settings are fixed in the scripts, so a sweep can be repeated exactly: T=0, seed 1, and `cold_start` (each run starts from an unloaded model).

Before running: `ollama pull llama3.1:8b`, and nothing else using Ollama while a sweep runs.

```bash
bash experiments/llama_local/run_clark_match.sh      # 10 studies x 2 modes x 2 runs
bash experiments/llama_local/run_magneto.sh          # 10 studies x 1 run, slower
~/venvs/harmonize/bin/python -I experiments/llama_local/summarize.py
```

- Everything goes to `~/hbench_raw/llama_eval/`, outside the repo:
  - `summary.jsonl` and per-run CSVs (clark_match);
  - `magneto/` (Magneto's printed scores);
  - `raw/` (hbench run folders: every LLM call, predictions, scores, engine audit);
  - `run_*.log` (the full console output, with the repo commit and Ollama version).
- An interrupted sweep continues where it stopped when re-run: finished runs are skipped.
- Override settings with environment variables, e.g. `MODEL=qwen2.5:14b OUT=~/hbench_raw/qwen_eval bash ...`, or `STUDIES=clark RUNS=1` for a quick check.
- `summarize.py` prints accuracy per study and mode, candidate recall, and how many columns changed between repeats. At T=0 with `cold_start`, repeats should be identical.

Caveat: `llama3.1:8b` is not in `env/engine-lock.json`, so the engine check at the start reports drift and `hbench log validate` flags these runs until the model is added to the lock.
