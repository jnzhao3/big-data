# E4: repeated runs of the BDI-Kit agent

This directory holds the tooling for E4's nights. Each run is one `experiments/e3_agent/run_agent.py --task sm` run of the E3 driver on `qwen3.8-27b-bdikit-nb1024`. E1's `run_plan.py` executes the plans, so a night can resume after a stop.

## A night

From the pinned checkout `~/bigdata-wt/run-e4-<commit>`, in `~/venvs/harmonize`. The machine must be plugged in, with the lid open and the app's keep-awake on. Quit i4Tools.

```bash
python -s experiments/e4_agent_repeats/make_plan.py night N1 --out ~/hbench_raw/e4_plans/n1.json
experiments/e4_agent_repeats/night.sh ~/hbench_raw/e4_plans/n1.json ~/hbench_raw/e4_plans/n1
```

`night.sh` does the following:
1. Runs under `caffeinate -i -s`.
2. Checks that Ollama answers, and warns if i4Tools is running.
3. Creates the shared T=0 tool cache if the plan uses one.
4. Runs `run_plan.py run --resume`.
5. Writes `summary.md` and `summary.json` under the root. They give each step's status and, for each run, its outcome, wall time, SM score, engine audit and `hbench log validate` result.

To resume after a stop, rerun the same command. Steps that finished ok are skipped.

## Plans

- `make_plan.py night N1..N5 [--studies s1,s2,s3]` writes doc appendix A, with the studies filled in by role.
- `make_plan.py cells clark:D:1:2 ... --out x.json` writes any list of cells.
- `--sample-t` and `--sample-top-p` set the sampled side, by default T=1.0 and top_p 1.0.

| cell | agent | tools | flags |
|---|---|---|---|
| `s:Z1` | T=0 | T=0 | seed 1; fills the shared T=0 tool cache (`--bdikit-cache shared`) |
| `s:Z2`, `s:Z` | T=0 | T=0 | seed 1, own cache |
| `s:D:a:t` | sampled | sampled | `--seed a`; `--tool-seed t` only when t differs from a |
| `s:T:k` | T=0 | sampled | `--seed k` |
| `s:A:k` | sampled | T=0 | `--seed k`, shared T=0 tool cache |
| `s:P-hash:h` / `s:P-prompt` / `s:P-row:r` | as Z | as Z | `--hashseed random` / `--prompt-variant no_json_format` / `--row-order r` |

Every run also gets:
- `--model qwen3.8-27b-bdikit-nb1024 --task sm --turn-timeout 10800 --engine-protocol cold_start`;
- `--block` and `--replicate` labels;
- the same absolute `--raw-dir` (`~/hbench_raw`).

The raw dir must never change between E4 runs. The staged input path `<raw-dir>/_inputs/e3/<study>.csv` is in the task prompt, so another raw dir changes the prompt, every per-call seed and the tokens. The run_plan root, where `plan.json` and `execution.jsonl` live, is separate and is not in any prompt.

## The Z1 = Z2 check

When a plan holds a study's Z1 and Z2, a check step follows Z2 (`make_plan.py compare`). It compares the two runs' per-call request, seed, output and logprobs hashes, and their predictions. If anything differs, it fails, and `stop_on_fail` stops the plan.

Z1 must start from an empty cache `runs/` directory. Otherwise its tool calls would come from the cache, and the check would fail on the call counts. `prepare` warns when the cache is not empty.

The cache directory, `~/hbench_raw/e4_t0cache` by default, links `models/` and `embeddings/` to `~/.cache/bdikit`. Its path must not contain mpnet, roberta, e5, arctic or minilm, because Magneto picks its base model by a substring of the path.

## Driver flags added for E4 (`run_agent.py`)

- **`--tool-seed t`.** The base seed of the tool LLM's per-call seeds (`HBENCH_TOOL_LLM_SEED`). It is recorded as `decoding.tool_llm_seed` and in the header's `llm.tool_llm.seed`.
  - When omitted, it is `--seed`, so every request is the same as before the flag.
- **`--row-order r|none`.** Stages the source CSV with its records permuted by a seeded RNG, at the same path, so the prompt does not change.
  - The header and each record keep their original bytes. A file with multi-line fields is refused.
  - The file is staged again on every run, atomically, and checked against the expected sha256. `none` restores the original.
  - The order and the staged sha256 are recorded in `condition.data` and in the header's `task.input`.
