# E1: the inference engine's noise floor

How much does the inference engine alone change a method's outputs, with sampling switched off (T=0)? The answer bounds what any later experiment can attribute to the method: a difference between two runs that the engine could have produced is not evidence about the method.

## What E1 measures

At T=0 the output is a deterministic function of the logits, and the logits depend on how the KV of every context position was computed: in a batch of prompt tokens, one prompt token alone (the last token recomputed when the whole prompt was cached), or one generated token at a time. hbench calls this the call's **lineage** (`hbench/engine/lineage.py`). The same input on the same lineage gives bitwise-identical logprobs. So repeating a call measures nothing; E1 lists the lineages that later experiments actually produce and measures how far apart they are.

A decision flips when, at some position up to the end of its value, the gap between the chosen token and the best other candidate is smaller than the lineage's shift. hbench records each decision's margin (that minimum gap), so the risk of every decision can be computed in advance.

## Results so far (qwen2.5:14b, Ollama 0.40.0, one Apple-silicon machine)

**Stage 1: single-turn `clark_match`** (10 studies, full and shortlist modes, 1138 decisions).

- A single-turn method has exactly two lineages. Both are deterministic:
  - batched: every natural call but one;
  - last token recomputed alone: an identical resend, or a host-RAM prompt-cache restore of the same prompt.
- Batched prompt computation gives the same result however it is split. 150 of 150 batched resends were bitwise identical, at split points from 46 to 3737 cached tokens.
- Enumerating both lineages flips **3 of 1138 decisions (0.26%)**. All three were near-ties, with margins 0.0014, 0.0018 and 0.029. The noise model predicted 3.25.
- Gap shifts between the two lineages (4890 positions):

  | median | p90 | p99 | max |
  |---|---|---|---|
  | 0.023 | 0.065 | 0.14 | 0.33 |

  The shifts have no sign bias. 94% of decisions have a margin above the largest shift.
- `/api/chat` and `/v1` give bitwise-identical logprobs on the same lineage. They differ only in their default samplers.
- **Back-to-back repeats change the lineage.** Repeating clark shortlist immediately put all 16 calls of the second run on the single-token lineage, through RAM-cache restores. Repeats of full mode, whose prompts share a long prefix, stayed bitwise identical.
  - Single-turn repeats therefore use `--engine-protocol cold_start`, or report calls stratified by the audit's `ram_restore`.
- At default temperature, the T=0 logprobs predict 58–178 changed decisions per 560 between two runs, about 50x the engine floor. Since clark uses `format=json`, these predictions are upper bounds.
- `hbench lineage rebuild` reconstructed all 1550 Stage 1 calls from the server log with no uncertain cases. The classes match the known ones, and `hbench lineage check` found no violations.

**Stage 2: agent-shaped requests.** Not run yet: the tools below are new.

## Files

| File | What it does |
|---|---|
| `stage1.py` | Stage 1 driver: 1a margin survey, 1b resend replays of every call with margin < 0.5 plus a 5% sample, 1c transports |
| `analyze_stage1.py` | Stage 1 tables: margins, flips, gap-shift distribution, noise-model check, P2, transports, back-to-back repeats |
| `tool_loop.py` | A synthetic tool loop (T5): one episode = one run, three tools with fixed returns, a decision spec on every call |
| `run_plan.py` | Run-plan executor (T6): writes a plan, runs it in order, records each step's runs and outcome |
| `analyze_stage2.py` | Stage 2 tables (T7): repeat shapes per order, replay fidelity, gap shifts, tool-decision margins, noise budget |

## The synthetic tool loop

The model matches one source column with three tools:
- `list_candidate_columns`: the clark_match shortlist rule, 10 candidates;
- `get_column_values`: up to 5 sample values;
- `submit_match`: the decision.

Tool returns are a function of the arguments only, so two episodes of one column differ only through the engine at T=0. It is not the BDI-Kit agent, whose tools reorder rows from run to run; that agent is measured in E3 with the same instruments.

Every call carries a decision spec: the tool name, the inspected column, the submitted target, or a target in JSON text. Keys of parallel calls get `@1`, `@2`. Margins are prefix margins, so a submitted target's margin also covers the tool name before it.

## Running

Run from the repo root with `PY=~/venvs/harmonize/bin/python`. Every output root must be outside the repo, because Qwen's top-5 candidates contain Chinese tokens and the raw logs cannot be edited. The Ollama server must be idle: a plan's `engine_check` step stops when another model is loaded.

```bash
# Stage 1 (about 75 min) and its analysis
$PY -I experiments/e1_noise_floor/stage1.py ~/e1/stage1
$PY -I experiments/e1_noise_floor/analyze_stage1.py ~/e1/stage1

# Stage 1e: back-to-back single-turn repeats from an unload (about 5 min)
$PY -I experiments/e1_noise_floor/run_plan.py make stage1e --out ~/e1/stage1e.json
$PY -I experiments/e1_noise_floor/run_plan.py run ~/e1/stage1e.json --root ~/e1/stage1e

# Stage 2: 20 columns x 3 replicates x 3 orders, replays, a single-turn third repeat (about 1-1.5 h)
$PY -I experiments/e1_noise_floor/run_plan.py make stage2 --out ~/e1/stage2.json --columns clark:10,krug:10
$PY -I experiments/e1_noise_floor/run_plan.py run ~/e1/stage2.json --root ~/e1/stage2
$PY -I experiments/e1_noise_floor/analyze_stage2.py ~/e1/stage2 --stage1-shifts ~/e1/stage1/stage1_gap_shifts.csv.gz
```

- `run_plan.py run ... --resume` continues an interrupted plan and re-runs only the steps that did not finish `ok`.
- The root receives:
  - `plan.json`: the plan with its sha256;
  - `execution.jsonl`: per step, its kind, times, status, the run folders it created and a tail of its output;
  - `validate_summary.json` and `lineage_check.json`;
  - the run folders, under `raw/`.

The Stage 2 plan, each block starting from an unload:

| Block | Order | Prediction |
|---|---|---|
| `unload_each` | every episode starts cold (`cold_start`) | all replicates bitwise identical (P1) |
| `back_to_back` | A A A B B B ... | replicate 1 differs, later replicates agree: their first prompt is a prefix of the cached state (P4) |
| `interleaved` | A B ... A B ... | as back_to_back, through RAM-cache restores (P4) |
| replays | each `unload_each` replicate 1, replayed per call cold and whole from cold | per-call cold differs exactly where generated tokens were reused (P5); the whole replay is bitwise identical (P1) |
| single turn | clark shortlist runs 6, 7, 8 back to back | run 6 differs, runs 7 and 8 are bitwise identical |
