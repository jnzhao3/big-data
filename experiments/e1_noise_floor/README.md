# E1: the inference engine's noise floor

How much does the inference engine alone change a method's outputs, with sampling switched off (T=0)? The answer bounds what any later experiment can attribute to the method: a difference between two runs that the engine could have produced is not evidence about the method.

## What E1 measures

At T=0 the output is a deterministic function of the logits, and the logits depend on how the KV of every context position was computed: in a batch of prompt tokens, one prompt token alone (the last token recomputed when the whole prompt was cached), or one generated token at a time. hbench calls this the call's **lineage** (`hbench/engine/lineage.py`). The same input on the same lineage gives bitwise-identical logprobs. So repeating a call measures nothing; E1 lists the lineages that later experiments actually produce and measures how far apart they are.

A decision flips when, at some position up to the end of its value, the gap between the chosen token and the best other candidate is smaller than the lineage's shift. hbench records each decision's margin (that minimum gap), so the risk of every decision can be computed in advance.

## Results

All three stages have run (2026-11-03 to 2026-11-12). The analysis is in [RESULTS.md](RESULTS.md) and the summary
tables in [results/](results/). In short: at T=0 the engine changed 0 of 360 agent final answers (180 on
qwen2.5:14b, 180 on qwen3.8-27b-bdikit) and 3 of 1138 single-turn decisions; every bitwise difference follows the
lineage, which the host-RAM prompt cache and Ollama's load-time batch size can change; default-temperature sampling
on 27b changed 12 of 100 agent final answers.

## Stage 3: sampling at the default temperatures (design; ran 2026-11-12 on 27b)

Stages 1 and 2 measured the engine at T=0. Methods are usually run at the endpoints' defaults: `/api/chat` T=0.8, top_p=0.9; `/v1` (the agent's route) T=1.0. Stage 3 asks four questions:

1. **Size.** How often do decisions and final answers change at the defaults, compared with the T=0 floor?
2. **Prediction.** Does a single T=0 run predict which decisions will change? It gives `p_path`, the probability that the sampler reproduces the greedy tokens through the end of a decision value.
3. **Seeds.** Does a fixed seed reproduce a T>0 run? Does that still hold on another lineage?
4. **Amplification.** Does a multi-turn loop change its final answer more often than its final call does when resampled alone?

**Predictions, written before the run:**

- **P6a.** The same seed on the same lineage gives bitwise-identical output.
- **P6b.** The same seed on the single-token lineage changes the text of a few calls, far fewer than a different seed does.
- **P7.** For clark at the `/api/chat` defaults, the observed rate of path changes matches `p_path`. The test is one-sided: clark's `format=json` grammar can only make the greedy token likelier.
  - Controls with `p_path` = 1 never change.
  - The value changes at most as often as the path.
- **P8.** For an agent call, `p_path` is near 0 at T=1, because it includes the free reasoning text before the tool call. In Stage 2's calls, the value tokens alone were nearly certain. The forecast `p_value_local` (the value's own tokens, given the greedy prefix) should therefore underestimate how often the value changes.
- **P9.** The whole episode changes its final answer more often than its final call does when resampled alone.
- **P10.** The final answers of agent episodes change far more often than at T=0 (0/180 in Stage 2).

**Design.** `run_plan.py make stage3` writes 222 steps, about 2-2.3 h.

| block | what | answers |
|---|---|---|
| 3c-ref | Each of the 20 columns once on `/v1` at T=0 from an unload, then replayed per call cold: batched-lineage greedy sources | P5 on `/v1`; are `/v1` and `/api/chat` finals the same? |
| 3b chat | 50 clark decisions from Stage 1 (8 per bin of 1 - `p_path`, plus 10 controls), T=0.8, seeds 1-12, one cold_start pass per seed | P7 |
| 3b v1 | 30 calls from the 3c-ref replays (every episode's final call, plus the 10 least certain earlier calls), T=1.0, seeds 1-10 | P8 |
| 3a | Passes for seeds 1-3 repeated whole; a P6 subset resent with the same seeds | P6 |
| 3c | 20 columns x seeds 1-6, whole episodes on `/v1` at T=1.0, each from an unload; then seed 1 again | P9, P10, seeded agent reproducibility |

- **Sources and samples share a lineage.**
  - A pass unloads first, then sends distinct prompts in order, so every call is computed in batches (P2).
  - Agent calls that reused generated tokens are first replayed per call cold. Each replayed call borrows the spec of the episode call it replays (`spec_from`).
  - The lineage check confirms that every sample was batched.
- **Every T>0 call gets its own seed.** llama-server reseeds its sampler per request. With one seed for every call, the first smoke test sampled all clark decisions (tokens 4-9) with the same random draw: 0 of 80 values changed, where p_path predicted 22.8. Sampled episodes state the replicate's seed as a base and LLMClient derives each call's seed (`hbench.seeds`, rule `call/1`). Resampled calls go through `replay_call`, which sends seeds as given, so `resample.call_seed` derives them from the pass's seed and the source.
- **Selections are fixed before the run.** Within each `p_path` stratum, calls are ordered by sha256 of their span id. Each selection is written to `selections/*.json`.

**On qwen3.8-27b-bdikit (thinking on).** After the pilot, Stage 3 runs on the model E3 uses. Its decisions follow
about 300 tokens of reasoning, so `p_path` is about 1e-30 for every call at its default sampler and the path-level
questions (P7, path controls) have no content there; the value-level ones remain. The changes for it:

- **The model's own sampler.** qwen3.8-27b-bdikit sets temperature 1, top_k 20 and top_p 0.95 in its Modelfile.
  Predictions use them (`resample.model_sampler_of`: the call's `model_sampler`, else the run header's engine
  snapshot for runs logged before hbench 49eae7b); `/api/chat` runs this model at T=1.0
  (`run_plan.default_temperatures`). qwen2.5:14b sets none, so its plan and numbers are unchanged.
- **Selection by value.** `--chat-by value` takes every clark call of the pilot's single-turn runs (31 with a
  decision) instead of `p_path` bins; the 1a runs may sit under `1a/` or `1a/raw/`.
- **Smaller.** Seeds 4 (chat), 4 (v1), 2 (P6), 5 (agent); P6 subsets of 8; the same-seed repeats send only the
  P6 subset (`--repeat-subset p6`). About 6 h at the pilot's rates (a single-turn call takes about 50 s, an
  episode about a minute).
- **No answer.** A thinking model can stop after its reasoning with an empty answer (1 of 32 pilot calls).
  Samples carry `no_answer` and `empty_output`, and the analysis counts them apart from other value changes.
- `think` goes to every episode; `/v1` has no switch and the model thinks there by default.

Stage 2 is repeated on the same 20 columns with the three blocks (207 steps, about 3.8 h; the single-turn check
is dropped because the pilot covered single turns), so Stage 3 has T=0 episodes of the same model to compare with.
Stage 1 is not repeated: the engine findings do not depend on the model, and the pilot's clark runs are the
27b single-turn source for the `/api/chat` selection.

## qwen3.8:27b pilot

E3 moved the BDI-Kit agent to `qwen3.8-27b-bdikit` with thinking on, so a small pilot repeats the model-dependent
parts of Stages 1 and 2 on it before Stage 3: decision margins and resend flips on clark (full + shortlist), and
three clark columns x 3 repeats in two blocks. With thinking, the token stream carries the reasoning up to
`</think>` and tool calls are written in Qwen3.5's XML format; `hbench.decisions` looks for decisions after
`</think>`, parses the XML calls, and reports `margin_free` (the prefix margin without positions a format=json
grammar forced). Driver options: `stage1.py --model ... --studies clark --think --skip-1c`; `run_plan make stage2
--model ... --think on --blocks unload_each,back_to_back --no-single-turn --columns "clark:Case_ID+Tumor/Normal+Age"`.

Result (2026-11-11): 0/32 single-turn resend changes and 0/18 agent final changes; back_to_back pairs were all
bitwise equal. With thinking the lowest prefix margin sits inside the reasoning (median about 0.01-0.02 single
turn, 0.027 for agent final decisions vs 0.040 on 14b), so at the model's default sampler (T=1, top_k 20,
top_p 0.95) every decision path changes between runs and only value-level measures stay informative. The analyses
read blob-stored outputs (`resample.read_run_events`), pool all path pairs when no `partial->single_token` resend
exists (`noise_model_pairs`), and leave grammar-forced positions out of `decision_gaps`.

## Files

| File | What it does |
|---|---|
| `stage1.py` | Stage 1 driver: 1a margin survey, 1b resend replays of every call with margin < 0.5 plus a 5% sample, 1c transports |
| `analyze_stage1.py` | Stage 1 tables: margins, flips, gap-shift distribution, noise-model check, P2, transports, back-to-back repeats |
| `tool_loop.py` | A synthetic tool loop (T5): one episode = one run, three tools with fixed returns, a decision spec on every call |
| `run_plan.py` | Run-plan executor (T6): writes a plan, runs it in order, records each step's runs and outcome |
| `analyze_stage2.py` | Stage 2 tables (T7): repeat shapes per order, replay fidelity, gap shifts, tool-decision margins, noise budget |
| `resample.py` | Stage 3 resampling: selections of greedy source calls, one T>0 pass per seed and protocol, path- and value-level comparison with each source |
| `analyze_stage3.py` | Stage 3 tables: calibration of `p_path` and `p_value_local`, seed x lineage, episode-level changes and amplification |
| `RESULTS.md`, `results/` | The analysis of all stages, and the summary tables it cites |

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

# Stage 3: default-temperature sampling (about 2-2.3 h; --seeds chat,v1,p6,agent lowers the seed counts)
$PY -I experiments/e1_noise_floor/run_plan.py make stage3 --out ~/e1/stage3.json --stage1 ~/e1/stage1
$PY -I experiments/e1_noise_floor/run_plan.py run ~/e1/stage3.json --root ~/e1/stage3
$PY -I experiments/e1_noise_floor/analyze_stage3.py ~/e1/stage3 --stage2-episodes ~/e1/stage2/stage2_episodes.csv

# qwen3.8-27b-bdikit, thinking on (Stage 2 about 3.8 h, Stage 3 about 6 h)
$PY -I experiments/e1_noise_floor/run_plan.py make stage2 --out ~/e1/s2_27b.json --model qwen3.8-27b-bdikit --think on --no-single-turn
$PY -I experiments/e1_noise_floor/run_plan.py make stage3 --out ~/e1/s3_27b.json --model qwen3.8-27b-bdikit --think on \
  --stage1 <pilot>/single --seeds 4,4,2,5 --chat-by value --p6 8 --repeat-subset p6
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
