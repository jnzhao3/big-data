# E1 results: the inference engine's noise floor

One Apple-silicon machine (64 GB), Ollama 0.40.0 (llama.cpp on Metal), `NUM_PARALLEL=1`. Models: `qwen2.5:14b`
(Stages 1-2) and `qwen3.8-27b-bdikit` with thinking on (pilot, Stages 2-3). Tasks: single-turn `clark_match` on
GDC-SM, and the synthetic tool-loop agent on 20 columns (clark 10, krug 10). Every run passed `hbench log validate`,
and `hbench lineage check` found no violations. Methods, predictions and commands are in [README.md](README.md);
the summary tables are in [results/](results/).

## Bottom line

1. **At T=0 the engine does not change agent answers.** 0 of 180 episodes changed their final answer on 14b, and
   0 of 180 on 27b. Single-turn: 3 of 1138 decisions (0.26%) flip, all near-ties.
2. **Every bitwise difference is explained by the computation lineage.** Same lineage gives bitwise-identical
   logprobs; a different lineage gives small differences. No exceptions in any comparison.
3. **Two hidden factors change the lineage:** the server's host-RAM prompt cache, and the batch size Ollama picks
   from free RAM at load time. Unloading before each run removes the first; a pinned `num_batch` removes the second.
4. **Sampling, not the engine, is what moves results.** At 27b's default sampler (T=1), 12 of 100 agent episodes
   ended with a different final answer than at T=0, concentrated in hard columns; mean accuracy did not move.
5. **For a thinking model, T=0 logprobs do not predict sampling stability.** Estimate it by resampling.

## Findings

### 1. Mechanism: lineage decides the numbers

| comparison | result |
|---|---|
| same lineage, rerun (cold start or same split) | bitwise identical, every time (14b and 27b, single turn and multi-turn) |
| batched prompt, different split points (46-3737 cached tokens) | identical: 150/150 resends, 16/16 cold starts |
| batched vs last token recomputed alone (14b) | every position differs |
| 27b: whole first prompt in one batch (`B521`) vs `B512 b9` | differ; a tail of up to 4 tokens (`b4`) is equivalent to one batch |
| `/api/chat` vs `/v1` on one lineage | identical: 16/16 calls (14b), 20/20 final answers (27b) |

The likely cause is that llama.cpp picks different Metal kernels by batch size, so the accumulation order and
rounding change. This is inferred from the pattern, not checked at kernel level.

### 2. What changes the lineage in practice

- **Host-RAM prompt cache.** In a back-to-back repeat, the first call of run 2 restores the prompt from the cache and
  recomputes only the last token. 14b agent back-to-back: in 20/20 columns, replicate 1 differed and replicates 2-3
  agreed. Unloading before each run gave 60/60 bitwise-identical pairs. Single-turn repeats with a long shared
  prefix do not hit the cache and stay identical.
- **Batch size chosen at load.** Ollama passes `-b/-ub 1024` or `512` to llama-server depending on free system RAM
  (from 409 logged 27b loads: 1024 above about 30 GiB free, 512 below; 14b's threshold is about 19 GiB). 27b's first
  agent prompt is about 520 tokens, so the choice changes the first call's lineage. Across 27b comparisons, same
  lineage was bitwise identical in 156/156 cases and different lineage in 0/110.
- **Pinning.** `/api/chat` honors `options.num_batch`; `/v1` ignores it and honors only the Modelfile parameter. The
  tag `qwen3.8-27b-bdikit-nb1024` (a Modelfile with `FROM qwen3.8-27b-bdikit` and `PARAMETER num_batch 1024`) was
  checked against the original tag with `num_batch=1024`: bitwise identical on both
  APIs (531-token prompt, thinking on, 396 generated tokens); the `num_batch=512` control differed (max logprob
  difference 0.020). Short probes (64 tokens, no thinking) could not tell 512 from 1024.

### 3. Size of the perturbation

Gap shift = change in the logprob gap between the chosen token and the best other candidate.

| source | positions | median | p90 | p99 | max |
|---|---|---|---|---|---|
| 14b single turn: batched vs single-token lineage | 4890 | 0.023 | 0.065 | 0.14 | 0.33 |
| 14b agent: live vs per-call cold replay | 4112 | 0.026 | 0.11 | 0.31 | 1.76 |

Shifts have no sign bias and do not depend on the original gap. Flip probability for a position with margin m:
26% at m=0.02, 8% at 0.05, 1.3% at 0.1, 0.04% at 0.2.

### 4. Effect on decisions at T=0

**Single turn (14b).** 94% of 1138 decisions have a margin above the largest observed shift (0.33) and cannot flip.
Enumerating both lineages flipped 3 (margins 0.0014, 0.0018, 0.029); the noise model, fitted leave-one-out,
predicted 3.25. Annotated-column accuracy did not change. `results/stage1_14b/stage1_decisions.csv` gives each
decision's `p_flip_lineage`.

**Agent.**

| | 14b | 27b |
|---|---|---|
| episodes | 180 | 180 |
| final answer changed | **0** | **0** |
| bitwise pairs: unload each / back-to-back / interleaved | 60/60, 20/60, 54/60 | 48/60, 60/60, 60/60 |
| cause of differences | RAM-cache restore | batch size at load |
| diverged pairs whose text changed | 16 | 0 |
| median prefix margin of the final target | 0.040 | 0.017 |
| noise budget: expected divergences before the final value, per episode | 0.20 | 0.12 |

The noise budget bounds divergence, not answer change.

**Replay (for E5).** 14b: replaying the whole episode in order from a cold start reproduced 61/61 calls bitwise;
replaying single calls cold reproduced 20/20 calls without generated-token reuse but 0/41 calls with it (6 changed
their text). 27b sends history without the thinking text, so it has no reuse; its replay differences (outputs equal
60/62) all came from a batch-size mismatch.

### 5. Sampling at the default temperature (27b, T=1, top_k 20, top_p 0.95)

**Single calls resampled with 4 seeds.**

| decision | samples | reasoning path changed | answer changed | forecast by `p_value_local` |
|---|---|---|---|---|
| `/api/chat` clark single turn | 124 | 124 | 10 (8%) | 0 |
| `/v1` column to inspect | 100 | 100 | 64 (64%) | 36 |
| `/v1` tool choice | 180 | 180 | 33 (18%) | 0.27 |
| `/v1` submitted target | 80 | 80 | 14 (18%) | 0 |

The reasoning path always changes (`p_path` is about 1e-30). `p_value_local`, the answer tokens' probability after
the reasoning, is near 1 because the decision is made inside the reasoning; it badly underestimates change (P8).
Prefix margins do not separate changed from unchanged decisions either (median 0.022 vs 0.035).

**Whole agent episodes, 20 columns x 5 seeds.**

- 12/100 final answers differ from T=0, in 5 columns; 15 columns agree in all 5 seeds.
- Accuracy: T=0 0.60; seeds 1-5: 0.60, 0.65, 0.65, 0.60, 0.65.
- By whether the gold target is reachable:

  | gold target | columns | episodes changed | accuracy when sampled |
  |---|---|---|---|
  | in the shortlist | 8 | 3/40 | 0.95 |
  | not in the shortlist | 3 | 4/15 | 0.00 |
  | not annotated | 9 | 5/45 | - |

- Amplification (P9): resampling only the final call changed the answer 21% of the time, but 15% was "do not submit
  yet" and only 6% a different answer. Whole episodes changed the final answer 12% of the time: switching answers is
  amplified (6% to 12%), while most second thoughts return to the original answer.
- Same seed rerun: final answers 20/20 equal, bitwise 13/20; the 7 others differ in first-call batch size.

At 14b, T=0 logprobs predicted about 50 times more decision changes between two default-temperature runs than the
T=0 engine floor.

## Predictions

| # | prediction | outcome |
|---|---|---|
| P1 | rerunning a sequence from a cold start is bitwise identical | holds |
| P2 | batched prompt computation is split-invariant | holds for 46-3737; on 27b `b9`-`b16` tails are not equivalent, `b4` is |
| P3 | a single-turn method has two lineages, each deterministic | holds |
| P4 | back-to-back: first run differs, later runs agree; unload-first all agree | holds on 14b; it depends on the lineage, not the order |
| P5 | per-call cold replay differs exactly where generated tokens were reused | holds; whole-episode cold_start replay is exact |
| P6a-c | same seed and lineage identical; same seed, other lineage mostly same text; other seed differs | hold |
| P7 | `p_path` predicts default-temperature changes | held on 14b in a retrospective check; no content on 27b (paths always change) |
| P8 | thinking model: answers change more than `1 - p_value_local` | holds, by far |
| P9 | episodes amplify answer changes | partly: different answers 6% to 12%; all changes incl. "not yet" 21% to 12% |
| P10 | default sampling changes agent answers far more than T=0 | holds: 12/100 vs 0/180 |

## Rules for E2-E6

1. **Pin the engine.** New 27b runs use `qwen3.8-27b-bdikit-nb1024`; never run the two tags at the same time (a
   different `num_batch` forces a reload). Repeats at T=0 unload first, or are stratified by first-call lineage.
2. **At T=0 one run is representative.** Engine noise does not change agent answers. In single-turn tasks the few
   decisions that can flip are known in advance from `p_flip_lineage`.
3. **At default sampling, run several seeds per column** and report by whether the gold target is in the candidates.
   State temperature and top_p in every condition.
4. **E2 (Magneto, T=0):** large run-to-run differences point to retrieval, tie-breaking or ordering, not the engine.
5. **E5:** replay whole episodes in order from a cold start. Per-call cold replay cannot reproduce turns that reused
   generated tokens.
6. **E6:** for a thinking model, pick decisions for voting or re-asking by resampling, not by T=0 logprobs.
7. **Engine upgrade:** rerun Stage 1b (about 20 minutes) to see whether the floor moved.

## Limits

- One machine, one engine version, Metal only; lineage rules may differ on other hardware or CUDA.
- 20 agent columns from two studies; 5 seeds per column at T=1, so 12/100 has a wide interval.
- Full Stage 3 ran only on 27b; 14b ran a Stage 3 smoke test.
- E1 covers the LLM engine only, not Magneto's embeddings, BDI-Kit tool output order or cloud models.
- Open: in Stage 3 block 3a (`/api/chat`), 2 of 16 same-seed same-lineage pairs were not bitwise identical; probably
  a batch-size mismatch, not checked per call.

## Files in `results/`

Analysis outputs only; raw logs stay outside the repo because Qwen's top-5 candidates contain Chinese tokens. The
`root` field of each JSON is blanked.

| folder | contents |
|---|---|
| `stage1_14b/` | `stage1_analysis.json`; `stage1_decisions.csv` (one row per decision, with margin and `p_flip_lineage`) |
| `stage2_14b/` | `stage2_analysis.json`; episodes and replicate pairs |
| `pilot_27b/` | single-turn and agent analyses of the 27b pilot |
| `stage2_27b/` | `stage2_analysis.json`; episodes and replicate pairs |
| `stage3_27b/` | `stage3_analysis.json`; per column, per decision, per episode |
