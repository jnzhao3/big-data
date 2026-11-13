# E3 results: the BDI-Kit agent on local Qwen

**Question.** Can BDI-Kit 0.11.0's chat agent run headless on a local model, with every step logged by hbench, scored against GDC-SM / GDC-VM, and repeated exactly?

**Answer.**

- **Yes, on qwen3.8-27b. No on qwen2.5:14b.**
- **Infrastructure works.** On every run, hbench matched each logged LLM call to an Ollama generation. No call came from another client, and the sampler settings were never off.
- **Reproducible.** Two runs at T=0.6 with the same seed produced byte-identical requests, outputs and logprobs on all 26 calls.
- **The agent adds a measurable step.** On clark, Magneto's initial schema matches are 10/14. After the agent's corrections the final is 12/14, in every 27b run.
- **Only one condition, n=1 per prompt variant.** These runs show that the agent can be measured. They are not evidence about its accuracy or variance; that is E4.

All runs use study clark, cold-start engine protocol and a per-run BDI-Kit cache. Raw logs stay outside the repo, in `~/hbench_raw`. [results/](results/) holds the summaries, which `analyze_runs.py` regenerates from the raw logs.

## Runs

| label | run | model | task, prompt | sampling | end | SM top-1 (Magneto -> final) | time |
|---|---|---|---|---|---|---|---|
| 14b_verbatim | `e10c` | qwen2.5-14b-bdikit | sm_vm, verbatim | T=0 | table after 3 steps, 1 tool call | 9 -> 9 /14 | 3.4 min |
| 27b_verbatim_1h | `f69e` | qwen3.8-27b-bdikit | sm_vm, verbatim | T=0 | 1 h turn timeout | 10 -> 12 /14 | 61 min |
| 27b_dropline_1h | `70fe` | qwen3.8-27b-bdikit | sm_vm, format line dropped | T=0 | 1 h turn timeout | 10 -> 12 /14 | 60 min |
| 27b_verbatim_smvm | `5058` | qwen3.8-27b-bdikit (loaded b=1024) | sm_vm, verbatim | T=0 | full workflow, 3 h limit | 10 -> 12 /14 | 77 min |
| 27b_nb1024_T0.6_r1 | `f948` | qwen3.8-27b-bdikit-nb1024 | sm, verbatim | T=0.6, top_p 0.95, seed 1 | final table | 10 -> 12 /14 | 26 min |
| 27b_nb1024_T0.6_r2 | `44c3` | same as r1 | same | same | final table | 10 -> 12 /14 | 27 min |

Code versions:
- `f948` and `44c3` ran from a clean checkout of the third E3 commit.
- `5058` ran from a clean checkout of the second.
- The 2026-11-11 runs ran from a dirty worktree; the run headers keep each diff hash.

`5058` was labeled `no_final_table` because of a driver bug, since fixed (see "Fixes" below). Its scores come from the tool state, so the bug does not affect them.

## 1. Logging and reproducibility

- **Engine audit.** In every run, each logged LLM call matched an Ollama generation: 19/19, 74/74, 94/94, 80/80, 26/26 and 26/26. The audit found no foreign tasks and no sampler mismatches.
- **Seeded repeats are bitwise identical.** `f948` and `44c3` match on every call (4 agent + 22 tool-internal) in:
  - request hash;
  - per-call seed;
  - output hash;
  - logprobs hash.

  Their predictions and claims also match ([results/acceptance_pair.json](results/acceptance_pair.json)). Both loads ran at batch 1024. Three driver changes made this possible:
  - LangChain's random block ids are dropped from requests;
  - Ollama's random tool-call ids are renamed by position, because they are echoed back and would otherwise change every later prompt hash, and with it the seed;
  - the batch size is pinned through the `-nb1024` tag.
- **Batch size changes bits, not decisions.** `f69e` loaded batch 512 and `5058` loaded 1024. Their first 5 agent steps made the same tool calls with the same arguments, and their final schema mappings are the same. This agrees with E1. E4 should still pin the batch, because a different lineage breaks bitwise repeats.

## 2. Model choice

- **qwen2.5:14b: no-go.**
  - With the verbatim prompt, it calls `match_schema` once. After that it writes tool calls as JSON text, which comes from the prompt's line "If calling a tool, output a JSON object...". LangGraph treats that text as the final answer, so nothing is corrected and value matching never runs.
  - With that line dropped (earlier 14b runs, not summarized in results/), it makes native calls, but it invents arguments, batches calls and generates without stopping.
  - Neither variant follows the BDI-Kit workflow.
- **qwen3.8-27b (thinking on): go.**
  - Under both prompts it follows the workflow: match, inspect candidates and domains, correct, value-match.
  - Every tool call was native; text JSON appeared 0 times.
  - It needed one driver deviation: past thinking blocks are stripped from the history (`--history-thinking strip`), because Ollama `/v1` rejects them.
- **Verbatim vs dropped line on 27b.** The two trajectories split at step 2: one ranks first, the other previews first; one makes 6 corrections, the other 5. They reach the same SM score. With n=1 this shows only that prompt variants change the path.

## 3. What the agent changes in schema matching

Compare each run's initial `match_schema` output with its final state ([results/sm_changes.csv](results/sm_changes.csv)). All four 27b verbatim runs make the same changes:

| column | Magneto | agent's final | GT | effect |
|---|---|---|---|---|
| Tumor/Normal | tumor_regression_grade | tissue_type | tissue_type | fixed |
| Path_Stage_Reg_Lymph_Nodes_pN | lymph_nodes_positive | ajcc_pathologic_n | ajcc / uicc_pathologic_n | fixed |
| Histologic_Type | cog_renal_stage | papillary_renal_cell_type | primary_diagnosis | wrong -> wrong |
| Tumor_Site | percent_tumor_nuclei | max_tumor_bulk_site | primary_site, ... | wrong -> wrong |
| Case_ID | cases | code | none (unannotated) | not scored |
| Tumor_Site_Other | laterality | max_tumor_bulk_site | none (unannotated) | not scored |

- The agent fixed 2 columns and broke none.
- The dropped-line variant leaves Tumor_Site alone and picks `uicc_pathologic_n` for pN, which the GT also accepts.
- Magneto's reranker uses the same LLM, so the agent's gain must be measured against Magneto under the same model. On 14b Magneto's initial score is 9/14; on 27b it is 10/14.

## 4. Value matching (`5058`)

| | after `match_values` | final |
|---|---|---|
| categorical micro (77 items) | 0.532 | 0.558 |
| categorical coverage | 0.584 | 0.584 |
| all 414 GT items, micro | | 0.106 |

- **Accurate where attempted.** 43 of the 45 attempted categorical items are right. The two errors:
  - Race: an empty value became `unknown`, where the GT has `Unknown`, so this is a case mismatch;
  - Tumor/Normal: `TN` became `Tumor`, where the GT has `Normal`.
- **Corrections.** The agent changed 2 values after `match_values`: Ethnicity `White` and `WHITE`, from no match to `not hispanic or latino`. Both changes are right.
- **Error recovery.** `rank_value_matches` failed with the default method (`llm`, which is not supported). The agent retried with `tfidf` on its own.
- **The low overall score is mostly by design, not agent error.** It does not measure the agent's value-matching skill. Of the 369 items it did not attempt:
  - 337 are numeric or age items, which E3's task prompt does not ask for (it says "categorical attributes"). BDI-Kit has an `llm_numeric` method.
  - 30 belong to a second GT target of a column the agent mapped correctly, such as Ethnicity -> race or pT -> uicc_pathologic_t. BDI-Kit maps each source column to one target, so these items cannot be reached.
  - 2 come from the wrong Histologic_Type mapping.

## 5. Claims vs state (for E7)

The agent's final text is not a reliable record of what it did:

- **`5058`.** The final table says two Histologic_Type values were changed to "no match". No `update_value_matching` call touched them, and in the final state one of them is still `Type 1`.
- **`f948` / `44c3`.** The final text says GDC has "no perfect single attribute for general RCC histology". The GT target `primary_diagnosis` exists. It was not among the 10 candidates that `rank_schema_matches` showed for Histologic_Type, so the agent stated a fact about the whole GDC schema based only on the tool's shortlist.

The adapter scores the tool state, not the claims. The claims are saved apart, in `claims.md`, for E7.

## 6. Cost

Time is dominated by the tool-internal LLM calls with thinking on:
- each Magneto rerank call takes 30-110 s;
- `match_values` makes about 47 calls of about 37 s each.

Each agent step takes 1-2.5 min.

| task | time per run |
|---|---|
| `sm` | about 26 min |
| `sm_vm` with the final-table fix | about 65-70 min |
| `sm_vm` before the fix (`5058`) | 77 min (about 9 min wasted on follow-ups) |

Use `--turn-timeout` of at least 3 h for `sm_vm`.

## Fixes made during E3

- **Final-table check.** A thinking reply arrives as `[thinking, text]` content blocks. The driver checked a JSON dump of that list, so the escaped newlines never matched the table check. That is why `5058` was labeled `no_final_table` and got 2 needless follow-ups. The driver now reads only the text blocks (`message_text`). The fix was verified in `f948` and `44c3`: outcome `final_table`, 0 follow-ups.
- **Tool-call ids renamed by position**, and **the `-nb1024` tag**, as in section 1.

## Not shown here

- Variance across seeds, T>0 accuracy, comparisons across models or prompts (E4).
- Any study other than clark.
- Value matching for numeric and age attributes.

## Decisions for the team (E4)

1. Whether the task prompt should ask for numeric and age value matching. That adds a `match_values` pass to every run. Its time has not been measured.
2. Whether to turn thinking off for the tool-internal LLM only. This would be the largest speed-up. It is a deviation from the app, and no mechanism for it has been tested yet.
3. How to score GT items that cannot be reached because they are a second target of one column. Report coverage with and without them.
4. Run budget: at about 26 min per `sm` run or 70 min per `sm_vm` run, choose the number of replicates and conditions.

## Reproduce

```bash
ollama create qwen3.8-27b-bdikit -f experiments/e3_agent/Modelfile.qwen3.8-27b-bdikit
ollama create qwen3.8-27b-bdikit-nb1024 -f experiments/e3_agent/Modelfile.qwen3.8-27b-bdikit-nb1024
```

```bash
python -s experiments/e3_agent/run_agent.py --study clark --task sm --model qwen3.8-27b-bdikit-nb1024 --temperature 0.6 --top-p 0.95 --seed 1 --turn-timeout 10800
```

```bash
python -s experiments/e3_agent/analyze_runs.py --raw ~/hbench_raw
```

`analyze_runs.py` finds runs by the suffix of their directory name (the `RUNS` table in the script).
