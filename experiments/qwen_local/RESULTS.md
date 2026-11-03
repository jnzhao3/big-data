# Feasibility study: local Qwen through Ollama (2026-11-02)

Goal: check, before the team meeting, whether local Qwen models (via Ollama) are usable for our LLM data-harmonization agent experiments.
Test machine: MacBook Pro M1 Max 64GB, Ollama 0.40.0.

**Conclusion**: usable. Recommended main model: `qwen2.5:14b`. Its accuracy is about the same as the 27B model,
it is 3-6x faster, and tool calling works. Use the 27B model only for a few comparison runs.
The bottleneck is candidate-column recall, not the model.

These runs used the settings of the time. The model's default temperature applied unless stated. clark_match used `num_ctx` 16384. Magneto's reranker used LiteLLM's `ollama_chat/` route.
They were first scored against the older `GDC_new/<study>/groundtruth.csv` (197 rows) and later rescored against GDC-SM v1.0 (259 pairs, n:m); results below are reported against both. The predictions are unchanged, and so are the conclusions.

The results in `results/` were first produced in separate environments: a Python 3.11 env for `clark_match.py`, and a
Magneto venv built from the GitHub clone with litellm 1.106.6 and transformers 5. On 2026-11-02, both Magneto
runs were repeated in the shared environment. The no-LLM run and the qwen2.5:14b reranker run (T=0) gave the same
predictions for every column as the saved results. The reranker run was repeated again through the new default
`openai/` route, and again gave the same ranking for every column.

## 1. Speed (`results/bench_speed.txt`)

| Model | Cold load | Generation speed | Notes |
|---|---|---|---|
| qwen3:4b | 9.0s | ~64 tok/s | Still "thinks" with `think:false`; with `max_tokens=50` the OpenAI endpoint returns an empty string. Not recommended |
| qwen2.5:14b | 3.8s | ~20 tok/s | |
| qwen3.8:27b-q4_K_M | 10.7s | ~13 tok/s | |

## 2. OpenAI-compatible endpoint + tool calling (`results/test_openai_tools.txt`)

All three models work through `localhost:11434/v1`. All three emit tool calls with correct arguments
and answer correctly after receiving the tool result.
qwen2.5:14b put garbled text (a few Thai characters, then Chinese) in `content` alongside its tool call,
and answered the follow-up in Chinese although the prompt was in English. The tool call itself was correct.
Those two outputs are replaced with English translations in the results file.

Why it happened (follow-up check, same tool-calling prompt repeated):

| Condition | Runs | Non-English text |
|---|---|---|
| Default temperature, model already loaded | 20 | 0 |
| Temperature 0, model already loaded | 20 | 0 |
| Default temperature, fresh load -> "OK" prompt -> tool prompt (original order) | 6 | 1 (Thai) |
| Temperature 0, same order | 10 | 0 |

- The model sets no temperature, so Ollama's default (0.8) applies. Occasionally the first sampled token is a
  low-probability Thai/Chinese token and the model keeps writing in that language. The tool call itself stays correct.
- The Chinese follow-up answer came from feeding that text back in the chat history.
- Ollama's qwen2.5 template renders only the text of an assistant message that has both text and tool calls,
  so the tool call disappears from the history.
- For agent experiments: use `temperature=0` unless run-to-run variation is what you are measuring, and drop the
  text of assistant messages that contain tool calls before sending them back.
- Small sample: whether a fresh load makes it more likely is not established.

## 3. clark column matching (`results/gdc_sm/`, original scoring in `results/`)

Data: the clark study in GDC_new. Source: 110 rows x 16 columns (CPTAC clear cell renal cell carcinoma clinical data).
Target: 736 columns (GDC fields; the columns that have values mostly hold the list of allowed values for that field).
Ground truth, 14 source columns in both versions:
- GDC-SM: 25 pairs. Seven columns have several correct targets, e.g. Age -> {age_at_diagnosis, age_at_index, days_to_birth},
  Ethnicity_Self_Identify -> {ethnicity, race}, pT/pN/stage -> both the `ajcc_*` and `uicc_*` fields.
- Old groundtruth: 15 pairs; only Tumor_Site has two targets.

A prediction counts as correct if it is any member of the column's gold set.
Case_ID and Tumor_Site_Other have no matching column in either version, so the expected answer is NONE.

Two candidate modes:
- `shortlist`: 20 candidates chosen by column-name similarity + value overlap (fuzzy containment), shown with sample values. Candidate recall 11/14.
- `full`: all 736 target column names in the prompt (no sample values); the shared prefix lets Ollama reuse the KV cache. Recall 14/14.

Model default temperature (temperature not set), `format=json`, `think=false`.

| Config | GDC-SM run1 / run2 | Old GT run1 / run2 | Columns that changed between runs | Time run1/run2 |
|---|---|---|---|---|
| shortlist + qwen2.5:14b | 9/14 / 9/14 | 8/14 / 8/14 | 1/16 | 62s / 31s |
| full + qwen2.5:14b | 11/14 / 12/14 | 9/14 / 11/14 | 5/16 | 54s / 35s |
| shortlist + qwen3.8:27b | 10/14 / 11/14 | 10/14 / 9/14 | 3/16 | 187s / 179s |
| full + qwen3.8:27b | 10/14 / 11/14 | 8/14 / 9/14 | 5/16 | 242s / 200s |

Shortlist candidate recall is 11/14 under both ground truths.

Failure modes:
- All outputs were valid JSON; no invalid output.
- Hallucinated column names only appeared in `full` mode: 2 in 14b run1 (`anatomic_site`, `histology`).
  (A lowercase `none` in 27b run2 was originally counted as hallucinated; scoring is now case-insensitive and the results were rescored.)
- Under GDC-SM, three columns are wrong in all 8 runs (systematic, not random):
  Tumor/Normal -> tissue_type (only Magneto's embedding retrieval finds it), Histologic_Type -> primary_diagnosis,
  and Case_ID (expected NONE; models pick `submitter_id`, `id` or `bcr_id`, which is arguably reasonable).
  Tumor_Site, Tumor_Site_Other and Tumor_Size_cm are wrong in only some runs.
- The Ethnicity_Self_Identify -> ethnicity "error" under the old ground truth is correct under GDC-SM.
- `full` mode sees only column names, no sample values, yet reaches 11-12/14. The model may have memorized
  GDC field names; a renamed-target control is needed before trusting such numbers.

Caveat: one study and two runs only show that the setup is usable; these are not final results.

## 4. Magneto + local Qwen (`results/gdc_sm/magneto_*.txt`; first runs in `results/magneto_*.txt`)

- Config of the runs below: `Magneto(use_gpt_reranker=True, llm_model="ollama_chat/qwen2.5:14b", llm_model_kwargs={"api_base": "http://localhost:11434", "temperature": 0})`.
- The script now defaults to `llm_model="openai/qwen2.5:14b"` with `api_base` `http://localhost:11434/v1`, which also returns logprobs. A rerun that way gave the same ranking for every column (GDC-SM top-1 9/14, MRR 0.708; 396s).
- Pitfall: the latest `valentine==1.0.1` fails with `'MatcherResults' object has no attribute 'pop'`; use `valentine==0.5.0`.
- clark results:

| Setting | GDC-SM top1 | GDC-SM MRR | Old GT top1 | Old GT MRR | Time |
|---|---|---|---|---|---|
| No LLM (mpnet embeddings + bipartite reranker) | 7/14 | 0.591 | 6/14 | 0.551 | 14s |
| + qwen2.5:14b reranker | 9/14 | 0.708 | 7/14 | 0.625 | 414s |

These are reruns that score the same matches against both ground truths. The old-GT numbers are identical to
the first runs (6/14, 0.551 and 7/14, 0.625; first reranker run took 425s), so with `temperature=0` this rerun reproduced exactly.
LLM output parse failures: 0.
