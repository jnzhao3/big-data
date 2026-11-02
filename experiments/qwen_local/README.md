# Local Qwen feasibility test (2026-11-02, Freya)

Goal: check before the team meeting whether local Qwen models (via Ollama) are usable for our
LLM data-harmonization agent experiments.
Test machine: MacBook Pro M1 Max 64GB, Ollama 0.40.0.

**Conclusion**: usable. Recommended main model: `qwen2.5:14b`. Its accuracy is about the same as the 27B model,
it is 3-6x faster, and tool calling works. Use the 27B model only for a few comparison runs.
The bottleneck is candidate-column recall, not the model.

## Files

| File | Purpose |
|---|---|
| `bench_speed.sh` | Load time and generation speed per model |
| `test_openai_tools.py` | Checks the OpenAI-compatible endpoint (`localhost:11434/v1`) and tool calling |
| `clark_match.py` | Column matching test: for each source column, the model picks one target column from the candidates; scored against groundtruth |
| `run_all.sh` | 2 models x 2 candidate modes x 2 runs |
| `magneto_ollama.py` | Runs Magneto (VIDA-NYU) with its LLM reranker pointed at local Ollama |
| `requirements-magneto.txt` | Dependencies for the separate Magneto venv (includes the valentine version pin) |
| `results/` | All outputs from this test (see below) |

## Setup

1. Install Ollama and pull the model: `ollama pull qwen2.5:14b` (for the 27B comparison: `qwen3.8:27b-q4_K_M`, 17GB).
2. Data is not in git. Download `GDC_new` from
   [VIDA-NYU/data-harmonization-benchmark](https://github.com/VIDA-NYU/data-harmonization-benchmark)
   and extract it to `datasets/GDC_new/` at the repo root (ignored by `.gitignore`),
   or point `--data-dir` / the `GDC_DATA_DIR` environment variable somewhere else.
3. `clark_match.py` only needs pandas (tested with Python 3.11).

```bash
./bench_speed.sh qwen2.5:14b qwen3.8:27b-q4_K_M
python test_openai_tools.py qwen2.5:14b
python -I clark_match.py --model qwen2.5:14b --mode shortlist --run 1
PYTHON=python ./run_all.sh            # all 8 configs, about 20 minutes
```

Magneto (separate venv; run from the repo root; `third_party/` and `.venv*/` are ignored by `.gitignore`):

```bash
git clone https://github.com/VIDA-NYU/magneto-matcher third_party/magneto-matcher
python -m venv .venv_magneto
.venv_magneto/bin/pip install -e third_party/magneto-matcher/algorithms/magneto -r experiments/qwen_local/requirements-magneto.txt
.venv_magneto/bin/python -I experiments/qwen_local/magneto_ollama.py --llm ollama_chat/qwen2.5:14b
```

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

## 3. clark column matching (`results/clark_*.csv`, `results/summary.jsonl`)

Data: the clark study in GDC_new. Source: 110 rows x 16 columns (CPTAC clear cell renal cell carcinoma clinical data).
Target: 736 columns (GDC fields; the columns that have values mostly hold the list of allowed values for that field).
Groundtruth: 15 pairs covering 14 source columns.
Case_ID and Tumor_Site_Other have no matching column, so the expected answer is NONE.
Tumor_Site has two correct targets; picking either counts as correct.

Two candidate modes:
- `shortlist`: 20 candidates chosen by column-name similarity + value overlap (fuzzy containment), shown with sample values. Candidate recall 11/14.
- `full`: all 736 target column names in the prompt (no sample values); the shared prefix lets Ollama reuse the KV cache. Recall 14/14.

Model default temperature (temperature not set), `format=json`, `think=false`.

| Config | run1 | run2 | Columns that changed between runs | Time run1/run2 |
|---|---|---|---|---|
| shortlist + qwen2.5:14b | 8/14 | 8/14 | 1/16 | 62s / 31s |
| full + qwen2.5:14b | 9/14 | 11/14 | 5/16 | 54s / 35s |
| shortlist + qwen3.8:27b | 10/14 | 9/14 | 3/16 | 187s / 179s |
| full + qwen3.8:27b | 8/14 | 9/14 | 5/16 | 242s / 200s |

Failure modes:
- All outputs were valid JSON; no invalid output.
- Hallucinated column names only appeared in `full` mode: 2 in 14b run1 (`anatomic_site`, `histology`).
  (A lowercase `none` in 27b run2 was originally counted as hallucinated; scoring is now case-insensitive and the results were rescored.)
- Common errors: Ethnicity_Self_Identify -> ethnicity (groundtruth says race, which is debatable);
  Case_ID -> submitter_id (not in groundtruth, but semantically reasonable);
  Histologic_Type -> primary_diagnosis is almost never found;
  Tumor/Normal -> tissue_type is only found by Magneto's embedding retrieval.

Caveat: one study and two runs only show that the setup is usable; these are not final results.

## 4. Magneto + local Qwen (`results/magneto_*.txt`)

- Config: `Magneto(use_gpt_reranker=True, llm_model="ollama_chat/qwen2.5:14b", llm_model_kwargs={"api_base": "http://localhost:11434", "temperature": 0})`.
- Pitfall: the latest `valentine==1.0.1` fails with `'MatcherResults' object has no attribute 'pop'`; use `valentine==0.5.0`.
- clark results:

| Setting | top1 | MRR | Time |
|---|---|---|---|
| No LLM (mpnet embeddings + bipartite reranker) | 6/14 | 0.551 | 14s |
| + qwen2.5:14b reranker | 7/14 | 0.625 | 425s |

LLM output parse failures: 0.
