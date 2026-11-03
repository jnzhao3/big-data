# hbench: shared scoring and run logging (E0)

Every experiment, method and track scores with the same rules and writes the same prediction and log formats.
Then numbers from different runs, tools and people can be compared, and every run can be traced afterwards.
The design and its rationale are in the team's E0 design doc; this file covers how to use the code.

## Setup

- Environment: `env/setup_env.sh` installs `hbench` in editable mode.
- Without installing, run from the repo root: `python -m hbench ...`. Tests use `pyproject.toml`'s pythonpath.
- Data: `$HBENCH_DATA`, else `<repo>/datasets`, laid out as in `datasets/README.md`.
- `datasets/MANIFEST.json` (tracked) records sizes and md5 sums of the 54 files every score depends on:

```bash
python -m hbench manifest verify     # does my data match the manifest?
python -m pytest                     # unit tests; data-dependent tests skip when the data is missing
```

## Prediction format (`predictions.jsonl`)

There is one decision per line. Write it with `hbench.predictions.sm_record / vm_record / write_predictions`,
or convert existing outputs with `hbench.adapters` (clark_match CSV, Magneto, BDI-Kit DataFrames).

```json
{"schema": "hbench.pred/1", "run_id": "...", "kind": "sm", "study": "clark", "source_column": "Tumor_Site",
 "output_type": "ranked", "targets": ["site_of_resection_or_biopsy", "primary_site"], "scores": [0.81, 0.77],
 "shown": "*", "status": "ok"}
{"schema": "hbench.pred/1", "run_id": "...", "kind": "vm", "study": "clark", "source_attribute": "Race",
 "target_attribute": "race", "source_value": "White", "output_type": "top1", "targets": ["white"], "status": "ok"}
```

| Field | Meaning |
|---|---|
| `output_type` | `top1`, `ranked` or `set` |
| `targets` | `[]` means "no match" |
| `shown` | candidates the method saw: a list, `"*"` for the whole GDC schema, or `null` if unknown. Needed to tell "picked an unshown column" from "picked a wrong one", and to recompute candidate recall under another ground truth |
| `status` | `ok`, `invalid_output` or `error`; raw model output belongs in the run log |
| `evidence` | optional: the `llm_call` that made the decision (`span_id`) and how close it was to flipping (`margin`, `runner_up`, `p_flip`); see "Decisions and margins" |
| study names | any spelling (`gilette`, `Gillette`) |

`python -m hbench preds validate predictions.jsonl` checks a file.

## Scoring

```bash
python -m hbench score <run_dir>/predictions.jsonl --gt gdc-sm --also gdc-new-raw   # writes scores.<gt>.json, items.<gt>.csv
python -m hbench stability runA/items.gdc-sm.csv runB/items.gdc-sm.csv runC/items.gdc-sm.csv
```

From Python: `hbench.score_sm.score_sm(records, Data(), "clark")` and `hbench.score_vm.score_vm(...)`.
Each returns a per-item table and a summary. Every summary carries `scoring_version` (`hbench-score/1.0.0`) and
the ground-truth id.

**Schema matching (SM)**
- Ground truth: GDC-SM v1.0 by default. `gdc-new-raw` is the old 197-row file, stripped and deduplicated, used only to compare with old numbers.
- A prediction is correct if it is any member of the column's gold set.
- Primary metrics use annotated columns only: top-1 accuracy, MRR (rank of the first correct target), hit@1/3/5/10/20.
- Unannotated columns are reported apart: `abstained` when the method answers NONE, `unannotated_claims` to audit when it names a target.
- `closed_world_correct` also treats them as "should be NONE"; the old clark scripts reported this.
- Target names compare case-insensitively. Names outside the 736 GDC columns count as `hallucinated`.
- Also reported:
  - VIDA's pair-averaged MRR and Valentine's recall@GT, for comparison with published numbers;
  - set-output precision, recall and F1, with open-world and closed-world precision.

**Value matching (VM)**
- Ground truth: GDC-VM.
- An item is (source attribute, target attribute, canonical source value), and only values present in the source table count.
- Missing cells are the value `''`; an empty gold set means only "no match" is correct.
- Report by category, never as one headline number:

| Category | Correct when |
|---|---|
| `categorical` | exact match with a gold value (GDC codes are case-sensitive); also reported leniently |
| `numeric_identity` | numerically equal |
| `age_conversion` | within 1 day; also reported exactly and leniently (2% relative error) |
| `no_legal_target` | the method answers "no match" (trivially correct for many matchers, so excluded from `all_but_no_legal_target`) |

- String `"None"` is a value only for the 8 GDC attributes that permit it.

**Normalization**
- `canon_value()` strips, applies NFC, and maps NaN or None to `''`.
- `num()` turns `1031.0` into `1031`. It is purely lexical and never round-trips through float.

**Stability (RQ1)**
- `stability_report()` gives prediction flip rate, Fleiss' kappa, raw agreement, pass@k and pass^k (unbiased estimators), accuracy mean ± sd, and always/sometimes/never-correct counts.
- Prediction-level metrics need no ground truth.
- `majority_vote()` combines k runs into one virtual run.

## Run logs

`hbench.runlog.RunLog` writes one folder per run under a git-ignored root (e.g. `experiments/<exp>/results/raw/`):

```
runs_index.jsonl                       start/end line per run
_blobs/sha256/ab/<hash>.json.gz        content-addressed payloads (messages, large results), written once
<run_id>/events.<proc>.jsonl           one JSON event per line, one file per process
<run_id>/predictions.jsonl, items.*.csv
<run_id>/engine_audit.json             the run's requests as the Ollama server saw them (start_run runs that use Ollama)
<run_id>/replay_compare.csv            replay runs only
```

How the logs are written:
- Nothing is overwritten: run folders and event files are created exclusively.
- Every event is flushed and fsynced, and readers skip a truncated last line.
- Field values over 4 KB go to the blob store.
- Messages are stored once each, so an agent's repeated history costs disk space once.

Event types:
- `run_header`, from `collect_run_header()` (or `start_run()`, below):
  - git commit and dirty flag;
  - Python flags, including whether `PYTHONHASHSEED` took effect;
  - package versions;
  - the engine snapshot (below) and its drift against `env/engine-lock.json`;
  - environment variables (names only for anything secret-like);
  - data manifest check, prompt hashes;
  - the condition and its `condition_id`.
- `llm_call`, schema `hbench.llm_call/2` (below).
- `tool_call`, `agent_step`, `state_snapshot`: for agents.
- `prediction_batch`, `score`, `run_end`. `run_end` carries the engine audit summary.

### Starting a run

```python
from hbench.design import make_condition
from hbench.runlog.session import start_run

cond = make_condition(method={"name": "clark_match", "mode": "full"}, model={"name": "qwen2.5:14b"},
                      decoding={"temperature": 0, "seed": 1}, engine={"transport": "ollama", "num_ctx": 32768},
                      data={"study": "clark"}, replicate=1)
log = start_run(raw_root, "e1", cond, task={"kind": "sm"}, models=["qwen2.5:14b"], prompts=PROMPTS)
...
log.close()   # audits the server log for this run and releases the Ollama lease
```

`start_run` does four things:
- takes an advisory Ollama lease, so two of our runs never interleave;
- writes the header;
- records the server log's byte offset;
- registers the audit and the lease release as close hooks.

Conditions use fixed namespaces: `method`, `model`, `decoding`, `engine`, `data`. The `replicate`, `block` and `order` keys sit outside them. `decoding.temperature` must always be stated, because the server default differs by endpoint (T=0.8 on `/api/chat`, T=1.0 on `/v1`). Replicates of one condition share a `condition_id`, so `python -m hbench stability` groups them automatically; it reads the id from the header of each item table's run folder.

### LLM calls (`hbench.llm`)

`LLMClient(log, model, transport="ollama" | "openai_v1")` is one client for Ollama's native `/api/chat` and its OpenAI-compatible `/v1`:
- it requests logprobs by default (`top_logprobs=5`);
- `num_ctx` defaults to 32768, the context `/v1` uses for qwen2.5:14b, so both endpoints share one loaded runner and the model does not reload.

Every call, from this client or from LiteLLM (`runlog.litellm_cb`), is written as one `llm_call` in schema `hbench.llm_call/2`:

| Part | Content |
|---|---|
| raw | `request_raw` (the exact body sent) and `response_raw`, both as blobs. Normalization can be redone from them at any time: `python -m hbench llm rederive <run_dir>` |
| decoding | the parameters actually sent; `null` if not sent |
| engine | `prompt_tokens`, `cached_tokens`, `computed_tokens`, `load_ms`, and `path_class`: `cold_full` / `full` / `partial` / `single_token`. At T=0 the computation path shifts logprobs (an identical resend recomputes only its last token), so this is a covariate in every run-to-run comparison |
| output | text, tool calls, finish reason, and `token_logprobs`, a blob of `[token, logprob, [[candidate, logprob], ...]]` |
| hashes | `input_sha256` (what determines generation), `request_sha256` (exact bytes), `output_sha256` (text and tool calls), `logprobs_sha256` (bitwise identity) |
| `decision_spec` | where the method's decision is in the output, when the caller stated it (see below) |
| `unavailable` | what this transport could not provide, and why |

What each route provides:
- LiteLLM's `openai/` route provides logprobs and cache counts; use it for agents and Magneto.
- `ollama_chat/` rejects logprobs and drops cached-token counts, and its `unavailable` field says so.

### Decisions and margins (`hbench.decisions`)

A decision's margin is the smallest gap, over the tokens of the decided value, between the chosen token's logprob and the best other candidate's. Two caveats:
- The minimum over the whole output is the wrong quantity: it lands on punctuation or explanation tokens.
- At T=0 a decision can only flip when a perturbation exceeds its margin.

Usage:
- A method states where its decision is when it calls the model: `client.chat(..., decision={"type": "json_field", "field": "target"})`. The spec is stored on the `llm_call` as `decision_spec`, so replay and analysis tools find the decision from the log alone, without the method's code. Spec types: `json_field` (one field of a JSON output) and `text` (the whole output).
- Calls made through LiteLLM carry no spec. For those, register a parser for the caller: `register_decision_parser("method:x", json_field_parser("target"))`.
- `evidence(span_id, token_logprobs, text, value, json_key, temperature)` builds the `evidence` dict of a prediction record: `span_id`, `margin`, `runner_up`, and `p_flip` at the given temperature.
- The scorers carry `margin` and `evidence_span` into item tables.
- `stability_report` adds margin quantiles and `flips_by_margin`; `fragile_items(long, eps)` counts items whose margin is below a measured perturbation size.

### The engine (`hbench.engine`)

- **`engine_snapshot()`** records three things:
  - Ollama's API state;
  - what only the server log knows: the server's `OLLAMA_*` configuration, each model's llama-server flags (KV-cache precision, flash attention, context, batch sizes), and the host-RAM prompt cache size;
  - the server log's current byte offset.
- **`env/engine-lock.json`** pins the Ollama version, model digests and templates, and server-level runner invariants. Context and batch sizes are not pinned: they follow each caller's `num_ctx` and are recorded per run instead.
  - `python -m hbench engine check` reports drift.
  - `python -m hbench engine lock --write --force` updates the lock deliberately.
- **`python -m hbench audit <run_dir>`** (run automatically at close by `start_run`) parses the server-log bytes between the run's start and end offsets and joins them with the run's calls. Per call it recovers effective sampler parameters, reused and recomputed tokens, RAM-cache restores, truncation and HTTP status, and writes `engine_audit.json`. Requests that match none of the run's calls are foreign: another client used the server.

### Replay (`hbench.replay`)

`python -m hbench replay <run_dir> --calls 0 --protocol cold resend after_unrelated --out <root>` re-sends logged requests exactly, after preparing the engine state:

| Protocol | Engine state before the call |
|---|---|
| `cold` | model unloaded |
| `resend` | the same request sent once before, so only its last token is recomputed |
| `after_unrelated` | an unrelated request sent first |
| `as_is` | unchanged |

It compares each replay with its source: text, first diverging token, logprob shifts, and the decision margin with whether it flipped. The comparison goes to `replay_compare.csv`, one row per replayed call, keyed by `source_call` (the call's index in the source run). This is the instrument behind E1 (path-induced noise), E5 (replay fidelity) and E6.

### Checking a run

`python -m hbench log validate <run_dir>` checks that:
- every process starts with a header;
- sequence numbers are contiguous;
- the blobs exist;
- the run ended;
- no call left the machine;
- the engine matches the lock;
- the audit shows no foreign requests, no unlogged calls, no truncation and the requested temperatures.

A run whose engine could not be checked or audited, for example because the server log is missing, is reported as a problem, not passed.

## Not yet built

- A run-plan executor: conditions × replicates in randomized order, with the plan and what actually ran both recorded. To be designed with E1, whose factors decide its shape.
- With the agent driver for E3:
  - a logged BDI-Kit MCP server wrapper (tool calls, state snapshots, cache control);
  - LangChain step callbacks;
  - first-divergence search across two agent runs.
