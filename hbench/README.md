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
```

How the logs are written:
- Nothing is overwritten: run folders and event files are created exclusively.
- Every event is flushed and fsynced, and readers skip a truncated last line.
- Field values over 4 KB go to the blob store.
- Messages are stored once each, so an agent's repeated history costs disk space once.

Event types:
- `run_header`, from `collect_run_header()`: git commit and dirty flag, Python flags (including whether `PYTHONHASHSEED` took effect), package versions, Ollama version, model digest, template hash and currently loaded models, environment variables (names only for anything secret-like), data manifest check, prompt hashes, and the experimental condition.
- `llm_call`: full request and response, parameters, token counts, latency, Ollama timings including `load_duration` (cold or warm), and `input_sha256` / `output_sha256` to find where two runs diverge.
- `tool_call`, `agent_step`, `state_snapshot`: for agents.
- `prediction_batch`, `score`, `run_end`.

Capturing LLM calls:
- Direct Ollama calls: `hbench.runlog.llm.ollama_chat(body, log, caller)`.
- Anything that goes through LiteLLM (Magneto's reranker, BDI-Kit matchers, agents):
  - Call `hbench.runlog.litellm_cb.install(log)`. It records the request body actually sent; LiteLLM routes rewrite messages, e.g. `ollama_chat/` drops assistant tool calls.
  - It flags calls to a non-local `api_base` as `cloud_call`.

`python -m hbench log validate <run_dir>` checks four things:
- every process starts with a header;
- sequence numbers are contiguous;
- the blobs exist;
- the run ended and no call left the machine.

Not yet built (phase 2, with the agent driver for E3):
- logged BDI-Kit MCP server wrapper (tool calls, state snapshots, cache control);
- LangChain step callbacks;
- first-divergence and prefix-replay tools.
