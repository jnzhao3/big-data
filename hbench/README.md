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

The tests never take the real Ollama lease or read the real server log (`tests/conftest.py` points `HBENCH_LEASE` and `HBENCH_OLLAMA_LOG` at a temporary folder), so they can run while an experiment does.

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
| `evidence` | optional: the `llm_call` that made the decision (`span_id`), how close it was to flipping at T=0 (`margin`, `margin_value`, `runner_up`, `n_constrained`), and how likely sampling is to reproduce it (`p_path_default`, plus `p_path` and `greedy_path` for runs at T>0); see "Decisions and margins" |
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
- For runs at T>0, `sampling_check` asks whether the runs changed more than sampling allows (see "Decisions and margins"). Each item's P comes from a run that followed the greedy path. Failing that, it comes from a T=0 reference run: `python -m hbench stability <items...> --reference <T=0 items.csv> --endpoint /api/chat`.

## Run logs

`hbench.runlog.RunLog` writes one folder per run under a raw-results root. That root is `runlog.session.raw_root()`: `$HBENCH_RAW` if set, otherwise `~/hbench_raw`. Raw logs never go into the repo, because Qwen's top-k candidates in them contain Chinese tokens. The scripts' `--raw-dir` overrides the root.

```
runs_index.jsonl                       start/end line per run
_blobs/sha256/ab/<hash>.json.gz        content-addressed payloads (messages, large results), written once
<run_id>/events.<proc>.jsonl           one JSON event per line, one file per process
<run_id>/predictions.jsonl, items.*.csv
<run_id>/engine_audit.json             the run's requests as the Ollama server saw them, with each call's lineage (start_run runs that use Ollama)
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
  - the condition and its `condition_id`, and `effective_decoding`: the stated decoding over the endpoint's defaults (see "The engine"), with the seed a `per_replicate` condition resolves to.
- `llm_call`, schema `hbench.llm_call/3` (below).
- `engine_action`: an unload, with the protocol that asked for it (see "Starting a run").
- `tool_call`, `agent_step`, `state_snapshot`: for agents.
- `prediction_batch`, `score`, `run_end`. `run_end` carries the engine audit summary.

### Starting a run

```python
from hbench.design import make_condition
from hbench.runlog.session import raw_root, start_run

cond = make_condition(method={"name": "clark_match", "mode": "full"}, model={"name": "qwen2.5:14b"},
                      decoding={"temperature": 0, "seed": 1}, engine={"transport": "ollama", "num_ctx": 32768},
                      data={"study": "clark"}, replicate=1)
log = start_run(raw_root(), "e1", cond, task={"kind": "sm"}, models=["qwen2.5:14b"], prompts=PROMPTS)
...
log.close()   # audits the server log for this run and releases the Ollama lease
```

`start_run` does five things:
- takes an advisory Ollama lease, so two of our runs never interleave;
- writes the header;
- records the server log's byte offset;
- performs the start action of the engine protocol (below);
- registers the audit and the lease release as close hooks.

Conditions use fixed namespaces: `method`, `model`, `decoding`, `engine`, `data`. The `replicate`, `block` and `order` keys sit outside them.

`decoding.temperature` must always be stated, because the server default differs by endpoint (T=0.8 on `/api/chat`, T=1.0 on `/v1`). `decoding.seed` is an integer, `null`, or `"per_replicate"`: each replicate then samples with its own seed (its replicate number), and the replicates still share one condition.

Replicates of one condition share a `condition_id`, so `python -m hbench stability` groups them automatically; it reads the id from the header of each item table's run folder.

`engine.protocol` says how the engine state is prepared. Each protocol has a start action and a per-call action:

| Protocol | Start action | Per-call action | Use |
|---|---|---|---|
| `natural` (default) | none | none | whatever state the server is in |
| `cold_start` | unload the model, which also empties the host-RAM prompt cache | none | every replicate starts from the same state. Back-to-back repeats of short prompts otherwise run on a different lineage from the second run on (E1 Stage 1e) |
| `cold` | none | unload before every call | full recompute each time (slow) |

- `start_run` performs the start action for every route. A start action needs the run's `models`, and `start_run` refuses one without them.
- `LLMClient` performs the per-call action. Routes it does not drive (LiteLLM) cannot take a per-call protocol, and `start_run` refuses one for them.
- Every unload is logged as an `engine_action`.
- The audit checks that the protocol really happened: under `cold_start`, a runner start before the first call and nothing reused; under `cold`, a runner start before every call.

### LLM calls (`hbench.llm`)

`LLMClient(log, model, transport="ollama" | "openai_v1")` is one client for Ollama's native `/api/chat` and its OpenAI-compatible `/v1`:
- it requests logprobs by default (`top_logprobs=5`);
- `num_ctx` defaults to 32768, the context `/v1` uses for qwen2.5:14b, so both endpoints share one loaded runner and the model does not reload;
- it performs the per-call action of the run's engine protocol.

Every call, from this client or from LiteLLM (`runlog.litellm_cb`), is written as one `llm_call` in schema `hbench.llm_call/3`. Readers also accept `/2` events, which lack `prompt_sha256`:

| Part | Content |
|---|---|
| raw | `request_raw` (the exact body sent) and `response_raw`, both as blobs. Normalization can be redone from them at any time: `python -m hbench llm rederive <run_dir>` |
| decoding | the parameters actually sent; `null` if not sent |
| engine | `prompt_tokens`, `cached_tokens`, `computed_tokens`, `load_ms`, and `path_class`: `cold_full` / `full` / `partial` / `single_token`. At T=0 the computation path shifts logprobs (an identical resend recomputes only its last token), so this is a covariate in every run-to-run comparison |
| output | text, tool calls, finish reason, and `token_logprobs`, a blob of `[token, logprob, [[candidate, logprob], ...]]`. Tool-call arguments are parsed into objects on both endpoints, so the same call compares and hashes the same on both (`/v1` sends a re-serialized JSON string) |
| hashes | `input_sha256` (what determines generation), `prompt_sha256` (model, messages, tools, format and `think`, without sampling settings: the same prompt at T=0 and at T=0.8 shares it), `request_sha256` (exact bytes), `output_sha256` (text and tool calls), `logprobs_sha256` (bitwise identity) |
| `decision_spec` | where the method's decision is in the output, when the caller stated it (see below) |
| `unavailable` | what this transport could not provide, and why |

What each route provides:
- LiteLLM's `openai/` route provides logprobs and cache counts; use it for agents and Magneto.
- `ollama_chat/` rejects logprobs and drops cached-token counts, and its `unavailable` field says so.

### Decisions and margins (`hbench.decisions`, `hbench.sampler`)

A decision is a value taken from the generated tokens, whichever channel carries it:
- the message text: one JSON field, or the whole text;
- a tool call: its name, one argument, or the whole call.

A tool call's tokens are the raw `<tool_call>{...}</tool_call>` text, even though the message content is empty.

At T=0 a decision changes only if a perturbation (another computation lineage, another prompt) moves some gap by more than its size, at any position up to the end of the value. A token that flips before the value changes everything after it. So:
- `margin` is the smallest gap, between the chosen token's logprob and the best other candidate's, over positions 0 to the end of the value (`margin_def: "prefix/1"`).
  - It is a guarantee: perturbations smaller than it everywhere cannot change the decision.
  - Positions after the value, such as a reason text, never count.
- `margin_value` is the same minimum over the value's own tokens. That was the definition before E0 v3; it is kept as a diagnostic.
- The logprobs are the model's distribution before any constraint. A `format=json` grammar can rule out the top candidate: E1 Stage 1 shows 12 such positions, all `}` chosen over `}\n\n`. At T=0 the chosen token is the best one allowed, so at such a position:
  - the gap is taken to the best visible candidate below it, or 0 if there is none;
  - the count goes to `n_constrained`;
  - the margin stays a guarantee, only more conservative.

Specifying a decision:
- A method states where its decision is when it calls the model, e.g. `client.chat(..., decision={"type": "json_field", "field": "target"})`.
  - The spec is stored on the `llm_call` as `decision_spec`, so replay and analysis tools find the decision from the log alone, without the method's code.
  - Spec types: `json_field`; `text`; `tool_call` with `field` `name`, `arguments.<path>` or `*` (the whole call), and `index` (which call, default 0).
  - A list of specs names several decisions of one call. A tool call's decisions are keyed by their field, with `@<index>` added from the second call on (`name`, `name@1`), so replay pairs them call by call.
  - A tool-call decision is found from the tokens alone, even when the reply carries no text (OpenAI-style `content: null`).
- Calls made through LiteLLM carry no spec. For those, register a parser for the caller:
  - `register_decision_parser("method:x", json_field_parser("target"))`, or `tool_call_parser("name")` for agent tool calls. Parsers built by hbench read the whole output.
  - A plain function of the text also works.

Reproduction probability (`hbench.sampler`):
- llama-server samples through top-k, then top-p on the T=1 probabilities, then min-p, then temperature.
  - So under `/api/chat` defaults (T=0.8, top_p=0.9) a position whose top token has probability at least 0.9 is deterministic.
  - `/v1` defaults (T=1.0, top_p=1.0) give no such protection.
- `sampler.p_path(tokens, end, params)` is the probability that sampling with `params` reproduces the tokens through position `end`.
  - It is exact when the top-p nucleus lies within the visible candidates; the result says so.
  - It returns `None`, with the reason, when another sampler stage (a penalty, typical, xtc, dry, mirostat) is on.
- `sampler.effective(decoding, endpoint)` is the sampler a call ran with: what it stated, over the endpoint's defaults in `env/engine-lock.json`.

Evidence:
- `evidence(span_id, token_logprobs, text, value, json_key, *, decision=None, temperature=None, sampler_params=None, **extra)` builds the `evidence` dict of a prediction record:
  - `margin`, `margin_value`, `margin_def`, `runner_up`, `n_constrained`;
  - `p_path_default`: the reproduction probability under each endpoint's default sampler, with `exact`. A T=0 run thereby says how the same decision behaves at default temperature.
  - For a call above T=0, also `p_path` (under its own sampler), `p_path_exact` and `greedy_path` (every token up to the end was the top candidate). A sampler stage `p_path` cannot model, such as a requested `repeat_penalty`, gives `p_path: null` and the reason in `p_path_note`.
  - With `grammar=True` (the call used `format` or a JSON schema) every `p_path` is marked `lower_bound`. Where a T=0 call shows a constraint ruled out higher-scoring candidates, those are left out of the computation. A constraint only removes competitors, so the true value is at least this.
- The scorers carry these into item tables (`decisions.evidence_columns`). A v2 record keeps its `margin` and has no `margin_value`.
- `stability_report` adds margin quantiles and `flips_by_margin`; `fragile_items(long, eps)` counts items whose margin is below a measured perturbation size.
- `sampling_check` is one-sided. P is the probability of the greedy path, only a lower bound on the probability of the decision, so an item can change across n runs with probability at most 1 - P^n. It raises an alarm when more items changed than that allows. Its pairwise interval [Σ2P(1-P), Σ1-P²] is a range only when the tokens before each decision are fixed structure, as in clark's `{"target": "`.

### The engine (`hbench.engine`)

- **`engine_snapshot()`** records three things:
  - Ollama's API state;
  - what only the server log knows: the server's `OLLAMA_*` configuration, each model's llama-server flags (KV-cache precision, flash attention, context, batch sizes), and the host-RAM prompt cache size;
  - the server log's current byte offset.
- **`env/engine-lock.json`** pins the Ollama version, model digests and templates, and server-level runner invariants. Context and batch sizes are not pinned: they follow each caller's `num_ctx` and are recorded per run instead.
  - `sampler_defaults` records what each endpoint uses for settings a request leaves out. `/api/chat` uses T 0.8, top_k 40, top_p 0.9; `/v1` uses T 1.0, top_k 40, top_p 1.0; min_p is 0 on both. These come from the audited sampler parameters of E1 Stage 1's 1234 calls. Regenerating the lock keeps them.
  - `python -m hbench engine check` reports drift.
  - `python -m hbench engine lock --write --force` updates the lock deliberately.
- **`python -m hbench audit <run_dir>`** (run automatically at close by `start_run`) parses the server-log bytes between the run's start and end offsets and joins them with the run's calls. Per call it recovers:
  - the effective sampler parameters, every one compared with what the call asked for over its endpoint's defaults;
  - reused and recomputed tokens, RAM-cache restores, truncation and HTTP status;
  - the computation lineage (below).

  It also checks the engine protocol, and flags calls whose server-side sampler had a stage `p_path` cannot model (`sampler_unmodelled`, for example a penalty set in a Modelfile). Then it writes `engine_audit.json`. Requests that match none of the run's calls are foreign: another client used the server.
- **Lineage (`hbench.engine.lineage`).** At T=0 the output is a deterministic function of the input and of how the KV of every context position was computed:
  - `B`: a batch of prompt tokens, with the batch size of that forward pass;
  - `S`: one prompt token alone, the last token recomputed when the whole prompt was cached;
  - `G`: one generated token at a time.

  A position also depends on how everything before it was computed, so a single-token recompute is inherited by every later call that reuses it. `path_class` cannot see that.
  - The audit reconstructs each call's lineage by replaying the server's prompt state from the log. It starts reading at the last runner start before the run, so state left by earlier runs is known. It follows runner starts, prompt-cache saves, evictions and restores, and each task's reused and computed tokens, chunked by the runner's `-b`/`-ub`.
  - The result per call is a canonical text such as `B237 G16 B54` and a `fingerprint` under a versioned rule. Rule `v1`:
    - batches of at least 54 tokens are equivalent however they split, which E1 verified for 54 to 1024;
    - smaller batches, `S` and `G` are kept apart;
    - `?` marks positions the log does not show.
  - Uncertain cases are listed, not guessed: an unknown start, an ambiguous or out-of-sync restore, several runners alive at once, a task without counts.
  - The doubt is inherited: a later call that reuses positions computed under uncertainty is marked `inherits an uncertain lineage`, until a runner start clears the state.
  - `python -m hbench lineage check <runs or roots>` groups T=0 calls by input, engine and fingerprint. Within a group the logprobs must be bitwise identical; a violation means an unknown noise source or engine drift.
    - It also lists inputs whose different fingerprints gave identical logprobs, which is evidence for relaxing the rule.
    - Per run, it reports how many T=0 calls were checked, and how many were left out as uncertain or without a lineage.
    - `log validate` runs the same check within a run.
  - Runs audited before E0 v3: `python -m hbench lineage rebuild <runs> --out-dir <dir>` writes `<run>.lineage.json` without touching the run folders, and `lineage check --lineage-dir <dir>` reads them.

### Replay (`hbench.replay`)

`python -m hbench replay <run_dir> --calls 0 --protocol cold resend after_unrelated --out <root>` re-sends logged requests exactly, after preparing the engine state:

| Protocol | Engine state |
|---|---|
| `cold_start` | model unloaded once before the replay run, then the calls replayed in order. A source run that itself started cold gets its whole lineage back. Only a prefix `0..k` of the calls may be replayed |
| `cold` | model unloaded before each call |
| `resend` | the same request sent once before, so only its last token is recomputed |
| `after_unrelated` | an unrelated request sent first |
| `as_is` | unchanged |

Sampler settings can be overridden, to resample logged requests at T>0:
- `--set temperature=0.8` (repeatable; temperature, seed, top_p, top_k, and min_p on `/api/chat`);
- `--seeds 1-15`: one replay run per seed, all replicates of one condition.
- Under `cold_start` every `--repeat` is its own run, so each repeat really starts cold.

`replay_compare.csv` has one row per decision of each replayed call (a tool call's name and its arguments are separate decisions), keyed by `source_call` (the call's index in the source run). It compares each replay with its source:
- text, first diverging token, logprob shifts;
- each decision's margin and whether it flipped;
- `lineage_source`, `lineage_replay` and `lineage_same`, which says whether bitwise identity is predicted;
- with overrides, `p_path_source`: the probability that the replay's sampler reproduces the source's decision path. The summary prints observed decision changes against what the source's logprobs allow.

This is the instrument behind E1 (lineage-induced noise and T>0 calibration), E5 (replay fidelity) and E6.

### Checking a run

`python -m hbench log validate <run_dir>` checks that:
- every process starts with a header;
- sequence numbers are contiguous;
- the blobs exist;
- the run ended;
- no call left the machine;
- the engine matches the lock;
- the audit shows no foreign requests, no unlogged calls, no truncation, the requested sampler settings, and the engine protocol carried out;
- T=0 calls with the same input and lineage have identical logprobs.

A run whose engine could not be checked or audited, for example because the server log is missing, is reported as a problem, not passed.

## Not yet built

- A run-plan executor: conditions × replicates in randomized order, with the plan and what actually ran both recorded. To be designed with E1, whose factors decide its shape.
- With the agent driver for E3:
  - a logged BDI-Kit MCP server wrapper (tool calls, state snapshots, cache control);
  - LangChain step callbacks;
  - first-divergence search across two agent runs.
