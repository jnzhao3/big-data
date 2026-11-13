# hbench internals

How the run log, evidence, engine audit, lineage and replay work. This is for analyzing why runs differ: engine noise (E1), replay fidelity (E5) and interventions (E6). For running experiments, [`README.md`](README.md) is enough.

## Run log format

One folder per run under the raw-results root (`runlog.session.raw_root()`):

```
runs_index.jsonl                       start/end line per run
_blobs/sha256/ab/<hash>.json.gz        content-addressed payloads (messages, large results), written once
<run_id>/events.<proc>.jsonl           one JSON event per line, one file per process
<run_id>/predictions.jsonl, items.*.csv
<run_id>/engine_audit.json             the run's requests as the Ollama server saw them, with each call's lineage
<run_id>/replay_compare.csv            replay runs only
```

How the logs are written:
- Nothing is overwritten: run folders and event files are created exclusively.
- Every event is flushed and fsynced, and readers skip a truncated last line.
- Field values over 4 KB go to the blob store, except an llm_call's `output`, which stays inline (long thinking answers pass 4 KB). Calls logged before that may hold a blob ref there; read it with `hbench.llm.record.load_output(event, blobs)`, as `evidence_from_call` and `replay.compare` do.
- Messages are stored once each, so an agent's repeated history costs disk space once.

Event types:
- `run_header`, from `collect_run_header()` (called by `start_run()`):
  - git commit and dirty flag. For a dirty tree, `code_snapshot` also holds the diff against the commit and the untracked source files (small text files only; names that look like secrets are skipped). `python -m hbench log code <run_dir> --out <dir>` writes them out, with the commands that rebuild the tree. `log validate` reports `code.recoverable`: false for a dirty run from before snapshots, which recorded only a hash of its diff;
  - Python flags, including whether `PYTHONHASHSEED` took effect;
  - package versions;
  - the engine snapshot and its drift against `env/engine-lock.json`;
  - environment variables (names only for anything secret-like);
  - data manifest check, prompt hashes;
  - the condition and its `condition_id`;
  - `effective_decoding`: the stated decoding over the endpoint's defaults for the condition's model (below), with the seed a `per_replicate` condition resolves to.
- `llm_call`, schema `hbench.llm_call/3` (below).
- `engine_action`: an unload, with the protocol that asked for it.
- `tool_call`, `agent_step`, `state_snapshot`: for agents.
- `prediction_batch`, `score`, `run_end`. `run_end` carries the engine audit summary.

## The `llm_call` event

Every call, from `LLMClient` or from LiteLLM (`runlog.litellm_cb`), is one `llm_call` in schema `hbench.llm_call/3`. Readers also accept `/2` events, which lack `prompt_sha256`.

| Part | Content |
|---|---|
| raw | `request_raw` (the exact body sent) and `response_raw`, both as blobs. Normalization can be redone from them at any time: `python -m hbench llm rederive <run_dir>` |
| decoding | the parameters actually sent; `null` if not sent |
| engine | `prompt_tokens`, `cached_tokens`, `computed_tokens`, `load_ms`, and `path_class`: `cold_full` / `full` / `partial` / `single_token`. A coarse label; the lineage (below) is the full account |
| output | text, tool calls, finish reason, and `token_logprobs`, a blob of `[token, logprob, [[candidate, logprob], ...]]`. Tool-call arguments are parsed into objects on both endpoints, so the same call compares and hashes the same on both (`/v1` sends a re-serialized JSON string) |
| hashes | `input_sha256` (what determines generation), `prompt_sha256` (model, messages, tools, format and `think`, without sampling settings: the same prompt at T=0 and at T=0.8 shares it; tool-call ids are renamed `call_0`, `call_1`, ... in order of appearance first, because Ollama gives each tool call a random id that the next request sends back. The qwen2.5 template does not render ids, and on qwen3.8-27b T=0 agent episodes with different random ids were bitwise identical. Calls logged before 2026-11-12 hashed the raw ids), `request_sha256` (exact bytes), `output_sha256` (text and tool calls), `logprobs_sha256` (bitwise identity) |
| `decision_spec` | where the method's decision is in the output, when the caller stated it |
| `model_sampler` | the model's own sampler settings (its Modelfile `PARAMETER` lines, from the run header's engine snapshot), when the run recorded them |
| `unavailable` | what this transport could not provide, and why |

## Margins and reproduction probability (`hbench.decisions`, `hbench.sampler`)

A decision is a value taken from the generated tokens. A tool call's tokens are the raw `<tool_call>{...}</tool_call>` text, even though the message content is empty.

At T=0 a decision changes only if a perturbation (another computation lineage, another prompt) moves some gap by more than its size, at any position up to the end of the value. A token that flips before the value changes everything after it.
- `margin` is the smallest gap, between the chosen token's logprob and the best other candidate's, over positions 0 to the end of the value (`margin_def: "prefix/1"`).
  - It is a guarantee: perturbations smaller than it everywhere cannot change the decision.
  - Positions after the value, such as a reason text, never count.
- `margin_value` is the same minimum over the value's own tokens (the older definition), kept as a diagnostic. Records without `margin_def` use the older definition for `margin`.
- The logprobs are the model's distribution before any constraint, and a `format=json` grammar can rule out the top candidate (for example `}` chosen over `}\n\n`). At T=0 the chosen token is the best one allowed, so at such a position:
  - the gap is taken to the best visible candidate below it, or 0 if there is none;
  - the count goes to `n_constrained`;
  - the margin stays a guarantee, only more conservative.

Reproduction probability:
- llama-server samples through top-k, then top-p on the T=1 probabilities, then min-p, then temperature (its log prints the chain).
  - So under `/api/chat` defaults (T=0.8, top_p=0.9) a position whose top token has probability at least 0.9 is deterministic.
  - `/v1` defaults (T=1.0, top_p=1.0) give no such protection.
- `sampler.p_path(tokens, end, params)` is the probability that sampling with `params` reproduces the tokens through position `end`.
  - It is exact when the top-p nucleus lies within the visible candidates; the result says so.
  - It returns `None`, with the reason, when another sampler stage (a penalty, typical, xtc, dry, mirostat) is on.
- `sampler.effective(decoding, endpoint, model=...)` is the sampler a call ran with. From the top:
  - what the call stated;
  - on `/v1`, temperature and top_p: Ollama's OpenAI layer sends both in every request, 1.0 when the request leaves them out;
  - the model's own settings, the `PARAMETER` lines of its Modelfile (`/api/show` `parameters`). qwen3.8-27b-bdikit sets temperature 1, top_k 20, top_p 0.95; qwen2.5:14b sets none. Every run header's engine snapshot records them (`modelfile_parameters`), and every `llm_call` carries them as `model_sampler`;
  - the server's defaults in `env/engine-lock.json`.
- `sampler.defaults(endpoint, model=...)` is the same without stated settings: what `p_path_default` samples with. For qwen3.8-27b-bdikit that is T 1.0, top_k 20, top_p 0.95 on `/api/chat`, and T 1.0, top_k 20, top_p 1.0 on `/v1`.
- With a grammar (`format` or a JSON schema) every `p_path` is a lower bound. Where a T=0 call shows that a constraint ruled out higher-scoring candidates, those are left out of the computation.

Stability over runs at T>0 (`stability.sampling_check`):
- P is the probability of the greedy path. It is only a lower bound on the probability of the decision, so an item can change across n runs with probability at most 1 - P^n.
- The check is one-sided. It raises an alarm when more items changed than that bound allows, beyond three standard deviations.
- Its pairwise interval [Σ2P(1-P), Σ1-P²] is a range only when the tokens before each decision are fixed structure, as in clark's `{"target": "`.

## The engine (`hbench.engine`)

- **`engine_snapshot()`** records three things:
  - Ollama's API state;
  - what only the server log knows: the server's `OLLAMA_*` configuration, each model's llama-server flags (KV-cache precision, flash attention, context, batch sizes), and the host-RAM prompt cache size;
  - the server log's current byte offset.
- **`env/engine-lock.json`** pins the Ollama version, model digests and templates, and server-level runner invariants.
  - Context and batch sizes are not pinned by default: they follow each caller's `num_ctx` and the free RAM, and are recorded per run and per call instead. Ollama 0.40 picks `-b` 1024 or 512 from free RAM when it loads a model, and on qwen3.8-27b the two give bitwise-different logprobs. A model whose Modelfile sets `PARAMETER num_batch` loads with a fixed batch; its lock entry then states `loaded_batch` (`{"b": 1024, "ub": 1024}`), by hand, and regenerating the lock keeps it. Request-level `options.num_batch` works only on `/api/chat`; `/v1` honors only the Modelfile.
  - `sampler_defaults` records what each endpoint uses for settings a request leaves out, as read from audited calls. `/api/chat` uses T 0.8, top_k 40, top_p 0.9; `/v1` uses T 1.0, top_k 40, top_p 1.0; min_p is 0 on both. Regenerating the lock (`python -m hbench engine lock --write --force`) keeps them.
- **The audit** (`python -m hbench audit <run_dir>`, run automatically when a `start_run` run closes) parses the server-log bytes between the run's start and end offsets and joins them with the run's calls. Per call it recovers:
  - the effective sampler parameters, every one compared with `sampler.effective` for the call and its model;
  - reused and recomputed tokens, RAM-cache restores, truncation and HTTP status;
  - the computation lineage;
  - `batch`: the `-b`/`-ub` and model blob of the runner that served it, from the last `starting llama-server` line before it (read from before the run when the model was already loaded). Calls of a model with `loaded_batch` in the lock that ran otherwise are `batch_mismatches`, a problem.

  `log validate` recomputes that comparison (and `batch_mismatches`, against the current lock) from `engine_audit.json`, so audits written before the model's own settings were taken into account (they flagged top_k 20 against 40 on qwen3.8-27b-bdikit) are judged by the current rule.

  It also checks the engine protocol: under `cold_start`, a runner start before the first call with nothing reused; under `cold`, a runner start before every call. It flags calls whose server-side sampler ran a stage `p_path` cannot model (`sampler_unmodelled`). Requests that match none of the run's calls are foreign: another client used the server.

## Lineage (`hbench.engine.lineage`)

At T=0 the output is a deterministic function of the input and of how the KV of every context position was computed:
- `B`: a batch of prompt tokens, with the batch size of that forward pass;
- `S`: one prompt token alone, the last token recomputed when the whole prompt was cached;
- `G`: one generated token at a time.

A position also depends on how everything before it was computed, so a single-token recompute is inherited by every later call that reuses it. `path_class` cannot see that.

How the audit reconstructs it:
- It replays the server's prompt state from the log, starting at the last runner start before the run, so state left by earlier runs is known.
- It follows runner starts, prompt-cache saves, skipped saves, evictions and restores. A restored entry moves out of the cache into the slot.
- It follows each task's reused and computed tokens, chunked by the runner's `-b`/`-ub`.

The result per call is a canonical text such as `B237 G16 B54` and a `fingerprint` under a versioned rule. Rule `v1`:
- batches of at least 54 tokens are equivalent however they split (verified for 54 to 1024);
- smaller batches, `S` and `G` are kept apart;
- `?` marks positions the log does not show.

Uncertain cases are listed, not guessed:
- an unknown start, an ambiguous or out-of-sync restore, several runners alive at once, a task without counts;
- the doubt is inherited: a later call that reuses positions computed under uncertainty is marked `inherits an uncertain lineage`, until a runner start clears the state.

Checks:
- `python -m hbench lineage check <runs or roots>` groups T=0 calls by input, engine and fingerprint. Within a group the logprobs must be bitwise identical; a violation means an unknown noise source or engine drift.
  - It also lists inputs whose different fingerprints gave identical logprobs, which is evidence for relaxing the rule.
  - Per run, it reports how many T=0 calls were checked, and how many were left out as uncertain or without a lineage.
  - `log validate` runs the same check within a run.
- Runs audited before lineage existed: `python -m hbench lineage rebuild <runs> --out-dir <dir>` writes `<run>.lineage.json` without touching the run folders, and `lineage check --lineage-dir <dir>` reads them.

## Replay (`hbench.replay`)

`python -m hbench replay <run_dir> --calls 0 --protocol cold resend after_unrelated --out <root>` re-sends logged requests exactly, after preparing the engine state:

| Protocol | Engine state |
|---|---|
| `cold_start` | model unloaded once before the replay run, then the calls replayed in order. A source run that itself started cold gets its whole lineage back. Only a prefix `0..k` of the calls may be replayed |
| `cold` | model unloaded before each call |
| `resend` | the same request sent once before, so only its last token is recomputed |
| `after_unrelated` | an unrelated request sent first |
| `as_is` | unchanged |

Sampler settings can be overridden, to resample logged requests at T>0:
- `--set temperature=0.8` (repeatable): temperature, top_p, top_k, and min_p on `/api/chat`.
- `--seeds 1-15`: one replay run per seed, all replicates of one condition. `--set seed=N` is the same as `--seeds N`.
  - Each sampled call gets its own seed by the rule natural runs use (Seeds, below). A call whose logged seed was derived from the same base is sent that seed, so `--seeds s` sends exactly the seeds a natural run with base seed s sent. `replay_compare.csv` has both `seed` and `call_seed`.
  - Without a seed, a sampled source call is resent with its logged seed, and a greedy source call resampled at T>0 takes its logged seed as the base.
  - `--repeat` reuses each call's seed: repeats are seeded repeats (does a seed reproduce?), not independent samples. Use `--seeds` for those.
- Under `cold_start` every `--repeat` is its own run, so each repeat really starts cold.

`replay_compare.csv` has one row per decision of each replayed call, keyed by `source_call` (the call's index in the source run). A tool call's name and its arguments are separate decisions. Each row compares the replay with its source:
- text, first diverging token, logprob shifts;
- each decision's margin and whether it flipped;
- `lineage_source`, `lineage_replay` and `lineage_same`, which says whether bitwise identity is predicted;
- with overrides, `p_path_source`: the probability that the replay's sampler reproduces the source's decision path. The summary prints observed decision changes against what the source's logprobs allow.

## Seeds (`hbench.seeds`)

llama-server reseeds its sampler with each request's seed. Calls sent with the same seed draw the same random number at the same token position, so their T>0 decisions are sampled with correlated draws: when the draw at a position is small, every call picks its top candidate there. E1 Stage 3's first smoke run sent one seed on every call and saw 0 decision changes in 80 samples where `p_path` expected 22.8.

Each decision's own probability of changing is unaffected, and so is the expected number of changes. What breaks is everything that assumes independent samples: variances, intervals, the `sampling_check` alarm.

The rule (`call/1`):
- The seed a condition or caller states is a base. Every sampled call (temperature above 0, or unset) is sent with `call_seed(base, scope, prompt_sha256, k)`: the first 31 bits of a SHA-256. prompt_sha256 ignores tool-call ids, so a seeded agent episode at T>0 repeats call for call.
  - `scope` is the process and the caller, such as `driver/method:clark_match`, so two processes of one run that send the same prompt get different seeds.
  - `k` counts that process's earlier calls with the same base, scope and prompt.
- Calls are independent, a run is reproducible from its base, and the same prompt in two conditions gets the same seed (common random numbers).
- Greedy calls keep the stated seed: the sampler does not use it, and T=0 requests stay byte-identical to earlier runs.
- `LLMClient.chat`, the LiteLLM callback and replay apply it. The LiteLLM callback replaces the seed in the request body LiteLLM is about to send. `LLMClient.send` sends a body exactly as given.
- A process attached with `RunLog.attach` takes the run's condition from the driver's run header, so its clients follow the same rule and settings.
- A derived call's `llm_call` event has `seed_base` and `seed_rule`; `decoding.seed` is the seed actually sent.
- `decoding.seed_mode: "shared"` turns the rule off for a condition.

Checks, from the log alone, whatever route sent the calls:
- `check_seeds`, run by `log validate`, per run:
  - a problem: sampled calls with different prompts share one seed that was not derived (derived seeds that collide do so by chance);
  - notes: the condition shares seeds on purpose; a prompt was sampled more than once with the same seed (seeded repeats); sampled calls sent no seed (independent, but not reproducible); the condition states a fixed integer seed at T>0.
  - `independent` is False after a shared seed or a seeded repeat.
- `check_replicates`, across the runs of one condition: two runs that sampled the same prompt with the same seed are copies, not independent replicates. A fixed integer seed at T>0 does this, and so do replay repeats without `--seeds`.
- `python -m hbench stability` uses both. When they find dependence, `sampling_check` keeps `bound_expected` (an expectation, valid under any dependence) and drops the alarm and the pairwise interval.
