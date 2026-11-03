# hbench: shared scoring and run logging

Every method in the project writes its predictions in one format, is scored by the same rules, and logs its LLM calls the same way. Then numbers from different runs, methods and people can be compared, and every run can be traced afterwards. How the evidence, engine audit, lineage and replay work inside is in [`INTERNALS.md`](INTERNALS.md); you need it only to investigate why two runs differ.

## Quick start

Most of hbench runs by itself: when a run closes, it audits the engine and records how each call was computed (its lineage), and `python -m hbench log validate` checks the run. Run every command from the repo root with `~/venvs/harmonize/bin/python` (written `python` below).

**1. Score predictions you already have** (no LLM):

```bash
python -m hbench preds validate my_run/predictions.jsonl
python -m hbench score my_run/predictions.jsonl --gt gdc-sm     # writes scores.gdc-sm.json and items.gdc-sm.csv
```

**2. Log a method that calls a local model.** This runs as written; `experiments/qwen_local/clark_match.py` is a fuller version:

```python
import json
from hbench.data import Data
from hbench.design import make_condition
from hbench.llm import LLMClient
from hbench.predictions import sm_record, write_predictions
from hbench.runlog.session import raw_root, start_run

data = Data()                                                   # $HBENCH_DATA, else <repo>/datasets
targets = data.target_columns                                   # the 736 GDC columns

def make_prompt(col):
    return (f'Source column: "{col}". Target columns: {", ".join(targets)}.\n'
            'Answer with JSON only: {"target": "<one target column, or NONE>"}')

cond = make_condition(method={"name": "my_method"}, model={"name": "qwen2.5:14b"},
                      decoding={"temperature": 0, "seed": 1},   # stated once, here
                      engine={"transport": "ollama", "protocol": "cold_start"},
                      data={"study": "clark"}, replicate=1)
log = start_run(raw_root(), "my_exp", cond, task={"kind": "sm"}, models=["qwen2.5:14b"])
client = LLMClient(log, "qwen2.5:14b")                          # takes its decoding from cond
records = []
for col in data.source_columns("clark"):
    res = client.chat([{"role": "user", "content": make_prompt(col)}], caller="method:my_method",
                      format="json", decision={"type": "json_field", "field": "target"})
    target = json.loads(res.text or "{}").get("target")
    records.append(sm_record(log.run_id, "clark", col, [] if target in (None, "NONE") else [target],
                             shown="*", status="ok" if target else "invalid_output", evidence=res.evidence()))
write_predictions(log.run_dir / "predictions.jsonl", records)
log.close()
```

Then score and check the run. Its folder is `log.run_dir`, under `~/hbench_raw/`:

```bash
python -m hbench score <run_dir>/predictions.jsonl --gt gdc-sm
python -m hbench log validate <run_dir>
```

- `res.evidence()` records, from the call itself, how close the decision was to flipping and how likely sampling is to reproduce it.
- For an agent, name a tool call as the decision: `decision={"type": "tool_call", "field": "arguments.column"}`.
- A method that calls the model through LiteLLM (Magneto, BDI-Kit) installs `hbench.runlog.litellm_cb` on the log and uses LiteLLM's `openai/` route, as `experiments/qwen_local/magneto_ollama.py` does. `hbench.decisions.evidence_from_call(event, log.blobs)` then gives the same evidence from a logged call.

**3. Value matching.** With the `data`, `log` and `client` of example 2, the items to map come from `data.vm_items(study)`, and predictions are `vm_record`s:

```python
from hbench.predictions import vm_record

items, _ = data.vm_items("clark")             # one row per (source attribute, target attribute, source value)
for it in items.itertuples():
    res = client.chat([{"role": "user", "content": ...}], caller="method:my_vm", format="json",
                      decision={"type": "json_field", "field": "value"})
    value = json.loads(res.text or "{}").get("value")
    records.append(vm_record(log.run_id, "clark", it.source_attribute, it.target_attribute, it.key,
                             [] if value in (None, "NONE") else [value], evidence=res.evidence()))
```

`data.permissible` gives each GDC attribute's allowed values. Score with `python -m hbench score ... --vm-mode oracle` when the method was given the attribute pairs, `--vm-mode end_to_end` when it chose them itself.

**4. Repeat a condition and measure stability.** Give each repeat its own `replicate` and keep `engine.protocol` `cold_start`, so every repeat starts from the same engine state. At T>0, sample a different seed per repeat with `decoding={"temperature": 0.8, "seed": "per_replicate"}`. Then:

```bash
python -m hbench stability runA/items.gdc-sm.csv runB/items.gdc-sm.csv runC/items.gdc-sm.csv
```

The tests never take the real Ollama lease or read the real server log, so `python -m pytest` can run while an experiment does.

## Data

`hbench.data.Data()` reads the benchmark from `$HBENCH_DATA`, else `<repo>/datasets`, laid out as in `datasets/README.md`:
- `source(study)`, `source_columns(study)`: a study's source table and its columns.
- `target_columns` (a list), `target_table()`: the 736 GDC target columns.
- `sm_gold(study)`: column → set of correct targets (GDC-SM).
- `vm_items(study)`, `permissible` (a dict): the value-matching items and each attribute's allowed values (GDC-VM).

Studies are `cao clark dou gillette huang krug mcdermott satpathy vasaikar wang`. Any spelling works (`gilette`, `Gillette`).

## Prediction format (`predictions.jsonl`)

There is one decision per line, written with `hbench.predictions.sm_record` / `vm_record` and `write_predictions`:

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
| `shown` | candidates the method saw: a list, `"*"` for the whole GDC schema, or `null` if unknown. Needed to tell "picked an unshown column" from "picked a wrong one" |
| `status` | `ok`, `invalid_output` or `error`; raw model output belongs in the run log |
| `evidence` | optional, from `res.evidence()`: which call decided it and how close the decision was (see "Decisions and evidence") |

Existing outputs convert with `python -m hbench preds convert <file> --format clark-match|magneto-txt|bdikit-sm-csv|bdikit-vm-csv --out predictions.jsonl` (add `--run-id`, `--study`).

## Scoring

```bash
python -m hbench score <run_dir>/predictions.jsonl --gt gdc-sm --also gdc-new-raw   # scores.<gt>.json, items.<gt>.csv
```

From Python: `hbench.score_sm.score_sm(records, Data(), "clark")` and `hbench.score_vm.score_vm(...)` return a per-item table and a summary. Every summary carries `scoring_version` and the ground-truth id.

**Schema matching (SM)**
- Ground truth: GDC-SM v1.0. `gdc-new-raw` is the old 197-row file, used only to compare with old numbers.
- A prediction is correct if it is any member of the column's gold set. Target names compare case-insensitively; names outside the 736 GDC columns count as `hallucinated`.
- Primary metrics use annotated columns only: top-1 accuracy, MRR (rank of the first correct target), hit@1/3/5/10/20.
- Unannotated columns are reported apart: `abstained` when the method answers NONE, `unannotated_claims` when it names a target. `closed_world_correct` treats them as "should be NONE".
- Also reported for comparison with published numbers: VIDA's pair-averaged MRR, Valentine's recall@GT, and set-output precision, recall and F1.

**Value matching (VM)**
- Ground truth: GDC-VM. An item is (source attribute, target attribute, canonical source value); only values present in the source table count.
- Missing cells are the value `''`; an empty gold set means only "no match" is correct.
- Report by category, never as one headline number:

| Category | Correct when |
|---|---|
| `categorical` | exact match with a gold value (GDC codes are case-sensitive); also reported leniently |
| `numeric_identity` | numerically equal |
| `age_conversion` | within 1 day; also reported exactly and leniently (2% relative error) |
| `no_legal_target` | the method answers "no match" (trivially correct for many matchers, so excluded from `all_but_no_legal_target`) |

**Stability**
- `python -m hbench stability` groups item tables by condition and reports the prediction flip rate, Fleiss' kappa, raw agreement, pass@k and pass^k, accuracy mean ± sd, and always/sometimes/never-correct counts. Prediction-level metrics need no ground truth.
- Flips are also broken down by decision margin. For runs at T>0, a sampling check asks whether the runs changed more than sampling alone allows.
- `stability.majority_vote()` combines k runs into one virtual run.

## Runs

**Conditions.** `make_condition` takes the namespaces `method`, `model`, `decoding`, `engine` and `data`, plus `replicate`:
- `decoding.temperature` must always be stated: the server default differs by endpoint (0.8 on `/api/chat`, 1.0 on `/v1`).
- `decoding.seed` is an integer, `null`, or `"per_replicate"` (each replicate samples with its replicate number as seed).
- Replicates of one condition share a `condition_id`, which is how `stability` groups them.

**Engine protocols** (`engine.protocol`):

| Protocol | What happens | Use |
|---|---|---|
| `natural` (default) | nothing; the server is used as it is | single runs |
| `cold_start` | the model is unloaded once before the run | repeated runs: every repeat starts from the same state |
| `cold` | the model is unloaded before every call | rarely; slow. Only with `LLMClient`, not LiteLLM |

`start_run` takes an Ollama lease (two of our runs never overlap), writes the run header, carries out the protocol, and audits the run when it closes.

**Run folders** go to `$HBENCH_RAW`, else `~/hbench_raw`; the scripts' `--raw-dir` overrides it. A run folder holds `events.*.jsonl` (the log), `predictions.jsonl`, `items.*.csv`, and `engine_audit.json`. Nothing in it is ever overwritten.

**Checking a run.** `python -m hbench log validate <run_dir>` reports a problem when:
- the log is incomplete;
- a call left the machine, or is missing from the server log;
- the engine differs from `env/engine-lock.json`;
- another client used Ollama during the run;
- a call was truncated, or ran with other sampler settings than requested (or with a sampler stage hbench cannot model);
- the protocol was not carried out;
- two T=0 calls computed the same way gave different results.

A run whose engine could not be checked, for example because the server log is missing, is reported as a problem too.

## Decisions and evidence

A method says where its decision is when it calls the model, and the spec is stored with the call:

| Spec | The decision is |
|---|---|
| `{"type": "json_field", "field": "target"}` | one field of a JSON output |
| `{"type": "text"}` | the whole output text |
| `{"type": "tool_call", "field": "name"}` | the name of a tool call; `"arguments.column"` for one argument (dots for nesting), `"*"` for the whole call; `"index": 1` for the second call |

A list of specs names several decisions of one call. `res.evidence()` gives the evidence of the first one; `res.evidence(key="arguments.column")` picks another. Calls through LiteLLM carry no spec; register a parser for their caller instead: `register_decision_parser("method:x", json_field_parser("target"))`.

The evidence fields:
- `margin`: how far the decision was from flipping at T=0, as a logprob gap. In E1, changing the engine state moved gaps by at most 0.33, so decisions with larger margins never flipped.
- `p_path_default`: the probability that sampling at each endpoint's default temperature reproduces the decision.
- For runs at T>0: `p_path` under the run's own sampler, and `greedy_path`.

`INTERNALS.md` has the exact definitions.

## LLM routes

- **`hbench.llm.LLMClient`** calls Ollama's `/api/chat` (`transport="ollama"`) or `/v1` (`transport="openai_v1"`). It requests token logprobs by default, logs every call, and with a log from `start_run` takes its decoding from the run's condition.
- **LiteLLM's `openai/` route** with `api_base="http://localhost:11434/v1"` and `hbench.runlog.litellm_cb` logs the same information. Use it for Magneto and agents.
- **The `ollama_chat/` route** cannot return logprobs, so its calls get no margins.

## Commands

| Command | Does |
|---|---|
| `python -m hbench manifest verify` | checks your data against `datasets/MANIFEST.json` |
| `python -m hbench preds validate\|convert` | checks or converts a predictions file |
| `python -m hbench score` | scores a predictions file |
| `python -m hbench stability` | stability metrics over several runs' item tables |
| `python -m hbench log validate` | checks a run folder |
| `python -m hbench engine check` | compares your Ollama with `env/engine-lock.json` |
| `python -m hbench audit`, `lineage check\|rebuild`, `replay`, `llm rederive` | analysis tools, see `INTERNALS.md` |

## Not yet built

- A run-plan executor: conditions × replicates in randomized order, with the plan and what actually ran both recorded.
- With the BDI-Kit agent driver: a logged MCP server wrapper, LangChain step callbacks, and first-divergence search across two agent runs.
