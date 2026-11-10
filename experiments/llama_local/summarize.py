"""Summarize a clark_match sweep: accuracy per study and mode, and whether the repeats of each agree.

Usage: ~/venvs/harmonize/bin/python -I experiments/llama_local/summarize.py [OUT]   (default ~/hbench_raw/llama_eval)

Reads OUT/summary.jsonl (one line per run, written by clark_match.py) and each run's per-column table
OUT/raw/<run_id>/items.gdc-sm.csv. Repeats of one configuration are runs whose tags differ only in _run<N>.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "hbench_raw" / "llama_eval").expanduser()
lines = [json.loads(l) for l in (out / "summary.jsonl").read_text().splitlines() if l.strip()]
if not lines:
    sys.exit(f"no runs in {out / 'summary.jsonl'}")

groups = defaultdict(list)
for s in lines:
    groups[re.sub(r"_run\d+$", "", s["tag"])].append(s)

rows = []
for config, runs in sorted(groups.items()):
    runs.sort(key=lambda s: s["tag"])
    preds = []
    for s in runs:
        items = out / "raw" / s["run_id"] / "items.gdc-sm.csv"
        if items.exists():
            preds.append(pd.read_csv(items, dtype=str, keep_default_na=False).set_index("item_id")["pred"])
    changed = None
    if len(preds) >= 2:
        both = pd.concat(preds, axis=1).dropna()
        changed = f"{int((both.nunique(axis=1) > 1).sum())}/{len(both)}"
    rows.append({"config": config, "runs": len(runs),
                 "accuracy (gold cols)": " / ".join(s["acc_gold_cols"] for s in runs),
                 "candidate recall": runs[0]["shortlist_recall"],
                 "columns changed between runs": changed or "-",
                 "secs": " / ".join(f"{s['total_secs']:.0f}" for s in runs)})

table = pd.DataFrame(rows)
with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 60):
    print(table.to_string(index=False))

# Totals per mode over all studies, first run of each configuration
tot = defaultdict(lambda: [0, 0])
for config, runs in groups.items():
    mode = "shortlist" if "_shortlist" in config else "full"
    c, n = map(int, sorted(runs, key=lambda s: s["tag"])[0]["acc_gold_cols"].split("/"))
    tot[mode][0] += c
    tot[mode][1] += n
print()
for mode, (c, n) in sorted(tot.items()):
    print(f"{mode:9s} run 1, all studies: {c}/{n} = {c / n:.1%}")
