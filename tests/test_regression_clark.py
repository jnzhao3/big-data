"""The shared scorer must reproduce the committed qwen_local numbers (old ground truth and GDC-SM)."""

import json
import re

import pandas as pd
import pytest

from hbench.adapters.clark_match import from_clark_match_csv
from hbench.adapters.magneto import from_magneto_txt
from hbench.score_sm import score_sm

CONFIGS = ["shortlist20_qwen2.5-14b", "full_qwen2.5-14b", "shortlist20_qwen3.8-27b-q4_K_M", "full_qwen3.8-27b-q4_K_M"]


def _summaries(path):
    return {json.loads(line)["tag"]: json.loads(line) for line in path.read_text().splitlines() if line.strip()}


@pytest.mark.parametrize("variant,summary", [("gdc-new-raw", "summary.jsonl"), ("gdc-sm", "gdc_sm/summary.jsonl")])
def test_clark_runs_rescore_identically(data, results_dir, variant, summary):
    saved = _summaries(results_dir / summary)
    csvs = sorted(results_dir.glob("clark_*.csv"))
    assert len(csvs) == 8
    for p in csvs:
        _, s = score_sm(from_clark_match_csv(p), data, "clark", variant)
        w = saved[p.stem]
        assert f"{s['top1_correct']}/{s['n_annotated']}" == w["acc_gold_cols"], p.stem
        assert f"{s['closed_world_correct']}/{s['n_columns']}" == w["acc_all_incl_NONE"], p.stem
        assert s["hallucinated"] == w["statuses"].get("hallucinated", 0), p.stem


def test_changed_columns_between_runs(data, results_dir):
    changed = {}
    for cfg in CONFIGS:
        preds = [score_sm(from_clark_match_csv(results_dir / f"clark_{cfg}_run{r}.csv"), data, "clark")[0]
                 .set_index("source_column").pred for r in (1, 2)]
        changed[cfg] = int((preds[0] != preds[1]).sum())
    assert changed == {"shortlist20_qwen2.5-14b": 1, "full_qwen2.5-14b": 5, "shortlist20_qwen3.8-27b-q4_K_M": 3,
                       "full_qwen3.8-27b-q4_K_M": 5}


@pytest.mark.parametrize("name,old_top1,sm_top1,sm_mrr", [("magneto_nollm.txt", 6, 7, 0.591),
                                                          ("magneto_ollama_qwen2.5-14b.txt", 7, 9, 0.708)])
def test_magneto_saved_runs(data, results_dir, name, old_top1, sm_top1, sm_mrr):
    text = (results_dir / "gdc_sm" / name).read_text()
    section = text.split("--- scored against GDC-SM")[1].split("--- scored against old GT")[0]
    recs = from_magneto_txt(section, "magneto", "clark")
    _, old = score_sm(recs, data, "clark", "gdc-new-raw")
    _, new = score_sm(recs, data, "clark", "gdc-sm")
    assert (old["top1_correct"], new["top1_correct"]) == (old_top1, sm_top1)
    ranks = [None if x == "None" else int(x) for x in re.findall(r"gold_rank=(\S+)", section)]
    assert round(sum(1 / r for r in ranks if r) / len(ranks), 3) == sm_mrr  # MRR definition: first gold hit


def test_gdc_sm_only_adds_correct_cells(data, results_dir):
    flips = []
    for p in sorted(results_dir.glob("clark_*.csv")):
        recs = from_clark_match_csv(p)
        old = score_sm(recs, data, "clark", "gdc-new-raw")[0].set_index("source_column").closed_world_correct
        new = score_sm(recs, data, "clark", "gdc-sm")[0].set_index("source_column").closed_world_correct
        flips += [(p.stem, c, bool(old[c]), bool(new[c])) for c in old.index if old[c] != new[c]]
    assert len(flips) == 11 and all(not o and n for _, _, o, n in flips)
    assert pd.Series([c for _, c, _, _ in flips]).value_counts().to_dict() == {
        "Ethnicity_Self_Identify": 7, "Tumor_Size_cm": 3, "Tumor_Site": 1}
