# Benchmark data

The data stays local and is not tracked by git: `.gitignore` ignores `datasets/*`, so only this README is committed. Sources, versions and licenses for each directory are listed below. If data is missing, download it with the steps in "Re-fetching".

## Files for the main experiments (GDC suite)

| Purpose | What to use | Path |
|---|---|---|
| Column-level ground truth | GDC-SM v1.0: Zenodo 14963588, 2025-04, 259 pairs, n:m | `gdc_sm/extracted/data/ground-truth/<Study>.csv` |
| Value-level ground truth | GDC-VM: Zenodo 18557064, 2026-02-09, 3,904 rows (3,902 unique), 227 attribute pairs | `gdc_vm/<Study>.csv` |
| Source tables (not on Zenodo) | GDC_new from VIDA's data-harmonization-benchmark | `GDC_new/<study>/source.csv` |
| Target table (column level, 736 columns) | GDC target. Byte-identical to `GDC_new/<study>/target.csv` (md5 `53ff20f7…`) | `gdc_sm/extracted/data/target-tables/gdc_unique_columns_concat_values.csv` |
| Permissible target values (value level) | GDC-VM's `gdc_schema.json`: GDC v3.1.0, 736 attributes, 353 of them enumerated, 16,659 values in total; same content as the copy bundled with BDI-Kit | `gdc_vm/gdc_schema.json` |

**Do not use** `GDC_new/<study>/groundtruth.csv`. It is an older 197-row GT that lacks 1:n correspondences and has 1 duplicate row, 3 targets with leading spaces, and 1 nonexistent target, `purity`. Use it only to compare against old results.

**Versions to cite:** GDC-SM v1.0 (Zenodo 14963588, 2025-04) + GDC-VM (Zenodo 18557064, 2026-02-09), with source tables from GDC_new in VIDA-NYU/data-harmonization-benchmark (main, commit 3207e37).

**Checked on 2026-11-02:**
- Every source column name in GDC-SM's 259 pairs appears in `source.csv`, and every target column name is in the 736-column target table.
- All 158 GDC-VM source attributes appear in `source.csv`.
- Every categorical target value in GDC-VM is a valid value in `gdc_schema.json`.

## Pitfalls

1. **Inconsistent number formats.** `source.csv` stores integers as decimals (e.g. `1031.0`), while GDC-VM writes `1031`. Of 2,873 distinct (attribute, source value) pairs, exact comparison matches only 2,009; after normalizing to integers, 2,869 match. The remaining 4 come from manual edits in the Clark and Wang source tables. Normalize formats before value-level scoring; see `num()` below.
2. **Read GDC-VM with `dtype=str, keep_default_na=False`.** An empty `target_value` means there is no suitable GDC value; an empty `source_value` means the source cell is missing. With default parsing both become NaN and cannot be told apart.
3. **A source value can have several correct targets.** For example, missing values map to {Not Reported, Unknown}, and capitalization varies by attribute. Score against the gold set: hitting any member counts as correct; when the gold set is empty, only answering "no match" is correct.
4. **Report numeric tasks separately.** 39 identity-copy pairs (1,933 matches) are easy and inflate the overall score. 10 `age_at_diagnosis` pairs need conversion to days: years × 365.25, months × 30.4375. Vasaikar's `Age`, Krug's `Age.in.Month` and McDermott's `Age in Months…` are all in months.
5. **Inconsistent study-name spelling.** The GDC_new directory is `gilette`; GDC-SM and GDC-VM use `Gillette`.
6. **Target-table "rows" are not records.** Each column stacks that attribute's permissible values from top to bottom, and 380 of the 736 columns are empty. Do not sample or shuffle target-table rows.
7. **Column names contain dots.** The 5 affected source tables have 178 such names (e.g. `Tumor.Stage`); quote them with backticks in Spark.
8. **BDI-Kit calls a cloud model by default.** Its LLM matchers default to `openai/gpt-4o-mini`. To run local Qwen, point LiteLLM at Ollama explicitly.
9. **LLMs may have memorized the GDC vocabulary** (see MetaHarmonizer). Add a control condition with renamed targets to the main experiments.

## Directories

| Directory | Contents | Source / license | Role |
|---|---|---|---|
| `GDC_new/` | `source.csv`, `target.csv` and the old `groundtruth.csv` for 10 CPTAC studies | `datasets/GDC_new.tar.gz` from VIDA-NYU/data-harmonization-benchmark (Apache-2.0; source-table license not verified) | Main: source tables |
| `gdc_sm/` | `gdc-sm-data.zip`, extracted `extracted/data/{ground-truth,target-tables}`, `papers_info.json` | Zenodo 14963588, CC-BY-4.0 | Main: column-level GT and target table |
| `gdc_vm/` | 10 `<Study>.csv` files, `gdc_schema.json`, `gdc_schema_flattened.csv` | Zenodo 18557064, CC-BY-4.0 | Main: value-level GT and permissible values |
| `gdc_source_xlsx/` | Supplementary-table xlsx files from the 10 CPTAC papers, the original origin of `source.csv` | Publisher supplementary materials, license not verified | Provenance |
| `valentine/Valentine-datasets/` | All 551 GT and config files, plus 41 complete tasks (Magellan 7, Wikidata 4, synthetic 30) | Zenodo 5084605, CC-BY-4.0. Files were read with range requests; the full 602 MB archive was not downloaded | Secondary: easy control |
| `omop_schema_matching/` | SMAT/OMAP, ReMatch MIMIC_2_OMOP, Parciak MIMIC-IV→OMOP, LLMatch SchemaNet | GitHub. ReMatch and Parciak are MIT-licensed; the others have no license | Optional: cross-target transfer |
| `bio_ml_ncit_doid/` | OAEI Bio-ML 2024 NCIT-DOID (zip and extracted files) | Zenodo 13119437, CC BY 4.0 | Optional: stress test |

In addition, `third_party/data-harmonization-benchmark/` is a clone of VIDA's evaluation repository (main branch, commit 3207e37), kept only as a code reference. Note that its evaluation scripts keep only the metrics from the last round.

## Loading

```python
import json
import pandas as pd

ROOT = "datasets"  # run from the big-data repo root
STUDIES = ["Cao", "Clark", "Dou", "Gillette", "Huang", "Krug", "McDermott", "Satpathy", "Vasaikar", "Wang"]
FOLDER = {s: ("gilette" if s == "Gillette" else s.lower()) for s in STUDIES}  # directory names in GDC_new


def num(v):
    """Normalize '1031.0' to '1031', matching GDC-VM's format."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else v
    except ValueError:
        return v


study = "Dou"
src = pd.read_csv(f"{ROOT}/GDC_new/{FOLDER[study]}/source.csv", dtype=str, keep_default_na=False)
target = pd.read_csv(f"{ROOT}/gdc_sm/extracted/data/target-tables/gdc_unique_columns_concat_values.csv",
                     dtype=str, keep_default_na=False, na_values=[""])
sm_gt = pd.read_csv(f"{ROOT}/gdc_sm/extracted/data/ground-truth/{study}.csv", dtype=str)
# columns: original_paper_variable_names, GDC_format_variable_names
vm_gt = pd.read_csv(f"{ROOT}/gdc_vm/{study}.csv", dtype=str, keep_default_na=False)
# columns: source_attribute, target_attribute, source_value, target_value
schema = json.load(open(f"{ROOT}/gdc_vm/gdc_schema.json"))
permissible = {a: list((v.get("value_data") or {}).keys()) for a, v in schema.items()}

# Value-level gold: (source attribute, target attribute, source value) -> set of correct target values; empty set means "no match"
gold = (vm_gt.groupby(["source_attribute", "target_attribute", "source_value"]).target_value
        .apply(lambda s: frozenset(x for x in s if x)))
attr = vm_gt.source_attribute.iloc[0]
values_to_map = src[attr].map(num).unique()  # normalize number formats before looking up gold
```

## Re-fetching

```bash
cd datasets
mkdir -p gdc_sm gdc_vm
# GDC-SM (Zenodo 14963588)
curl -L -o gdc_sm/gdc-sm-data.zip "https://zenodo.org/records/14963588/files/gdc-sm-data.zip?download=1"
curl -L -o gdc_sm/papers_info.json "https://zenodo.org/records/14963588/files/papers_info.json?download=1"
unzip -q gdc_sm/gdc-sm-data.zip -d gdc_sm/extracted
# GDC-VM (Zenodo 18557064)
for f in Cao Clark Dou Gillette Huang Krug McDermott Satpathy Vasaikar Wang; do
  curl -L -o "gdc_vm/$f.csv" "https://zenodo.org/records/18557064/files/$f.csv?download=1"
done
curl -L -o gdc_vm/gdc_schema.json "https://zenodo.org/records/18557064/files/gdc_schema.json?download=1"
curl -L -o gdc_vm/gdc_schema_flattened.csv "https://zenodo.org/records/18557064/files/gdc_schema_flattened.csv?download=1"
# GDC_new source tables (VIDA-NYU/data-harmonization-benchmark)
curl -L -o GDC_new.tar.gz "https://github.com/VIDA-NYU/data-harmonization-benchmark/raw/main/data_harmonization_benchmark/datasets/GDC_new.tar.gz"
tar -xzf GDC_new.tar.gz
```

Sources for the optional data:

- Valentine: <https://zenodo.org/records/5084605> (full archive 602 MB).
- The four OMOP sets: <https://github.com/JZCS2018/SMAT>, <https://github.com/meniData1/MIMIC_2_OMOP>, <https://github.com/UHasselt-DSI-Data-Systems-Lab/code-schema-matching-LLMs-artefacs>, <https://github.com/knowledge-fusion/LLMatch>.
- Bio-ML: <https://zenodo.org/records/13119437> (`ncit-doid.zip`).
