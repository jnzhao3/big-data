"""Counts the scoring rules depend on, checked against the real files (skipped without the data)."""

from collections import Counter

from hbench.normalize import canon_value
from hbench.studies import STUDIES


def test_gdc_sm_ground_truth(data):
    pairs = {(s, a, b) for s in STUDIES for a, b in data.sm_pairs(s).itertuples(index=False)}
    assert len(pairs) == 259
    gold = {s: data.sm_gold(s) for s in STUDIES}
    assert sum(len(g) for g in gold.values()) == 165                      # annotated source columns
    assert sum(len(data.source_columns(s)) for s in STUDIES) == 569
    fanout = Counter(len(t) for g in gold.values() for t in g.values())
    assert fanout == {1: 99, 2: 46, 3: 14, 4: 4, 5: 2}
    targets = {t.casefold() for t in data.target_columns}
    assert len(data.target_columns) == 736 and all(t in targets for g in gold.values() for v in g.values() for t in v)
    assert all(c in data.source_columns(s) for s in STUDIES for c in gold[s])


def test_old_ground_truth_versus_gdc_sm(data):
    old = {(s, a, b) for s in STUDIES for a, b in data.sm_pairs(s, "gdc-new-raw").itertuples(index=False)}
    new = {(s, a, b) for s in STUDIES for a, b in data.sm_pairs(s).itertuples(index=False)}
    assert (len(old), len(old & new), len(old - new), len(new - old)) == (196, 188, 8, 71)
    assert {b for _, _, b in old} - set(data.target_columns) == {"purity"}
    clean = {(s, a, b) for s in STUDIES for a, b in data.sm_pairs(s, "gdc-new-clean").itertuples(index=False)}
    assert len(clean & new) == 189


def test_schema(data):
    perm = data.permissible
    assert len(perm) == 736 and sum(1 for v in perm.values() if v) == 353 and sum(len(v) for v in perm.values()) == 16659
    assert sorted(a for a, v in perm.items() if "None" in v) == sorted([
        "extrathyroid_extension", "lymph_node_involved_site", "cdc_hiv_risk_factors", "exercise_frequency_weekly",
        "hysterectomy_margins_involved", "immunosuppressive_treatment_type", "first_event", "risk_factor"])


def test_vm_items_and_categories(data):
    tasks, items, info = Counter(), Counter(), Counter()
    units = {}
    for s in STUDIES:
        df, inf = data.vm_items(s)
        items.update(df.category)
        for (sa, ta), t in inf["tasks"].items():
            tasks[t["category"]] += 1
            if t["age_unit"]:
                units[(s, sa)] = t["age_unit"]
        info.update({k: len(inf[k]) for k in ("gt_only_keys", "table_only_keys", "key_collisions")})
    assert tasks == {"categorical": 172, "numeric_identity": 39, "age_conversion": 10, "no_legal_target": 6}
    assert items == {"categorical": 1010, "numeric_identity": 1963, "age_conversion": 545, "no_legal_target": 245}
    assert info == {"gt_only_keys": 5, "table_only_keys": 2, "key_collisions": 1}
    assert {k for k, v in units.items() if v == "months"} == {
        ("vasaikar", "Age"), ("krug", "Age.in.Month"), ("mcdermott", "Age in Months at Time of Tissue Procurement")}


def test_vm_keys_found_after_canonicalization(data):
    exact = canon = total = 0
    for s in STUDIES:
        rows, src = data.vm_rows(s), data.source(s)
        for sa, grp in rows[["source_attribute", "source_value"]].drop_duplicates().groupby("source_attribute"):
            raw, keys = set(src[sa]), {canon_value(v) for v in src[sa]}
            for sv in grp.source_value:
                total += 1
                exact += sv in raw
                canon += canon_value(sv) in keys
    assert (exact, canon, total) == (2009, 2869, 2873)


def test_manifest_matches_data(data):
    from hbench.manifest import MANIFEST_PATH, verify_manifest
    assert MANIFEST_PATH.exists(), "run: python -m hbench manifest build"
    assert verify_manifest(data.root, MANIFEST_PATH) == []
