import json
import math

import pandas as pd
import pytest

from hbench.predictions import sm_record
from hbench.stability import (bootstrap_ci, fleiss_kappa, majority_vote, pairwise_disagreement, pass_at_k, pass_hat_k,
                              stability_classes, stability_report)


def test_pass_at_k_and_pass_hat_k():
    assert pass_at_k(5, 2, 1) == pytest.approx(0.4) and pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert pass_hat_k(5, 2, 1) == pytest.approx(0.4) and pass_hat_k(5, 2, 2) == pytest.approx(0.1)
    assert pass_at_k(5, 0, 3) == 0 and pass_hat_k(5, 5, 5) == 1 and pass_at_k(5, 4, 2) == 1


def test_fleiss_kappa_textbook_example():
    # worked example (10 subjects, 14 raters, 5 categories) with kappa = 0.210
    t = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
         [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert fleiss_kappa(t) == pytest.approx(0.210, abs=1e-3)


def test_fleiss_kappa_degenerate_and_invalid():
    assert math.isnan(fleiss_kappa([[3, 0], [3, 0]]))
    with pytest.raises(ValueError):
        fleiss_kappa([[2, 0], [1, 0]])


def test_pairwise_disagreement():
    assert pairwise_disagreement(["a", "a", "a"]) == 0
    assert pairwise_disagreement(["a", "a", "b", "b"]) == pytest.approx(2 / 3)


def _long():
    return pd.DataFrame([dict(run_id=f"r{r}", item_id=i, pred=pr, correct=c)
                         for r in range(3) for i, pr, c in [("x", "a", True), ("y", "b" if r else "c", r > 0),
                                                            ("z", "d", False), ("u", "e" if r < 2 else "f", None)]])


def test_stability_report_three_way_classes_and_unscored_items():
    out = stability_report(_long())
    assert (out["always_correct"], out["sometimes_correct"], out["never_correct"]) == (1, 1, 1)
    assert out["n_items"] == 4 and out["n_scored_items"] == 3       # u has no ground truth but still counts for flips
    assert out["pred_flip_rate"] == pytest.approx((0 + 2 / 3 + 0 + 2 / 3) / 4)
    assert out["pass@1"] == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert stability_classes(_long()).to_dict() == {"x": "always", "y": "sometimes", "z": "never"}


def test_stability_report_drops_incomplete_items():
    long = pd.concat([_long(), pd.DataFrame([dict(run_id="r0", item_id="only_once", pred="a", correct=True)])])
    assert stability_report(long)["n_items_incomplete"] == 1


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], n_boot=500)
    assert lo <= 0.75 <= hi
    lo2, hi2 = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], n_boot=500, groups=list("aaaabbbb"))
    assert lo2 <= 0.75 <= hi2


def test_majority_vote_ties_go_to_the_earliest_run():
    runs = [[sm_record("r1", "clark", "Age", ["age_at_diagnosis"]), sm_record("r1", "clark", "Race", ["race"])],
            [sm_record("r2", "clark", "Age", ["AGE_AT_INDEX"]), sm_record("r2", "clark", "Race", [])],
            [sm_record("r3", "clark", "Age", ["age_at_index"]), sm_record("r3", "clark", "Race", ["ethnicity"])]]
    out = {r["source_column"]: r for r in majority_vote(runs, "vote3")}
    assert out["Age"]["targets"] == ["AGE_AT_INDEX"] and out["Age"]["votes"] == {"age_at_diagnosis": 1, "age_at_index": 2}
    assert out["Race"]["targets"] == ["race"] and out["Race"]["run_id"] == "vote3"


def test_stability_cli_groups_items_by_the_run_headers_condition(tmp_path, capsys):
    from hbench.cli import main
    from hbench.design import condition_id, make_condition
    from hbench.runlog import RunLog

    paths = []
    for i, temp in enumerate([0, 0, 0.8, 0.8]):
        cond = make_condition(method={"name": "m"}, model={"name": "q"}, decoding={"temperature": temp},
                              engine={"protocol": "natural"}, data={"study": "clark"}, replicate=i)
        log = RunLog.create(tmp_path, "e", "t", "clark", run_id=f"r{i}")
        log.emit("run_header", condition=cond, condition_id=condition_id(cond))
        log.close()
        items = pd.DataFrame({"kind": ["sm", "sm"], "study": ["clark", "clark"], "run_id": [f"r{i}"] * 2,
                              "item_id": ["a", "b"], "pred": ["x", "y" if i % 2 else "z"], "correct": ["True", ""]})
        items.to_csv(log.run_dir / "items.gdc-sm.csv", index=False)  # as the scripts write it: no condition_id
        paths.append(str(log.run_dir / "items.gdc-sm.csv"))
    assert main(["stability", *paths]) == 0
    keys = list(json.loads(capsys.readouterr().out))
    assert len(keys) == 2 and all(k.startswith("sm:clark:") for k in keys)  # T=0 and T=0.8 are not pooled
