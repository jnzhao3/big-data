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


def _sampled(changes_a=False):
    """4 runs at default temperature. a: P=1 (always greedy); b: P=0.5 (changes); c: P=0.9; d: never greedy."""
    rows = []
    for r in range(4):
        rows += [dict(run_id=f"r{r}", item_id="a", pred="x" if not (changes_a and r) else "x2", p_path=1.0,
                      greedy_path=True, prompt_sha256="pa"),
                 dict(run_id=f"r{r}", item_id="b", pred="y" if r % 2 else "z", p_path=0.5 if r == 0 else 0.2,
                      greedy_path=r == 0, prompt_sha256="pb"),
                 dict(run_id=f"r{r}", item_id="c", pred="w" if not (changes_a and r) else "w2", p_path=0.9,
                      greedy_path=True, prompt_sha256="pc"),
                 dict(run_id=f"r{r}", item_id="d", pred="v" if r else "u", p_path=0.3, greedy_path=False,
                      prompt_sha256="pd")]
    return pd.DataFrame(rows).assign(correct=None, margin=None)


def test_sampling_check_is_one_sided():
    from hbench.stability import sampling_check
    got = sampling_check(_sampled())
    assert (got["n_checked"], got["n_unchecked"], got["observed_changed"]) == (3, 1, 1)
    assert got["bound_expected"] == pytest.approx((1 - 0.5 ** 4) + (1 - 0.9 ** 4)) and got["alarm"] is False
    assert got["interval_pairwise"][1] == pytest.approx((1 - 0.25) + (1 - 0.81))
    hot = sampling_check(_sampled(changes_a=True))   # an item with P=1 changed: more change than sampling allows
    assert hot["observed_changed"] == 3 and hot["alarm"] is True
    ref = pd.DataFrame([{"prompt_sha256": "pd", "p_path_api_chat": 0.6, "p_path_v1": 0.4}])
    with_ref = sampling_check(_sampled(), reference=ref, endpoint="/api/chat")
    assert (with_ref["n_checked"], with_ref["n_unchecked"], with_ref["observed_changed"]) == (4, 0, 2)
    assert "sampling_check" in stability_report(_sampled())
