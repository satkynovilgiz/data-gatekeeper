"""Sprint 11 fairness measurement on tiny hand-made examples."""
import numpy as np
import pandas as pd
import pytest

from src.impact.fairness import MIN_CLASS_ROWS, fairness_report, gaps, group_metrics, wilson


def test_group_metrics_by_hand():
    # group a: y 1 1 0 0, p .9 .4 .6 .1 -> predicted 1 0 1 0: TPR 1/2, FPR 1/2,
    #          AUC: .9 beats both negatives, .4 beats .1 -> 3/4
    # group b: y 1 0 0 0, p .8 .2 .7 .3 -> predicted 1 0 1 0: TPR 1, FPR 1/3, AUC 1
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    p = np.array([.9, .4, .6, .1, .8, .2, .7, .3])
    a, b = group_metrics(y, p, pd.Series(list("aaaabbbb")))
    assert (a["tpr"], a["fpr"], a["roc_auc"], a["positive_rate"]) == (0.5, 0.5, 0.75, 0.5)
    assert (b["tpr"], b["fpr"], b["roc_auc"], b["positive_rate"]) == (1.0, 0.3333, 1.0, 0.25)
    assert a["predicted_positive_rate"] == b["predicted_positive_rate"] == 0.5
    assert a["small"] and b["small"]                             # < 50 positives


def test_wilson_interval_by_hand():
    assert wilson(5, 10) == [0.2366, 0.7634]
    assert wilson(0, 0) is None
    assert wilson(0, 20)[0] == 0.0


def test_gaps_leave_small_groups_out():
    big = 60
    y = np.r_[np.ones(big), np.zeros(big), np.ones(big), np.zeros(big), [1, 0]]
    p = np.r_[np.full(big, .9), np.full(big, .1),                # group x: TPR 1, FPR 0
              np.full(big, .4), np.full(big, .6),                # group y: TPR 0, FPR 1
              [.9, .9]]                                          # group z: tiny
    g = pd.Series(["x"] * 2 * big + ["y"] * 2 * big + ["z"] * 2)
    table = group_metrics(y, p, g)
    assert [t["small"] for t in table] == [False, False, True]
    gp = gaps(table)
    assert gp["tpr"] == 1.0 and gp["fpr"] == 1.0 and gp["groups_used"] == ["x", "y"]
    assert MIN_CLASS_ROWS == 50


def test_fairness_report_gap_change():
    y = np.array([1, 0] * 100)
    groups = pd.DataFrame({"sex": ["F"] * 100 + ["M"] * 100})
    good = np.where(y == 1, .9, .1)
    worse_for_f = good.copy()
    worse_for_f[:100] = np.where(y[:100] == 1, .3, .1)          # F positives missed by B
    rep = fairness_report(y, {"A": {"m": good}, "B": {"m": worse_for_f}}, groups, ["sex"])
    block = rep["columns"]["sex"]
    assert block["by_variant"]["A"]["m"]["gaps"]["tpr"] == 0.0
    assert block["by_variant"]["B"]["m"]["gaps"]["tpr"] == 1.0
    assert block["gap_change"]["B-A"]["m"]["tpr"] == 1.0          # cleaning widened the gap
    with pytest.raises(ValueError, match="not found"):
        fairness_report(y, {"A": {"m": good}}, groups, ["race"])
