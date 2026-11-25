"""Drift (Sprint 7): PSI against hand-computed values, KS, categories, missing, PII."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.validation.contract import parse_contract
from src.validation.drift import (
    EPS,
    bin_counts,
    categorical_drift,
    column_kind,
    drift_report,
    numeric_drift,
    psi,
    quantile_edges,
    status_for,
)


# ------------------------------------------------------------ PSI by hand
def test_psi_two_bins_by_hand():
    # ref 50% / 50%, cur 25% / 75%:
    # (0.25 - 0.5) ln(0.25 / 0.5) + (0.75 - 0.5) ln(0.75 / 0.5)
    # = 0.25 ln 2 + 0.25 ln 1.5 = 0.17329 + 0.10137 = 0.27465
    assert psi([50, 50], [25, 75]) == pytest.approx(0.27465, abs=1e-5)


def test_psi_identical_is_zero_and_symmetric_in_sign():
    assert psi([10, 20, 30], [1, 2, 3]) == pytest.approx(0.0)     # same shares
    assert psi([50, 50], [25, 75]) > 0 and psi([25, 75], [50, 50]) > 0


def test_psi_empty_bin_uses_eps():
    # ref 100% / 0% (-> EPS), cur 90% / 10%:
    # (0.9 - 1) ln(0.9) + (0.1 - 0.0001) ln(0.1 / 0.0001) = 0.010536 + 0.690093 = 0.70063
    assert EPS == 1e-4
    assert psi([100, 0], [90, 10]) == pytest.approx(0.70063, abs=1e-4)


def test_psi_needs_rows():
    with pytest.raises(ValueError):
        psi([0, 0], [1, 1])


def test_numeric_psi_small_example_by_hand():
    # reference 1..10 with 2 bins -> one edge at the median 5.5:  [1..5] | [6..10] = 50% / 50%
    # current has 3 values below 5.5 and 7 above:                30% / 70%
    # PSI = (0.3 - 0.5) ln(0.3 / 0.5) + (0.7 - 0.5) ln(0.7 / 0.5) = 0.10217 + 0.06729 = 0.16946
    ref = pd.Series([str(i) for i in range(1, 11)])
    cur = pd.Series(["1", "2", "3", "6", "7", "8", "9", "10", "10", "10"])
    out = numeric_drift(ref, cur, bins=2)
    assert out["psi"] == pytest.approx(0.1695, abs=1e-4)
    assert [b["ref_pct"] for b in out["bins"]] == [50.0, 50.0]
    assert [b["cur_pct"] for b in out["bins"]] == [30.0, 70.0]


def test_quantile_edges_and_bins():
    ref = np.arange(1, 101)
    edges = quantile_edges(ref, 10)
    assert len(edges) == 9                                         # 10 bins
    assert bin_counts(ref, edges).tolist() == [10] * 10
    assert len(quantile_edges(np.array([999] * 90 + list(range(10))), 10)) < 9  # ties collapse


@pytest.mark.parametrize("value, status", [
    (0.0, "stable"), (0.0999, "stable"), (0.1, "moderate"), (0.25, "moderate"),
    (0.2501, "major"), (8.0, "major"),
])
def test_status_thresholds(value, status):
    assert status_for(value, 0.1, 0.25) == status


# ------------------------------------------------------------------- KS
def test_ks_identical_and_shifted():
    rng = np.random.default_rng(42)
    base = rng.normal(50, 10, 2000)
    same = numeric_drift(pd.Series(base.astype(str)), pd.Series(base.astype(str)))
    assert same["ks_statistic"] == 0.0 and same["psi"] == pytest.approx(0.0, abs=1e-12)
    shifted = numeric_drift(pd.Series(base.astype(str)),
                            pd.Series((rng.normal(50, 10, 2000) + 10).astype(str)))
    assert shifted["ks_statistic"] > 0.3                           # a 1-sd shift: D ~ 0.38
    assert shifted["ks_pvalue"] < 1e-6
    assert shifted["psi"] > 0.25


def test_missing_values_get_their_own_bin():
    ref = pd.Series([str(i) for i in range(100)])
    cur = pd.Series([str(i) for i in range(95)] + [None] * 5)
    out = numeric_drift(ref, cur)
    assert out["bins"][-1] == {"bin": "<missing>", "ref_pct": 0.0, "cur_pct": 5.0}
    # the missing bin alone: 0.05 x ln(0.05 / 0.0001) = 0.3107
    assert out["psi"] > 0.3
    assert out["ks_statistic"] < 0.06                              # present values barely moved


# ------------------------------------------------------------ categories
def test_unseen_and_vanished_categories():
    ref = pd.Series(["a"] * 50 + ["b"] * 40 + ["c"] * 10)
    cur = pd.Series(["a"] * 50 + ["b"] * 30 + ["NEW"] * 20)
    out = categorical_drift(ref, cur)
    assert out["new_categories"] == ["NEW"] and out["new_categories_count"] == 1
    assert out["new_categories_rows_pct"] == 20.0
    assert out["missing_categories"] == ["c"]
    expected = psi([50, 40, 10, 0], [50, 30, 0, 20])
    assert out["psi"] == pytest.approx(expected, abs=1e-4)


def test_column_kind():
    assert column_kind(pd.Series(["1", "2.5", "3"])) == "numeric"
    assert column_kind(pd.Series(["x", "y"] * 10)) == "categorical"
    assert column_kind(pd.Series([f"free text {i}" for i in range(200)])) == "skipped"
    assert column_kind(pd.Series(["1", "2"]), dtype="string") == "categorical"


# ----------------------------------------------------------------- report
def _frames():
    ref = pd.DataFrame({"x": [str(i) for i in range(200)],
                        "g": ["a", "b"] * 100,
                        "only_ref": ["1"] * 200})
    cur = pd.DataFrame({"x": [str(i + 150) for i in range(200)],
                        "g": ["a", "b"] * 100,
                        "only_cur": ["1"] * 200})
    return ref, cur


def test_report_sorted_by_psi_with_status():
    ref, cur = _frames()
    report = drift_report(ref, cur)
    psis = [c["psi"] for c in report["columns"] if c["psi"] is not None]
    assert psis == sorted(psis, reverse=True)
    by = {c["column"]: c for c in report["columns"]}
    assert by["x"]["status"] == "major" and by["g"]["status"] == "stable"
    assert by["only_ref"]["status"] == by["only_cur"]["status"] == "skipped"
    assert report["major_columns"] == ["x"] and report["status"] == "major"


def test_thresholds_come_from_the_contract():
    ref = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50})
    cur = pd.DataFrame({"g": ["a"] * 25 + ["b"] * 75})             # PSI 0.2747
    assert drift_report(ref, cur)["columns"][0]["status"] == "major"
    contract = parse_contract({"columns": [{"name": "g"}],
                               "drift": {"psi_moderate": 0.2, "psi_major": 0.5}})
    report = drift_report(ref, cur, contract)
    assert report["columns"][0]["status"] == "moderate"
    assert report["thresholds"]["psi_major"] == 0.5


# -------------------------------------------------------------------- PII
def test_pii_values_never_appear_in_the_drift_report():
    emails = ["aida@example.com", "bek@example.com", "dana@example.com", "erlan@example.com"]
    ref = pd.DataFrame({"email": emails * 25,
                        "zip_code": ["72001", "72002"] * 50,
                        "plan": ["basic", "pro"] * 50})
    cur = pd.DataFrame({"email": emails[:3] * 30 + ["secret.person@example.com"] * 10,
                        "zip_code": ["72001", "72009"] * 50,
                        "plan": ["basic", "pro"] * 50})
    report = drift_report(ref, cur)
    text = json.dumps(report)
    for raw in [*emails, "secret.person@example.com", "72001", "72002", "72009"]:
        assert raw not in text, raw
    by = {c["column"]: c for c in report["columns"]}
    assert by["email"]["pii"] and by["email"]["new_categories"] == ["s***@e***.com"]
    assert "categories" not in by["email"] and "bins" not in by["zip_code"]
    assert math.isfinite(by["email"]["psi"]) and by["email"]["psi"] > 0
    assert "categories" in by["plan"]                              # non-PII keeps details
    assert set(report["pii_columns"]) == {"email", "zip_code"}


def test_contract_pii_flag_hides_a_column_detection_misses():
    ref = pd.DataFrame({"code": ["K1", "K2"] * 50})
    cur = pd.DataFrame({"code": ["K1", "K3"] * 50})
    contract = parse_contract({"columns": [{"name": "code", "pii": True}]})
    report = drift_report(ref, cur, contract)
    assert "K3" not in json.dumps(report) and "K2" not in json.dumps(report)
