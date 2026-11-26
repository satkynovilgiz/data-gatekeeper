"""Anomaly detection (Sprint 9): hand-computed baselines, forest + explanations, injectors,
metrics, and the privacy rules (no label, ID or PII used or printed)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cleaning.outliers import robust_scale
from src.profiling.profiler import read_csv_raw
from src.validation.anomaly import AnomalyModel, detect, evaluate, prepare
from src.validation.anomaly_inject import inject, usable_columns

SAMPLE = Path(__file__).resolve().parents[1] / "data/raw/bank_marketing_sample.csv"


# ------------------------------------------------------- baselines by hand
def test_robust_z_by_hand():
    # median 3, |x - 3| = 2, 1, 0, 1, 97 -> MAD 1 -> scale 1.4826; z(100) = 97 / 1.4826 = 65.43
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    assert robust_scale(s) == pytest.approx(1.4826)
    m = AnomalyModel().fit(pd.DataFrame({"x": s}))
    z = m.robust_z(pd.DataFrame({"x": [100.0, 3.0]}))["x"]
    assert z.tolist() == pytest.approx([65.4256, 0.0], abs=1e-3)


def test_mad_zero_column_is_not_used():
    X = pd.DataFrame({"flat": [0.0] * 8 + [5.0, 9.0],              # >= half identical: MAD 0
                      "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    assert robust_scale(X["flat"]) is None
    m = AnomalyModel().fit(X)
    z = m.robust_z(X)
    assert z["flat"].isna().all()
    assert m.score(X, "zscore").max() == pytest.approx(z["x"].max())   # only x counts


def test_iqr_distance_by_hand():
    # 1..8: Q1 2.75, Q3 6.25, IQR 3.5 -> fences -2.5 / 11.5; 18.5 is 7 above = 2.0 IQR
    m = AnomalyModel().fit(pd.DataFrame({"x": [float(v) for v in range(1, 9)]}))
    d = m.iqr_distance(pd.DataFrame({"x": [18.5, 5.0, -6.0]}))["x"]
    assert d.tolist() == pytest.approx([2.0, 0.0, 1.0])
    # row score = columns outside + d / (1 + d)
    assert m.score(pd.DataFrame({"x": [18.5]}), "iqr").iloc[0] == pytest.approx(1 + 2 / 3)


# --------------------------------------------------- forest + explanations
@pytest.fixture
def table():
    rng = np.random.default_rng(5)
    n = 400
    a = rng.normal(50, 5, n)
    return pd.DataFrame({"a": a, "b": a * 2 + rng.normal(0, 1, n),
                         "c": rng.normal(0, 1, n), "d": rng.normal(10, 2, n)})


def test_forest_is_deterministic(table):
    s1 = AnomalyModel().fit(table).score(table, "iforest")
    s2 = AnomalyModel().fit(table).score(table, "iforest")
    assert s1.equals(s2)


def test_leave_one_column_out_finds_the_planted_column(table):
    planted = table.copy()
    planted.loc[7, "c"] = 9.0                                       # 9 sd away, only column c
    m = AnomalyModel().fit(planted)
    scores = m.score(planted, "iforest")
    # a single-column extreme is in the forest's top 5%, not always first: rows extreme in
    # several columns isolate even faster (the same thing the Bank evaluation shows)
    assert scores.rank(ascending=False)[7] <= 0.05 * len(planted)
    reasons = m.explain(planted, 7, "iforest")
    assert reasons[0]["column"] == "c" and reasons[0]["value"] == 9.0
    assert reasons[0]["contribution"] > 0 and len(reasons) == 3
    assert m.explain(planted, 7, "zscore")[0]["column"] == "c"


def test_forest_sees_a_broken_combination_that_z_cannot(table):
    # plant into the most ordinary row, so nothing else about it is unusual
    row = AnomalyModel().fit(table).score(table, "zscore").idxmin()
    odd = table.copy()
    odd.loc[row, ["a", "b"]] = [58.0, 84.0]        # both normal alone, b should be ~116
    m = AnomalyModel().fit(odd)
    z_rank = m.score(odd, "zscore").rank(ascending=False)[row]
    f_rank = m.score(odd, "iforest").rank(ascending=False)[row]
    assert f_rank < z_rank


# ------------------------------------------------------------- injectors
@pytest.fixture(scope="module")
def bank():
    df = read_csv_raw(SAMPLE)
    clean = prepare(df)
    return df, clean, usable_columns(clean, skip=["y", "pdays"])


def test_injectors_change_exactly_one_percent(bank):
    df, clean, cols = bank
    out, truth = inject(df, clean, cols, share=0.01, seed=42)
    changed = np.flatnonzero((out.fillna("~") != df.fillna("~")).any(axis=1).to_numpy())
    assert len(truth) == round(0.01 * len(df)) == len(changed)
    assert set(changed) == set(truth["row"])
    assert truth["kind"].value_counts().to_dict() == {"shift": 4, "extreme": 3, "combination": 3}
    assert len(out) == len(df) and out["y"].equals(df["y"])          # label untouched
    again, truth2 = inject(df, clean, cols, share=0.01, seed=42)
    assert again.equals(out) and truth2.equals(truth)


def test_injected_values_follow_their_kind(bank):
    df, clean, cols = bank
    out, truth = inject(df, clean, cols, share=0.03, seed=1)
    num = clean[cols].astype(float)
    for _, t in truth[truth["kind"] == "extreme"].iterrows():
        assert float(out.at[t["row"], t["columns"]]) > num[t["columns"]].max()
    for _, t in truth[truth["kind"] == "combination"].iterrows():
        b = t["columns"].split("+")[1]
        v = float(out.at[t["row"], b])
        assert num[b].quantile(0.05) - 1e-6 <= v <= num[b].quantile(0.95) + 1e-6  # normal alone


# --------------------------------------------------------------- metrics
def test_metrics_on_a_tiny_labelled_example():
    # top 2 = rows 0 (anomaly) and 1 (normal): precision@2 = recall@2 = 0.5
    # ROC-AUC: (0.9 beats 3 normals) + (0.7 beats 2 of 3) = 5 / 6
    # PR-AUC (average precision): hits at ranks 1 and 3 -> (1 + 2/3) / 2 = 0.8333
    out = evaluate(pd.Series([0.9, 0.8, 0.1, 0.7, 0.2]), pd.Series([1, 0, 0, 1, 0]))
    assert out["k"] == 2 and out["precision_at_k"] == 0.5 and out["recall_at_k"] == 0.5
    assert out["roc_auc"] == pytest.approx(5 / 6)
    assert out["pr_auc"] == pytest.approx(0.8333, abs=1e-4)
    with pytest.raises(ValueError):
        evaluate(pd.Series([0.1, 0.2]), pd.Series([0, 0]))


# --------------------------------------------------------------- privacy
def test_label_id_and_pii_are_never_used_or_printed():
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "customer_id": [f"C{9000 + i}" for i in range(n)],
        "email": [f"person{i}@example.com" for i in range(n)],
        "income": [str(round(v)) for v in rng.normal(3000, 400, n)],
        "score_id": [str(i) for i in range(n)],
        "spend": [f"{v:.2f}" for v in rng.normal(100, 10, n)],
        "y": rng.choice(["no", "yes"], n)})
    df.loc[11, "income"] = "99000"
    section = detect(df, exclude=["y"], top=5)
    assert section["features"] == ["income", "spend"]
    assert {"customer_id", "email", "score_id", "y"} <= set(section["excluded"])
    text = json.dumps(section)
    assert "@example.com" not in text and "C9011" not in text
    top = section["methods"]["zscore"]["top"][0]
    assert top["row"] == 11 and top["reasons"][0]["column"] == "income"


def test_reference_fit_and_bad_settings(table):
    raw = table.astype(str)
    shifted = (table + 40).astype(str)
    section = detect(shifted, reference=raw, methods=("iforest",))
    assert section["fitted_on"] == "reference"
    assert section["methods"]["iforest"]["flagged_share"] > 0.5        # all moved
    with pytest.raises(ValueError, match="unknown method"):
        detect(raw, methods=("lof",))
    with pytest.raises(ValueError, match="contamination"):
        detect(raw, contamination=0.9)


# ------------------------------------------------- Sprint 10: categories + mixture z
from src.validation.anomaly import fit_mixture  # noqa: E402


def test_frequency_encoding_and_one_hot():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 25,
                      "city": ["a"] * 60 + ["b"] * 30 + ["c"] * 9 + [None]})
    m = AnomalyModel().fit(X, categorical=["city"])
    enc = m.encode_categories(pd.DataFrame({"city": ["a", "c", "zzz", None]}))
    assert enc["city"].tolist() == pytest.approx([0.60, 0.09, 0.0, 0.01])   # unseen -> 0
    oh = AnomalyModel(encoding="onehot").fit(X, categorical=["city"])
    enc = oh.encode_categories(pd.DataFrame({"city": ["b", "zzz"]}))
    assert list(enc.columns) == ["city=<missing>", "city=a", "city=b", "city=c"]
    assert enc.iloc[0].tolist() == [0, 0, 1, 0] and enc.iloc[1].sum() == 0
    with pytest.raises(ValueError, match="encoding"):
        AnomalyModel(encoding="target")


def test_categorical_leave_one_column_out_names_the_rare_category():
    rng = np.random.default_rng(4)
    n = 600
    X = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(5, 1, n),
                      "plan": rng.choice(["basic", "pro"], n)})
    X.loc[17, "plan"] = "platinum"                                    # seen once only
    X.loc[17, ["x", "y"]] = [0.0, 5.0]                                # numbers perfectly normal
    m = AnomalyModel().fit(X, categorical=["plan"])
    reasons = m.explain(X, 17, "iforest_with_categories")
    assert reasons[0]["column"] == "plan" and reasons[0]["value"] == "platinum"
    assert reasons[0]["median"] in ("basic", "pro") and reasons[0]["robust_z"] is None
    rank = m.score(X, "iforest_with_categories").rank(ascending=False)[17]
    assert rank <= 0.10 * n
    assert m.score(X, "iforest").rank(ascending=False)[17] > rank      # numbers-only misses it


def _two_bumps() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(np.r_[rng.normal(1.0, 0.1, 700), rng.normal(5.0, 0.1, 1300)])


def test_mixture_z_two_bumps_by_hand():
    x = _two_bumps()
    mix = fit_mixture(x)
    assert mix is not None and mix["k"] in (2, 3)
    m = AnomalyModel().fit(pd.DataFrame({"x": x}))
    probe = pd.DataFrame({"x": [1.0, 5.0, 3.0]})                      # bump, bump, between
    mz = m.mixture_z(probe)["x"]
    assert mz.iloc[0] < 1 and mz.iloc[1] < 1                          # no flag in either bump
    assert mz.iloc[2] > 3.5                                           # the gap is flagged
    assert m.flagged(m.score(probe, "zscore_mixture"), "zscore_mixture").tolist() == [
        False, False, True]
    # the old robust z (median ~5) calls the whole first bump extreme
    assert m.robust_z(probe)["x"].iloc[0] > 3.5
    assert m.score(pd.DataFrame({"x": x}), "zscore").gt(3.5).mean() > 0.3


def test_bic_picks_one_bump_on_single_bump_data_and_old_z_is_kept():
    x = pd.Series(np.random.default_rng(1).normal(10, 2, 3000))
    assert fit_mixture(x) is None
    m = AnomalyModel().fit(pd.DataFrame({"x": x}))
    probe = pd.DataFrame({"x": [10.0, 20.0]})
    assert m.mixture_z(probe).equals(m.robust_z(probe))


def test_mixture_handles_tiny_or_flat_columns():
    assert fit_mixture(pd.Series([1.0] * 50)) is None
    assert fit_mixture(pd.Series([1.0, 2.0, 3.0])) is None             # too few values


def test_detect_reports_categories_and_mixtures():
    df = read_csv_raw(SAMPLE)
    section = detect(df, exclude=["y"], top=2)
    assert "job" in section["categorical_features"] and "y" not in section["categorical_features"]
    assert set(section["methods"]) == {"iforest", "zscore", "iqr", "zscore_mixture",
                                       "iforest_with_categories"}
    assert section["settings"]["encoding"] == "frequency"
    assert section["settings"]["mixture_components"]                  # some columns bimodal
    top = section["methods"]["zscore_mixture"]["top"][0]["reasons"][0]
    assert "mixture_z" in top
