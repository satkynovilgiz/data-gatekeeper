"""Sprint 6 — transformation module (src/transform/)."""
import numpy as np
import pandas as pd
import pytest

from src.transform.pipeline import load_pipeline, save_transform, split, transform_dataset
from src.transform.plan import bin_labels, plan_columns


def customers(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "age": rng.integers(18, 80, n),
        "income": rng.lognormal(10, 1, n).round(),               # long right tail
        "score": rng.normal(50, 10, n),
        "rating": rng.integers(1, 6, n),                          # small 1-5 scale
        "education": rng.choice(["basic.4y", "high.school", "university.degree", "unknown"], n),
        "default": rng.choice(["no", "yes", "unknown"], n),
        "is_anomaly": rng.integers(0, 2, n),
        "y": rng.choice(["no", "yes"], n, p=[0.8, 0.2]),
    })


def by_column(plan: list[dict]) -> dict[str, dict]:
    return {p["column"]: p for p in plan if p["kind"] != "bins"}


# ---- planning: each column gets a method and a reason --------------------------------
def test_plan_picks_a_method_per_column_with_a_reason():
    plan, dropped = plan_columns(customers(), target="y")
    p = by_column(plan)
    assert p["income"]["power"] in ("log", "yeo-johnson") and p["income"]["skew_before"] > 1
    assert p["score"]["power"] is None and p["score"]["scaler"] == "standard"
    assert p["rating"]["scaler"] == "minmax"
    assert p["education"]["kind"] == "ordinal" and "outside the order" in p["education"]["reason"]
    assert p["default"]["kind"] == "onehot" and "'unknown' kept" in p["default"]["reason"]
    assert p["is_anomaly"]["kind"] == "binary"
    assert all(item["reason"] for item in plan)
    assert {d["column"] for d in dropped} == {"customer_id", "y"}


def test_outlier_heavy_column_gets_robust_scaler():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0, 1, 180), rng.normal(0, 1, 20) * 50])   # 10% extremes
    plan, _ = plan_columns(pd.DataFrame({"x": values}))
    assert plan[0]["scaler"] == "robust" and "outliers" in plan[0]["reason"]


def test_left_skew_is_not_made_worse_by_log():
    # nr.employed in Bank is skewed to the LEFT; log1p makes that worse, so it must not win
    rng = np.random.default_rng(2)
    x = 5300 - rng.lognormal(3, 1, 500)
    plan, _ = plan_columns(pd.DataFrame({"x": x}))
    assert plan[0]["power"] == "yeo-johnson"


def test_pii_and_dates_are_dropped():
    df = pd.DataFrame({"email": [f"u{i}@example.com" for i in range(30)],
                       "hire_date": pd.date_range("2020-01-01", periods=30).astype(str),
                       "amount": range(30)})
    _, dropped = plan_columns(df)
    reasons = {d["column"]: d["reason"] for d in dropped}
    assert reasons["email"].startswith("personal data (email)")
    assert reasons["hire_date"].startswith("date")


def test_scaling_override_and_bad_scaler():
    plan, _ = plan_columns(customers(), target="y", scaling={"score": "robust"})
    assert by_column(plan)["score"]["scaler"] == "robust"
    with pytest.raises(ValueError, match="unknown scaler"):
        plan_columns(customers(), scaling={"score": "zscore"})


def test_bin_labels():
    assert bin_labels([18, 30, 45, 100]) == ["18-29", "30-44", "45-99"]


# ---- fitting: train only, then the same transformation for test ---------------------
def test_scaler_never_sees_test_rows():
    # The proof of "no data leakage": the fitted scaler counted exactly the train rows,
    # and its mean is the train mean, not the mean of all rows.
    df = customers(300)
    result = transform_dataset(df, target="y", test_size=0.2, seed=42,
                               config={"scaling": {"score": "standard"}})
    train_raw, test_raw = split(df, "y", 0.2, 42)
    scaler = result.pipeline.named_transformers_["score_numeric"].named_steps["scale"]
    assert scaler.n_samples_seen_ == len(train_raw) == 240
    assert scaler.mean_[0] == pytest.approx(train_raw["score"].mean())
    assert scaler.mean_[0] != pytest.approx(df["score"].mean())
    assert set(train_raw.index).isdisjoint(test_raw.index)


def test_minmax_learned_from_train_so_test_can_leave_0_1():
    df = pd.DataFrame({"rating": [1, 2, 3, 4, 5] * 20})
    result = transform_dataset(df, test_size=0.2)
    new = result.pipeline.transform(pd.DataFrame({"rating": [10]}))   # never seen in train
    assert new["rating"].iloc[0] > 1                                   # not refitted


def test_unknown_stays_a_category_and_unseen_values_do_not_crash():
    result = transform_dataset(customers(), target="y")
    assert "default_unknown" in result.train.columns
    assert "missingindicator_education" in result.train.columns          # 'unknown' education
    row = customers(1).assign(default="maybe", education="phd")       # never seen in train
    out = result.pipeline.transform(row)
    assert out.filter(like="default_").to_numpy().sum() == 0          # all one-hot columns 0
    assert out.notna().all().all()


def test_output_is_numeric_without_gaps_and_keeps_target():
    df = customers()
    df.loc[[3, 7], "score"] = np.nan
    result = transform_dataset(df, target="y")
    features = result.train.drop(columns="y")
    assert features.notna().all().all()
    assert all(np.issubdtype(t, np.number) for t in features.dtypes)
    assert set(result.train["y"]) == {"no", "yes"}
    assert result.train["y"].eq("yes").mean() == pytest.approx(
        result.test["y"].eq("yes").mean(), abs=0.03)                  # stratified split


def test_skew_report_before_and_after():
    report = transform_dataset(customers(), target="y").report
    income = next(s for s in report["skewed_columns"] if s["column"] == "income")
    assert abs(income["skew_after"]) < abs(income["skew_before"])


def test_bins_add_one_hot_groups():
    bins = {"age": [18, 30, 60, 120]}
    result = transform_dataset(customers(), target="y", config={"bins": bins})
    groups = sorted(c for c in result.train.columns if c.startswith("age_group_"))
    assert groups == ["age_group_18-29", "age_group_30-59", "age_group_60-119"]
    assert (result.train[groups].sum(axis=1) == 1).all()              # every row in one group
    assert "age" in result.train.columns                              # the column itself stays


def test_save_and_load_give_the_same_result(tmp_path):
    df = customers()
    result = transform_dataset(df, target="y")
    files = save_transform(result, tmp_path)
    loaded = load_pipeline(files["pipeline"])
    _, test_raw = split(df, "y")
    pd.testing.assert_frame_equal(loaded.transform(test_raw), result.test.drop(columns="y"))
    assert (tmp_path / "transform_report.json").exists()


def test_missing_target_rejected():
    with pytest.raises(ValueError, match="target column"):
        transform_dataset(customers(), target="label")


def test_known_order_is_used_only_when_it_fits_the_values():
    """Sprint 10: the Bank 'education' order must not wipe out UCI Adult's 'education'."""
    from src.transform.plan import plan_columns

    bank = pd.DataFrame({"education": ["basic.4y", "high.school", "university.degree",
                                       "unknown"] * 10})
    adult = pd.DataFrame({"education": ["Bachelors", "HS-grad", "Masters", "11th"] * 10})
    (bank_plan,), _ = plan_columns(bank)
    (adult_plan,), _ = plan_columns(adult)
    assert bank_plan["kind"] == "ordinal"
    assert adult_plan["kind"] == "onehot" and "matches only 0%" in adult_plan["reason"]
