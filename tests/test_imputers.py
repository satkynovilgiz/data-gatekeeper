"""ML imputers (src/cleaning/imputers.py)."""
import numpy as np
import pandas as pd
import pytest

from src.cleaning.imputers import ML_METHODS, decode, encode, impute_columns


def related_data(n: int = 200) -> pd.DataFrame:
    """y = 2x (+ small noise); zone depends on x. Rows 0-9 have y and zone missing."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, n)
    df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 1, n),
                       "zone": np.where(x < 50, "low", "high")})
    df.loc[:9, ["y", "zone"]] = np.nan
    return df


@pytest.mark.parametrize("method", ML_METHODS)
def test_ml_methods_use_other_columns(method):
    df = related_data()
    out = impute_columns(df, ["y", "zone"], ["x"], method)
    truth = 2 * df.loc[:9, "x"]
    assert out["y"].notna().all() and out["zone"].notna().all()
    # ML follows y = 2x closely; the median would be off by ~50 on average
    assert (out.loc[:9, "y"] - truth).abs().mean() < 15
    expected_zone = np.where(df.loc[:9, "x"] < 50, "low", "high")
    # 50% would be guessing; iterative (linear models) is weakest on this threshold rule
    assert (out.loc[:9, "zone"] == expected_zone).mean() >= 0.7


def test_median_baseline_ignores_other_columns():
    df = related_data()
    out = impute_columns(df, ["y", "zone"], ["x"], "median")
    assert out.loc[:9, "y"].nunique() == 1                     # same value for every gap
    assert out.loc[0, "y"] == pytest.approx(df["y"].median())
    assert out.loc[0, "zone"] == df["zone"].mode().sort_values().iloc[0]


@pytest.mark.parametrize("method", ["median", *ML_METHODS])
def test_only_target_gaps_change(method):
    df = related_data()
    df.loc[50, "x"] = np.nan                                    # a gap in a feature
    out = impute_columns(df, ["y"], ["x", "zone"], method)
    assert np.isnan(out.loc[50, "x"])                           # features are never filled
    known = df["y"].notna()
    pd.testing.assert_series_equal(out.loc[known, "y"], df.loc[known, "y"])


@pytest.mark.parametrize("method", ML_METHODS)
def test_whole_numbers_stay_whole_and_inside_known_range(method):
    rng = np.random.default_rng(1)
    age = rng.integers(20, 60, 100).astype(float)
    df = pd.DataFrame({"age": age, "years": age - 20 + rng.integers(0, 3, 100)})
    df.loc[:4, "age"] = np.nan
    out = impute_columns(df, ["age"], ["years"], method)
    filled = out.loc[:4, "age"]
    assert (filled % 1 == 0).all()
    assert filled.between(df["age"].min(), df["age"].max()).all()


def test_encode_decode_round_trip():
    df = pd.DataFrame({"n": [1.0, 2.0, np.nan], "c": ["a", None, "b"]})
    enc = encode(df, ["n", "c"])
    assert enc.X.shape == (3, 3)                                # 1 numeric + 2 one-hot
    assert np.isnan(enc.X[1, 1:]).all()                         # missing category = all NaN
    assert decode(enc, np.nan_to_num(enc.X), "c")[2] == "b"
    assert decode(enc, enc.X, "n")[1] == pytest.approx(2.0)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown imputation method"):
        impute_columns(related_data(), ["y"], ["x"], "magic")


def test_nothing_to_fill_returns_copy():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    out = impute_columns(df, ["a", "b"], [], "knn")
    pd.testing.assert_frame_equal(out, df)
