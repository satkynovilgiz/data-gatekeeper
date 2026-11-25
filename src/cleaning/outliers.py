"""Outlier detection — two methods that answer different questions.

IQR (one column at a time): "is this value extreme FOR THIS COLUMN?"
    Q1, Q3 = 25th and 75th percentile; IQR = Q3 - Q1 (the width of the middle half).
    Fences: Q1 - 1.5 x IQR and Q3 + 1.5 x IQR. Outside = outlier.
    Simple and explainable, and it tells us WHICH value is odd, so the actions
    (flag / cap / set to NaN) are applied to IQR outliers.

Isolation Forest (whole rows, scikit-learn): "is this ROW unusual?"
    Builds many random trees; each one splits the data at random cut points. An unusual
    row is isolated after only a few splits (few rows look like it); a normal row needs
    many. A short average path = an anomaly. It looks at all numeric columns together, so
    it also finds odd COMBINATIONS that no single column shows. We mark the most isolated
    ANOMALY_SHARE of rows (1%) in an is_anomaly column; it never changes values.

Robust z (one column, Sprint 9): "how many typical spreads is this value from the middle?"
    z = |x - median| / (1.4826 x MAD), MAD = median of |x - median|. The factor 1.4826 makes
    1.4826 x MAD equal the standard deviation for normal data, but unlike mean / std a few
    extreme values cannot pull the median or the MAD. MAD = 0 (at least half the values are
    identical) is handled like IQR = 0: the column is not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

IQR_FACTOR = 1.5
ANOMALY_SHARE = 0.01
SEED = 42


def iqr_fences(s: pd.Series) -> tuple[float, float] | None:
    """(lower, upper) IQR fences of a numeric column, or None when IQR = 0.

    IQR = 0 means at least half the values are identical (e.g. previous = 0); the fences
    would then call every other value an outlier, so the method is not used.
    """
    known = s.dropna().astype(float)
    if known.empty:
        return None
    q1, q3 = known.quantile(0.25), known.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q1 - IQR_FACTOR * iqr), float(q3 + IQR_FACTOR * iqr)


def iqr_outliers(s: pd.Series) -> tuple[pd.Series, tuple[float, float] | None]:
    """True/False mask of values outside the IQR fences, and the fences themselves."""
    fences = iqr_fences(s)
    if fences is None:
        return pd.Series(False, index=s.index), None
    lo, hi = fences
    return s.notna() & ((s < lo) | (s > hi)), fences


def robust_scale(s: pd.Series) -> float | None:
    """1.4826 x MAD of a numeric column, or None when MAD = 0 or there are no values."""
    known = s.dropna().astype(float)
    if known.empty:
        return None
    mad = float((known - known.median()).abs().median())
    return 1.4826 * mad if mad > 0 else None


def fill_for_forest(X: pd.DataFrame, medians: pd.Series | None = None) -> pd.DataFrame:
    """Numbers for the Isolation Forest: empty cells -> the column median (then 0).

    medians: use these (e.g. learned on a reference) instead of X's own.
    """
    X = X.astype(float)
    return X.fillna(X.median() if medians is None else medians).fillna(0)


def isolation_forest_model(X: pd.DataFrame, share: float = ANOMALY_SHARE,
                           seed: int = SEED) -> IsolationForest:
    """An Isolation Forest fitted on X (already numeric, no gaps). Trees need no scaling."""
    model = IsolationForest(n_estimators=100, contamination=share, random_state=seed)
    return model.fit(X.to_numpy())


def isolation_forest_rows(df: pd.DataFrame, columns: list[str],
                          share: float = ANOMALY_SHARE, seed: int = SEED) -> pd.Series:
    """True for the `share` most unusual rows according to an Isolation Forest.

    Empty cells are filled with the column median first (the model needs numbers);
    trees do not care about units, so no scaling is needed.
    """
    if not columns or share <= 0 or len(df) < 10:
        return pd.Series(False, index=df.index)
    X = fill_for_forest(df[columns])
    model = isolation_forest_model(X, share, seed)
    return pd.Series(model.predict(X.to_numpy()) == -1, index=df.index)


def main_reason(df: pd.DataFrame, columns: list[str], row: int) -> str:
    """Which value makes a row unusual: the column with the largest |z-score|.

    Example: "salary=9999999 (z=11.3)".
    """
    X = df[columns].astype(float)
    z = ((X.loc[row] - X.mean()) / X.std().replace(0, np.nan)).abs()
    if z.isna().all():
        return ""
    col = z.idxmax()
    value = df.at[row, col]
    return f"{col}={float(value):.10g} (z={z[col]:.1f})"
