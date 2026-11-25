"""Anomaly detection for validation (Sprint 9): which ROWS look unlike the others, and why?

Three methods, all built on src/cleaning/outliers.py (no second copy of the maths):

  zscore   robust z per numeric column, |x - median| / (1.4826 x MAD); row score = the largest.
           Finds a single extreme value. Columns with MAD = 0 are not used (like IQR = 0).
  iqr      per column, how far a value is outside the IQR fences, in IQR units; row score =
           number of columns outside + a tie-break by total distance (n + d / (1 + d)).
  iforest  Isolation Forest (100 trees, seed 42), score = -score_samples (higher = odder);
           `contamination` (default 1%) sets the flag threshold. Sees all columns at once, so
           it can find odd COMBINATIONS that no single column shows.

How a top row is explained (3 columns, each with value, column median and robust z):
  zscore / iqr   the columns with the largest robust z / IQR distance.
  iforest        leave-one-column-out: put ONE column of the row back to its median, score the
                 row again with the same forest, and see how much less odd it became. The column
                 whose "repair" lowers the score most contributed most. This explains the forest's
                 own decision; it is not a z-score in disguise.

What the methods look at (decided before any evaluation, never tuned on labels):
  - the data after the gatekeeper's row-wise cleaning (profile + steps 1-7): codes such as
    pdays = 999 ("never contacted") are known codes, not anomalies; impossible or
    out-of-range values are KEPT (steps 8 / 8b skipped), because they are what we look for
  - numeric columns only; never the label, IDs, PII (src/privacy) or columns > 50% empty
  - fitted on a reference file when one is given (then new rows are judged against it),
    otherwise on the file itself
Rows are reported by their position in the file (0 = first data row), never by an ID; the
data is never changed.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.cleaning.outliers import (
    ANOMALY_SHARE,
    SEED,
    fill_for_forest,
    iqr_fences,
    isolation_forest_model,
    robust_scale,
)
from src.profiling.semantic import is_id_name, numeric_parse_rate
from src.validation.contract import pii_columns

METHODS = ("iforest", "zscore", "iqr")
TOP_K = 10
REASONS = 3
Z_FLAG = 3.5            # robust z above this = flagged (Iglewicz & Hoaglin's usual cut-off)
MAX_MISSING = 0.5       # a column more empty than this is not a feature
NUMERIC_SHARE = 0.95


# ------------------------------------------------------------ preparation
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """The gatekeeper's view of the rows: row-wise cleaning steps, numbers as floats."""
    from src.impact.impact import RowCleaner  # avoid an import cycle at load time

    return RowCleaner(value_checks=False).fit(df).transform(df)


def choose_features(raw: pd.DataFrame, clean: pd.DataFrame, exclude: list[str] | None = None,
                    contract: dict | None = None) -> tuple[list[str], dict[str, str]]:
    """Numeric feature columns and {column: why it is left out} for the others."""
    exclude = set(exclude or [])
    pii = pii_columns(raw, contract)
    features, skipped = [], {}
    for c in clean.columns:
        s = clean[c]
        if c in exclude:
            skipped[c] = "label / excluded"
        elif c in pii:
            skipped[c] = f"personal data ({pii[c]})"
        elif is_id_name(c):
            skipped[c] = "identifier"
        elif not pd.api.types.is_numeric_dtype(s) and numeric_parse_rate(s) < NUMERIC_SHARE:
            skipped[c] = "not numeric"
        elif s.isna().mean() > MAX_MISSING:
            skipped[c] = f"{s.isna().mean():.0%} empty"
        else:
            features.append(c)
    return features, skipped


# ---------------------------------------------------------------- fitting
class AnomalyModel:
    """Medians, robust scales, IQR fences and a forest learned on one table."""

    def __init__(self, contamination: float = ANOMALY_SHARE, seed: int = SEED) -> None:
        self.contamination = contamination
        self.seed = seed

    def fit(self, X: pd.DataFrame) -> AnomalyModel:
        """Learn everything from X (numeric features only)."""
        X = X.astype(float)
        self.columns_ = list(X.columns)
        self.medians_ = X.median()
        self.scales_ = {c: robust_scale(X[c]) for c in self.columns_}
        self.fences_ = {c: iqr_fences(X[c]) for c in self.columns_}
        self.forest_ = isolation_forest_model(fill_for_forest(X, self.medians_),
                                              self.contamination, self.seed)
        self.offset_ = float(-self.forest_.offset_)    # iforest score above this = flagged
        return self

    # per-cell views -----------------------------------------------------
    def robust_z(self, X: pd.DataFrame) -> pd.DataFrame:
        """|x - median| / scale per cell; NaN where the value is empty or the column unused."""
        out = {}
        for c in self.columns_:
            scale = self.scales_[c]
            x = X[c].astype(float)
            out[c] = (x - self.medians_[c]).abs() / scale if scale else x * np.nan
        return pd.DataFrame(out, index=X.index)

    def iqr_distance(self, X: pd.DataFrame) -> pd.DataFrame:
        """How far outside the fences, in IQR units (0 inside; NaN if no fences)."""
        out = {}
        for c in self.columns_:
            x = X[c].astype(float)
            if self.fences_[c] is None:
                out[c] = x * np.nan
                continue
            lo, hi = self.fences_[c]
            iqr = (hi - lo) / 4.0                      # fences are Q1-1.5 IQR .. Q3+1.5 IQR
            out[c] = np.maximum(np.maximum(lo - x, x - hi), 0) / iqr
        return pd.DataFrame(out, index=X.index)

    # row scores ---------------------------------------------------------
    def score(self, X: pd.DataFrame, method: str) -> pd.Series:
        """Row scores, higher = more unusual."""
        if method == "zscore":
            return self.robust_z(X).max(axis=1, skipna=True).fillna(0.0)
        if method == "iqr":
            d = self.iqr_distance(X)
            n_out = (d > 0).sum(axis=1)
            total = d.fillna(0).sum(axis=1)
            return n_out + total / (1 + total)
        if method == "iforest":
            filled = fill_for_forest(X[self.columns_], self.medians_)
            return pd.Series(-self.forest_.score_samples(filled.to_numpy()), index=X.index)
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")

    def flagged(self, scores: pd.Series, method: str) -> pd.Series:
        """Which rows count as anomalies for a method (its documented threshold)."""
        if method == "iforest":
            return scores > self.offset_
        if method == "zscore":
            return scores > Z_FLAG
        return scores >= 1                              # iqr: at least one column outside

    # explanations -------------------------------------------------------
    def explain(self, X: pd.DataFrame, row: Any, method: str, k: int = REASONS) -> list[dict]:
        """The k columns that made one row unusual, with value, median and robust z."""
        z = self.robust_z(X.loc[[row]]).iloc[0]
        if method == "iforest":
            base = X.loc[[row], self.columns_].astype(float)
            score0 = float(self.score(base, "iforest").iloc[0])
            contrib = {}
            for c in self.columns_:
                repaired = base.copy()
                repaired[c] = self.medians_[c]
                contrib[c] = score0 - float(self.score(repaired, "iforest").iloc[0])
            weight = pd.Series(contrib)
        elif method == "iqr":
            weight = self.iqr_distance(X.loc[[row]]).iloc[0].fillna(0)
        else:
            weight = z.fillna(0)
        top = weight.sort_values(ascending=False).head(k)
        return [{"column": c, "value": _num(X.at[row, c]), "median": _num(self.medians_[c]),
                 "robust_z": None if pd.isna(z[c]) else round(float(z[c]), 2),
                 "contribution": round(float(w), 4)} for c, w in top.items()]


def _num(v: Any) -> float | None:
    return None if pd.isna(v) else round(float(v), 4)


# ------------------------------------------------------------------- run
def detect(df: pd.DataFrame, methods: tuple[str, ...] = METHODS, top: int = TOP_K,
           contamination: float = ANOMALY_SHARE, reference: pd.DataFrame | None = None,
           exclude: list[str] | None = None, contract: dict | None = None,
           keep_scores: bool = False) -> dict:
    """Score every row with each method and explain the top rows -> report section dict.

    df / reference: raw files read as text. exclude: e.g. the label column.
    keep_scores: also return the full score Series (for the evaluation script).
    """
    bad = set(methods) - set(METHODS)
    if bad:
        raise ValueError(f"unknown method(s) {sorted(bad)}; use {', '.join(METHODS)}")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    t0 = time.perf_counter()
    clean = prepare(df)
    fit_raw = reference if reference is not None else df
    fit_clean = prepare(reference) if reference is not None else clean
    features, skipped = choose_features(fit_raw, fit_clean, exclude, contract)
    if not features:
        raise ValueError("no numeric feature columns to look at")
    model = AnomalyModel(contamination).fit(fit_clean[features])
    X = clean.reindex(columns=features)
    prep_seconds = time.perf_counter() - t0

    section: dict[str, Any] = {
        "rows": int(len(df)), "fitted_on": "reference" if reference is not None else "this file",
        "features": features, "excluded": skipped, "contamination": contamination,
        "settings": {"iforest_trees": 100, "seed": SEED, "zscore_flag": Z_FLAG,
                     "iqr_flag": "at least one column outside the fences"},
        "prepare_seconds": round(prep_seconds, 2), "methods": {}}
    scores_out = {}
    for m in methods:
        t = time.perf_counter()
        scores = model.score(X, m)
        seconds = time.perf_counter() - t
        flags = model.flagged(scores, m)
        order = scores.sort_values(ascending=False, kind="mergesort").index[:top]
        rows = [{"rank": i + 1, "row": int(df.index.get_loc(r)),
                 "score": round(float(scores[r]), 4), "reasons": model.explain(X, r, m)}
                for i, r in enumerate(order)]
        section["methods"][m] = {"seconds": round(seconds, 3), "flagged": int(flags.sum()),
                                 "flagged_share": round(float(flags.mean()), 4), "top": rows}
        if keep_scores:
            scores_out[m] = scores
    if keep_scores:
        section["_scores"] = scores_out
    return section


# ------------------------------------------------------------- evaluation
def evaluate(scores: pd.Series, labels: pd.Series, k: int | None = None) -> dict:
    """precision@k, recall@k, ROC-AUC and PR-AUC of anomaly scores against 0/1 labels.

    k defaults to the number of true anomalies (then precision@k = recall@k).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = labels.astype(int).to_numpy()
    s = scores.to_numpy(dtype=float)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError("labels need both normal and anomalous rows")
    k = k or positives
    top = np.argsort(-s, kind="mergesort")[:k]
    hits = int(y[top].sum())
    return {"k": k, "precision_at_k": hits / k, "recall_at_k": hits / positives,
            "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}
