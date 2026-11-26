"""Anomaly detection for validation (Sprint 9): which ROWS look unlike the others, and why?

Three methods, all built on src/cleaning/outliers.py (no second copy of the maths):

  zscore   robust z per numeric column, |x - median| / (1.4826 x MAD); row score = the largest.
           Finds a single extreme value. Columns with MAD = 0 are not used (like IQR = 0).
  iqr      per column, how far a value is outside the IQR fences, in IQR units; row score =
           number of columns outside + a tie-break by total distance (n + d / (1 + d)).
  iforest  Isolation Forest (100 trees, seed 42), score = -score_samples (higher = odder);
           `contamination` (default 1%) sets the flag threshold. Sees all columns at once, so
           it can find odd COMBINATIONS that no single column shows.

Sprint 10 adds two methods NEXT TO the old ones (so the Sprint 9 numbers stay comparable):
  zscore_mixture  for a column with several "bumps" (euribor3m: about a third of the rows
           near 1, the rest near 4.9), one median + MAD calls a whole bump extreme. Here a
           Gaussian mixture with 1-3 components is fitted per column (BIC picks the number,
           seed 42), and a value is scored by its distance to the NEAREST bump in that bump's
           own standard deviations. When BIC picks 1 bump, the old robust z is used. Guards
           decided before any evaluation: the mixture is fitted on standardised values (robust
           scale, or the SD when MAD = 0) on at most 5,000 sampled values, and no bump's SD may
           fall below 0.1 of that scale (a discrete column must not get a zero-width bump that
           calls every other value extreme).
  iforest_with_categories  the forest also sees the categorical columns: frequency encoding
           by default (a category becomes its share of the rows, so a rare category is a small
           number and an unseen one is 0), one-hot as an option. Free text (> 50 values and
           mostly unique), IDs, PII and the label are never used.

How a top row is explained (3 columns, each with value, column median and robust z):
  zscore / iqr   the columns with the largest robust z / IQR distance.
  iforest        leave-one-column-out: put ONE column of the row back to its median, score the
                 row again with the same forest, and see how much less odd it became. The column
                 whose "repair" lowers the score most contributed most. This explains the forest's
                 own decision; it is not a z-score in disguise. A categorical column is "repaired"
                 to its most common category (shown as the median).
  zscore_mixture the columns with the largest mixture z (also given as mixture_z).

What the methods look at (decided before any evaluation, never tuned on labels):
  - the data after the gatekeeper's row-wise cleaning (profile + steps 1-7): codes such as
    pdays = 999 ("never contacted") are known codes, not anomalies; impossible or
    out-of-range values are KEPT (steps 8 / 8b skipped), because they are what we look for
  - numeric columns (categories too for iforest_with_categories); never the label, IDs,
    PII (src/privacy), free text or columns > 50% empty
  - fitted on a reference file when one is given (then new rows are judged against it),
    otherwise on the file itself
Rows are reported by their position in the file (0 = first data row), never by an ID; the
data is never changed.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

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

METHODS = ("iforest", "zscore", "iqr", "zscore_mixture", "iforest_with_categories")
SPRINT9_METHODS = ("iforest", "zscore", "iqr")
ENCODINGS = ("frequency", "onehot")
MAX_COMPONENTS = 3      # mixture: 1, 2 or 3 bumps per column
MIXTURE_SAMPLE = 5000   # values used to fit one column's mixture (seed 42)
SD_FLOOR = 0.1          # a bump's SD is at least 0.1 x the column's scale
MAX_CATEGORIES = 50     # more values than this AND mostly unique = free text, not a category
MISSING = "<missing>"
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
                    contract: dict | None = None
                    ) -> tuple[list[str], list[str], dict[str, str]]:
    """(numeric features, categorical features, {column: why it is left out})."""
    exclude = set(exclude or [])
    pii = pii_columns(raw, contract)
    features, categorical, skipped = [], [], {}
    for c in clean.columns:
        s = clean[c]
        if c in exclude:
            skipped[c] = "label / excluded"
        elif c in pii:
            skipped[c] = f"personal data ({pii[c]})"
        elif is_id_name(c):
            skipped[c] = "identifier"
        elif s.isna().mean() > MAX_MISSING:
            skipped[c] = f"{s.isna().mean():.0%} empty"
        elif pd.api.types.is_numeric_dtype(s) or numeric_parse_rate(s) >= NUMERIC_SHARE:
            features.append(c)
        elif s.nunique() > MAX_CATEGORIES and s.nunique() > 0.5 * s.notna().sum():
            skipped[c] = "free text (too many different values)"
        else:
            categorical.append(c)
    return features, categorical, skipped


def fit_mixture(x: pd.Series, seed: int = SEED) -> dict | None:
    """1-3 Gaussian bumps for one numeric column (BIC); None when 1 bump is best.

    Returns {"k", "means", "sds"} in the column's own units.
    """
    v = x.dropna().astype(float)
    if len(v) < 20 or v.nunique() < 2:
        return None
    scale = robust_scale(v) or float(v.std()) or None
    if not scale:
        return None
    center = float(v.median())
    z = ((v - center) / scale).to_numpy()
    if len(z) > MIXTURE_SAMPLE:
        z = np.random.default_rng(seed).choice(z, MIXTURE_SAMPLE, replace=False)
    best = None
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", ConvergenceWarning)
        for k in range(1, min(MAX_COMPONENTS, int(v.nunique())) + 1):
            gm = GaussianMixture(k, random_state=seed, reg_covar=SD_FLOOR ** 2)
            gm.fit(z.reshape(-1, 1))
            bic = gm.bic(z.reshape(-1, 1))
            if best is None or bic < best[0]:
                best = (bic, k, gm)
    _, k, gm = best
    if k == 1:
        return None
    sds = np.maximum(np.sqrt(gm.covariances_.reshape(-1)), SD_FLOOR) * scale
    return {"k": int(k), "means": [center + float(m) * scale for m in gm.means_.reshape(-1)],
            "sds": [float(sd) for sd in sds]}


# ---------------------------------------------------------------- fitting
class AnomalyModel:
    """Medians, robust scales, IQR fences, mixtures, category shares and two forests,
    all learned on one table."""

    def __init__(self, contamination: float = ANOMALY_SHARE, seed: int = SEED,
                 encoding: str = "frequency") -> None:
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {', '.join(ENCODINGS)}")
        self.contamination = contamination
        self.seed = seed
        self.encoding = encoding

    def fit(self, X: pd.DataFrame, categorical: list[str] | None = None) -> AnomalyModel:
        """Learn everything from X: the numeric features, plus `categorical` columns of X."""
        self.cat_columns_ = list(categorical or [])
        num = X.drop(columns=self.cat_columns_).astype(float)
        self.columns_ = list(num.columns)
        self.medians_ = num.median()
        self.scales_ = {c: robust_scale(num[c]) for c in self.columns_}
        self.fences_ = {c: iqr_fences(num[c]) for c in self.columns_}
        filled = fill_for_forest(num, self.medians_)
        self.forest_ = isolation_forest_model(filled, self.contamination, self.seed)
        self.offset_ = float(-self.forest_.offset_)    # iforest score above this = flagged
        self.mixtures_ = {c: fit_mixture(num[c], self.seed) for c in self.columns_}
        # categories: shares (frequency), lists (one-hot) and the most common value
        self.shares_, self.categories_, self.modes_ = {}, {}, {}
        for c in self.cat_columns_:
            text = X[c].astype(object).where(X[c].notna(), MISSING).astype(str)
            self.shares_[c] = text.value_counts(normalize=True).to_dict()
            self.categories_[c] = sorted(text.unique())
            self.modes_[c] = text.value_counts().idxmax()
        both = pd.concat([filled, self.encode_categories(X)], axis=1)
        self.forest_cat_ = isolation_forest_model(both, self.contamination, self.seed)
        self.offset_cat_ = float(-self.forest_cat_.offset_)
        return self

    def encode_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        """Categorical columns as numbers: share of the category (or one 0/1 per category)."""
        out = {}
        for c in self.cat_columns_:
            text = X[c].astype(object).where(X[c].notna(), MISSING).astype(str)
            if self.encoding == "frequency":
                out[c] = text.map(self.shares_[c]).fillna(0.0).astype(float)
            else:
                for v in self.categories_[c]:
                    out[f"{c}={v}"] = (text == v).astype(float)
        return pd.DataFrame(out, index=X.index)

    # per-cell views -----------------------------------------------------
    def robust_z(self, X: pd.DataFrame) -> pd.DataFrame:
        """|x - median| / scale per cell; NaN where the value is empty or the column unused."""
        out = {}
        for c in self.columns_:
            scale = self.scales_[c]
            x = X[c].astype(float)
            out[c] = (x - self.medians_[c]).abs() / scale if scale else x * np.nan
        return pd.DataFrame(out, index=X.index)

    def mixture_z(self, X: pd.DataFrame) -> pd.DataFrame:
        """Distance to the nearest bump in its own SDs; the robust z where 1 bump fits best."""
        z = self.robust_z(X)
        for c, mix in self.mixtures_.items():
            if mix is None:
                continue
            x = X[c].astype(float).to_numpy()[:, None]
            d = np.abs(x - np.array(mix["means"])[None, :]) / np.array(mix["sds"])[None, :]
            z[c] = np.where(np.isnan(x[:, 0]), np.nan, d.min(axis=1))
        return z

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
        if method == "zscore_mixture":
            return self.mixture_z(X).max(axis=1, skipna=True).fillna(0.0)
        if method == "iforest_with_categories":
            both = pd.concat([fill_for_forest(X[self.columns_], self.medians_),
                              self.encode_categories(X)], axis=1)
            return pd.Series(-self.forest_cat_.score_samples(both.to_numpy()), index=X.index)
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")

    def flagged(self, scores: pd.Series, method: str) -> pd.Series:
        """Which rows count as anomalies for a method (its documented threshold)."""
        if method == "iforest":
            return scores > self.offset_
        if method == "iforest_with_categories":
            return scores > self.offset_cat_
        if method in ("zscore", "zscore_mixture"):
            return scores > Z_FLAG
        return scores >= 1                              # iqr: at least one column outside

    # explanations -------------------------------------------------------
    def explain(self, X: pd.DataFrame, row: Any, method: str, k: int = REASONS) -> list[dict]:
        """The k columns that made one row unusual, with value, median and robust z."""
        z = self.robust_z(X.loc[[row]]).iloc[0]
        mz = None
        if method in ("iforest", "iforest_with_categories"):
            cols = self.columns_ + (self.cat_columns_ if method != "iforest" else [])
            base = X.loc[[row], cols]
            score0 = float(self.score(base, method).iloc[0])
            contrib = {}
            for c in cols:
                repaired = base.copy()
                repaired[c] = self.modes_[c] if c in self.cat_columns_ else self.medians_[c]
                contrib[c] = score0 - float(self.score(repaired, method).iloc[0])
            weight = pd.Series(contrib)
        elif method == "iqr":
            weight = self.iqr_distance(X.loc[[row]]).iloc[0].fillna(0)
        elif method == "zscore_mixture":
            mz = self.mixture_z(X.loc[[row]]).iloc[0]
            weight = mz.fillna(0)
        else:
            weight = z.fillna(0)
        top = weight.sort_values(ascending=False, kind="mergesort").head(k)
        out = []
        for c, w in top.items():
            if c in self.cat_columns_:
                v = X.at[row, c]
                out.append({"column": c, "value": MISSING if pd.isna(v) else str(v),
                            "median": self.modes_[c], "robust_z": None,
                            "contribution": round(float(w), 4)})
                continue
            item = {"column": c, "value": _num(X.at[row, c]), "median": _num(self.medians_[c]),
                    "robust_z": None if pd.isna(z[c]) else round(float(z[c]), 2),
                    "contribution": round(float(w), 4)}
            if mz is not None:
                item["mixture_z"] = None if pd.isna(mz[c]) else round(float(mz[c]), 2)
            out.append(item)
        return out


def _num(v: Any) -> float | None:
    return None if pd.isna(v) else round(float(v), 4)


# ------------------------------------------------------------------- run
def detect(df: pd.DataFrame, methods: tuple[str, ...] = METHODS, top: int = TOP_K,
           contamination: float = ANOMALY_SHARE, reference: pd.DataFrame | None = None,
           exclude: list[str] | None = None, contract: dict | None = None,
           keep_scores: bool = False, encoding: str = "frequency") -> dict:
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
    features, categorical, skipped = choose_features(fit_raw, fit_clean, exclude, contract)
    if not features:
        raise ValueError("no numeric feature columns to look at")
    model = AnomalyModel(contamination, encoding=encoding).fit(
        fit_clean[features + categorical], categorical)
    X = clean.reindex(columns=features + categorical)
    prep_seconds = time.perf_counter() - t0

    section: dict[str, Any] = {
        "rows": int(len(df)), "fitted_on": "reference" if reference is not None else "this file",
        "features": features, "categorical_features": categorical, "excluded": skipped,
        "contamination": contamination,
        "settings": {"iforest_trees": 100, "seed": SEED, "zscore_flag": Z_FLAG,
                     "iqr_flag": "at least one column outside the fences",
                     "encoding": encoding,
                     "mixture_components": {c: m["k"] for c, m in model.mixtures_.items()
                                            if m is not None}},
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
