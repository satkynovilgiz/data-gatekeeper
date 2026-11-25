"""Imputers — fill missing values, from simple (median / mode) to ML (scikit-learn).

    impute_columns(df, targets, features, method) -> copy of df with the targets filled

Methods:
    median      baseline: median for numbers, mode (most common value) for categories
    mean, mode  simple per-column options (kept from Sprint 4)
    knn         KNNImputer: average (or majority vote) of the k most similar rows
    regression  one model per column, trained on the rows where it is known:
                LinearRegression for numbers, LogisticRegression for categories
    iterative   IterativeImputer (MICE): models every column from the others and repeats
                for several rounds, so the guesses improve each other

ML models only understand numbers, so the table is ENCODED first:
    numeric column      -> z-score (value - mean) / std, so salary (~80,000) does not
                           outweigh age (~40) when KNN measures how similar rows are
    categorical column  -> one-hot: one 0/1 column per category; a missing category = all NaN
and DECODED afterwards: z-score -> original units (clipped to the known min/max, rounded for
whole-number columns), one-hot -> the category with the highest score.
Only the target columns are changed; feature columns are just used to find similar rows.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (needed before import)
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression

ML_METHODS = ("knn", "regression", "iterative")
FOLD_METHODS = ("median", "knn", "iterative")    # methods FoldImputer can fit/apply
METHODS = ("median", "mean", "mode", *ML_METHODS)
KNN_NEIGHBOURS = 5
ITERATIVE_ROUNDS = 10
SEED = 42


def is_numeric(s: pd.Series) -> bool:
    """True for number columns; everything else is treated as a category."""
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def describe(method: str) -> str:
    """Short description of a method for the cleaning log, e.g. 'knn (k=5)'."""
    return {"knn": f"knn (k={KNN_NEIGHBOURS})",
            "regression": "regression (linear / logistic)",
            "iterative": f"iterative (MICE, {ITERATIVE_ROUNDS} rounds)"}.get(method, method)


# --------------------------------------------------------------- encoding
@dataclass
class Encoded:
    """A table turned into a numeric matrix X, and how to turn it back."""

    X: np.ndarray
    slices: dict[str, slice]                  # column -> its matrix columns
    scale: dict[str, tuple[float, float]]     # numeric column -> (mean, std)
    categories: dict[str, list[str]]          # categorical column -> its categories
    drop_first: bool = False                  # True: the first category has no column


def encode(df: pd.DataFrame, columns: list[str], drop_first: bool = False,
           like: Encoded | None = None) -> Encoded:
    """Numbers -> z-scores, categories -> one-hot (all NaN when missing). NaN stays NaN.

    drop_first=True leaves out the first category's column ("all zeros" = first category).
    IterativeImputer needs this: with full one-hot, zone_low = 1 - zone_high exactly, so
    when both are hidden each one is predicted from the other (a circular guess ~0.5).

    like: encode with another table's means / stds / category lists (e.g. new rows with
    what was learned on the train rows), so the columns line up and nothing is re-learned.
    A category the train rows never had becomes all zeros.
    """
    parts, slices, scale, categories = [], {}, {}, {}
    pos = 0
    for c in columns:
        s = df[c]
        numeric = c in like.scale if like else is_numeric(s)
        if numeric:
            x = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
            if like:
                mean, std = like.scale[c]
            else:
                mean, std = float(np.nanmean(x)), float(np.nanstd(x)) or 1.0
            parts.append(((x - mean) / std)[:, None])
            scale[c] = (mean, std)
            width = 1
        else:
            cats = like.categories[c] if like else sorted(s.dropna().astype(str).unique())
            onehot = (s.astype(str).to_numpy()[:, None] == np.array(cats)[None, :]).astype(float)
            onehot[s.isna().to_numpy()] = np.nan
            if drop_first:
                onehot = onehot[:, 1:]
            parts.append(onehot)
            categories[c] = cats
            width = onehot.shape[1]
        slices[c] = slice(pos, pos + width)
        pos += width
    X = np.hstack(parts) if parts else np.empty((len(df), 0))
    return Encoded(X, slices, scale, categories, drop_first)


def decode(enc: Encoded, X: np.ndarray, column: str) -> np.ndarray:
    """One column of a filled matrix back to original units / category labels."""
    block = X[:, enc.slices[column]]
    if column in enc.scale:
        mean, std = enc.scale[column]
        return block[:, 0] * std + mean
    if enc.drop_first:
        # score of the left-out first category = 1 - the sum of the others
        block = np.hstack([1 - block.sum(axis=1, keepdims=True), block])
    return np.array(enc.categories[column], dtype=object)[block.argmax(axis=1)]


def _finish(original: pd.Series, values: np.ndarray) -> pd.Series:
    """Fill only the gaps of original with values; keep numbers inside the known min/max.

    A model can predict impossible numbers (age -3); clipping to the known range and
    rounding whole-number columns keeps the filled values realistic.
    """
    gaps = original.isna().to_numpy()
    filled = pd.Series(values, index=original.index)
    if is_numeric(original):
        known = original.dropna().astype(float)
        filled = filled.astype(float).clip(known.min(), known.max())
        if (known % 1 == 0).all():
            filled = filled.round()
    out = original.copy() if not is_numeric(original) else original.astype(float)
    out[gaps] = filled[gaps]
    return out


# ---------------------------------------------------------------- methods
def _simple(s: pd.Series, method: str) -> pd.Series:
    """median / mean / mode for one column ('median' on a category means its mode)."""
    known = s.dropna()
    if method == "mode" or not is_numeric(s):
        value = known.mode().sort_values().iloc[0]         # ties: the first alphabetically
    else:
        value = float(known.median() if method == "median" else known.mean())
        if (known % 1 == 0).all():
            value = round(value)                            # whole-number columns stay whole
    return s.fillna(value)


def _regression(df: pd.DataFrame, target: str, features: list[str]) -> pd.Series:
    """Predict one column from the others: linear (numbers) or logistic (categories)."""
    s = df[target]
    enc = encode(df, [f for f in features if f != target])
    X = enc.X
    if X.shape[1] == 0:
        return _simple(s, "median")
    # feature gaps: use the column mean (0 for a z-score, the category share for one-hot)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = np.nan_to_num(X)                                    # a feature that is empty everywhere
    known = s.notna().to_numpy()
    if not is_numeric(s) and s[known].nunique() < 2:        # one class only: nothing to learn
        return _simple(s, "mode")
    values = np.empty(len(s), dtype=object if not is_numeric(s) else float)
    # same false macOS Accelerate "matmul" flags as in KNN (checked: results are finite)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if is_numeric(s):
            model = LinearRegression().fit(X[known], s[known].astype(float))
        else:
            model = LogisticRegression(max_iter=1000).fit(X[known], s[known].astype(str))
        values[~known] = model.predict(X[~known])
    return _finish(s, values)


def impute_columns(df: pd.DataFrame, targets: list[str], features: list[str],
                   method: str) -> pd.DataFrame:
    """Return a copy of df where the gaps in targets are filled with method.

    targets:  the columns to fill.
    features: the columns the ML methods may look at (targets are added automatically).
              They are never changed. Use only numeric / categorical columns, never IDs
              or personal data.
    """
    if method not in METHODS:
        raise ValueError(f"unknown imputation method {method!r}; use one of {METHODS}")
    out = df.copy()
    targets = [t for t in targets if out[t].isna().any() and out[t].notna().any()]
    if not targets:
        return out
    if method in ("median", "mean", "mode"):
        for t in targets:
            out[t] = _simple(out[t], method)
        return out
    if method == "regression":
        for t in targets:
            out[t] = _regression(df, t, features)
        return out

    # knn and iterative fill the whole encoded matrix at once
    columns = list(dict.fromkeys([*targets, *features]))       # targets first, no repeats
    columns = [c for c in columns if df[c].notna().any()]      # sklearn drops empty columns
    enc = encode(df, columns, drop_first=(method == "iterative"))
    if method == "knn":
        # numpy 2 with macOS Accelerate raises false "divide by zero in matmul" flags
        # inside the distance calculation; the distances themselves are correct
        # (checked: same result as einsum, all values finite)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            X = KNNImputer(n_neighbors=KNN_NEIGHBOURS).fit_transform(enc.X)
    else:
        # "early stopping not reached" only means the guesses still moved a little after
        # ITERATIVE_ROUNDS rounds; the evaluation measures how good the fills really are
        with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore",
                                                    invalid="ignore"):
            warnings.simplefilter("ignore", ConvergenceWarning)   # + the false matmul flags
            X = IterativeImputer(max_iter=ITERATIVE_ROUNDS,
                                 random_state=SEED).fit_transform(enc.X)
    for t in targets:
        out[t] = _finish(df[t], decode(enc, X, t))
    return out


# ------------------------------------------------------- fit on train, apply to new
class FoldImputer:
    """The same imputation as impute_columns, split into fit (train rows) and transform.

    impute_columns fills the table it learns from, which is right for cleaning one file.
    To measure a model honestly (Sprint 8, cross-validation) the imputer must learn on the
    train rows only and then fill the test rows with what it learned. FoldImputer does
    that with the same encoding, the same models and the same clipping:
        median     median (whole numbers stay whole) / mode, learned on train
        knn        KNNImputer on the train encoding
        iterative  IterativeImputer (MICE, 10 rounds, seed 42) on the train encoding
    targets: the columns to fill; features: the columns the ML methods may look at.
    Works on DataFrames (scikit-learn style fit / transform, usable in a Pipeline).
    """

    def __init__(self, method: str = "iterative", targets: list[str] | None = None,
                 features: list[str] | None = None) -> None:
        self.method = method
        self.targets = targets
        self.features = features

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn: the constructor arguments (so a Pipeline can clone it)."""
        return {"method": self.method, "targets": self.targets, "features": self.features}

    def set_params(self, **params: object) -> FoldImputer:
        """scikit-learn: change constructor arguments."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: pd.DataFrame, y: object = None) -> FoldImputer:
        """Learn fill values / the imputation model from the train rows X."""
        if self.method not in FOLD_METHODS:
            raise ValueError(f"FoldImputer method must be one of {FOLD_METHODS}")
        targets = [t for t in (self.targets if self.targets is not None else list(X.columns))
                   if t in X.columns and X[t].notna().any()]
        self.targets_ = targets
        # known range of each numeric target on train, for clipping like _finish()
        self.ranges_ = {}
        for t in targets:
            if is_numeric(X[t]):
                known = X[t].dropna().astype(float)
                self.ranges_[t] = (float(known.min()), float(known.max()),
                                   bool((known % 1 == 0).all()))
        if self.method == "median":
            self.values_ = {t: _fill_value(X[t]) for t in targets}
            return self
        columns = list(dict.fromkeys([*targets, *(self.features or [])]))
        self.columns_ = [c for c in columns if c in X.columns and X[c].notna().any()]
        self.encoding_ = encode(X, self.columns_, drop_first=(self.method == "iterative"))
        with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore",
                                                    invalid="ignore"):
            warnings.simplefilter("ignore", ConvergenceWarning)
            if self.method == "knn":
                self.model_ = KNNImputer(n_neighbors=KNN_NEIGHBOURS).fit(self.encoding_.X)
            else:
                self.model_ = IterativeImputer(max_iter=ITERATIVE_ROUNDS,
                                               random_state=SEED).fit(self.encoding_.X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill the gaps in X's target columns with what was learned in fit()."""
        out = X.copy()
        if self.method == "median":
            for t, value in self.values_.items():
                if t in out.columns:
                    out[t] = out[t].fillna(value)
            return out
        enc = encode(X, self.columns_, drop_first=self.encoding_.drop_first,
                     like=self.encoding_)
        with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore",
                                                    invalid="ignore"):
            warnings.simplefilter("ignore", ConvergenceWarning)
            filled = self.model_.transform(enc.X)
        for t in self.targets_:
            gaps = out[t].isna().to_numpy()
            if not gaps.any():
                continue
            values = decode(self.encoding_, filled, t)
            if t in self.ranges_:
                lo, hi, whole = self.ranges_[t]
                values = np.clip(values.astype(float), lo, hi)
                if whole:
                    values = np.round(values)
                col = pd.to_numeric(out[t], errors="coerce").astype(float)
            else:
                col = out[t].astype(object)
            col[gaps] = values[gaps]
            out[t] = col
        return out

    def fit_transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """fit(X) then transform(X)."""
        return self.fit(X, y).transform(X)


def _fill_value(s: pd.Series) -> object:
    """The single value _simple(s, "median") fills a gap with (median / mode)."""
    gap = pd.Series([np.nan], dtype=float if is_numeric(s) else object)
    return _simple(pd.concat([s, gap], ignore_index=True), "median").iloc[-1]
