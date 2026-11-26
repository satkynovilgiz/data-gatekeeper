"""Decide, per column, how to turn cleaned data into numbers a model can use.

Every decision is made on the TRAIN rows only and comes with a reason in plain words,
so the transform report can explain itself:

  dropped    IDs and PII (no signal, privacy risk), the target, dates, free text,
             numeric columns that are mostly empty (pdays: previously_contacted holds it)
  binary     0/1 flags (salary_outlier, is_anomaly) pass through unchanged
  ordinal    categories with a real order (education: basic.4y < high.school < university)
             -> codes 0..k scaled to 0..1; values outside the order ("unknown") get their
             own 0/1 indicator column, so "unknown" stays a category of its own
  onehot     other categories -> one 0/1 column per value ("default_unknown" included);
             more than MAX_ONEHOT values -> the rare ones share one "infrequent" column
  numeric    |skew| > SKEW_LIMIT -> log1p (values >= 0) or Yeo-Johnson (negatives too),
             then a scaler:
               robust   > OUTLIER_SHARE outliers (median/IQR are not pulled by extremes)
               minmax   few distinct values on a small scale (performance 1-5)
               standard everything else (mean 0, std 1)
  bins       optional groups, e.g. age 18-29 / 30-44 / ...; added next to the column
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.cleaning.outliers import iqr_outliers
from src.profiling.semantic import PII_TYPES, detect_semantic, infer_basic_type

# Bank Marketing education levels, lowest to highest ("unknown" is not on the scale)
ORDINAL_ORDERS = {
    "education": ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",
                  "professional.course", "university.degree"],
}
ORDER_MATCH = 0.5        # a known order is used only if >= 50% of the values are in it
MAX_ONEHOT = 15          # more values than this: rare ones grouped into "infrequent"
SKEW_LIMIT = 1.0         # |skew| above this counts as skewed (a common rule of thumb)
OUTLIER_SHARE = 0.05     # more IQR outliers than this -> robust scaler
SMALL_SCALE = 10         # this many distinct values or fewer -> minmax
MAX_MISSING = 0.5        # numeric column more empty than this is dropped
SAMPLE_ROWS = 2000       # rows used to detect what a column means (fast on big files)
SCALERS = {"standard", "robust", "minmax"}


def yeo_johnson() -> Pipeline:
    """Yeo-Johnson power transform on standardised values.

    Standardising first keeps the numbers small: on raw values like 5,191 (nr.employed)
    or 9,999,999 (salary) the power curve overflows and the fit fails.
    """
    return Pipeline([("center", StandardScaler()),
                     ("power", PowerTransformer(method="yeo-johnson", standardize=False))])


def _skew(values: pd.Series) -> float:
    """Skewness of the non-empty values (0 = symmetric); 0.0 for constant columns."""
    s = values.dropna().astype(float)
    return 0.0 if s.nunique() < 2 else round(float(s.skew()), 3)


def _numeric_plan(column: str, train: pd.Series, scaler: str | None) -> dict:
    """Power transform + scaler for one numeric column, decided on the train values."""
    x = train.dropna().astype(float)
    skew = _skew(x)
    power, why = None, [f"skew {skew}"]
    shaped = x
    if abs(skew) > SKEW_LIMIT:
        # try both, keep the more symmetric one (log only works for values >= 0 and
        # only fixes a long RIGHT tail; Yeo-Johnson learns its own curve for any values)
        tries = {}
        if x.min() >= 0:
            tries["log"] = pd.Series(np.log1p(x.to_numpy()))
        tries["yeo-johnson"] = pd.Series(yeo_johnson().fit_transform(x.to_frame()).ravel())
        best = min(tries, key=lambda k: abs(_skew(tries[k])))
        if abs(_skew(tries[best])) < abs(skew):
            power, shaped = best, tries[best]
            why.append(f"> {SKEW_LIMIT:g} -> {'log1p' if best == 'log' else 'Yeo-Johnson'} "
                       f"(tried {' and '.join(tries)}, kept the more symmetric)")
        else:
            why.append("no power transform makes it more symmetric -> left as it is")
    outliers = float(iqr_outliers(shaped)[0].mean())
    if scaler:
        why.append(f"scaler {scaler} chosen in the config")
    elif outliers > OUTLIER_SHARE:
        scaler = "robust"
        why.append(f"{outliers:.1%} outliers -> robust scaler (median/IQR ignore extremes)")
    elif x.nunique() <= SMALL_SCALE:
        scaler = "minmax"
        why.append(f"{x.nunique()} distinct values on a small scale -> minmax 0..1")
    else:
        scaler = "standard"
        why.append(f"{outliers:.1%} outliers -> standard scaler (mean 0, std 1)")
    return {"column": column, "kind": "numeric", "power": power, "scaler": scaler,
            "skew_before": skew, "reason": "; ".join(why)}


def _category_plan(column: str, train: pd.Series, order: list[str] | None) -> dict:
    """Ordinal (known order) or one-hot plan for one categorical column.

    A known order is only used when it fits the data: at least ORDER_MATCH of the filled
    train rows hold a value from it. Sprint 10 found why: the Bank order for "education"
    (basic.4y < ... < university.degree) was applied to UCI Adult's "education"
    (Bachelors, HS-grad, ...), every value fell outside it, and the column became empty.
    """
    values = set(train.dropna().astype(str))
    known = train.dropna().astype(str)
    if order and len(known) and known.isin(order).mean() < ORDER_MATCH:
        n = len(values)
        return {"column": column, "kind": "onehot", "categories": n,
                "reason": f"{n} categories -> one-hot (the known order for '{column}' matches "
                          f"only {known.isin(order).mean():.0%} of the values, so it is not used)"}
    outside = sorted(values - set(order)) if order else []
    if order:
        reason = "values have an order: " + " < ".join(order)
        if outside:
            reason += f"; {len(outside)} value(s) outside the order (e.g. 'unknown') -> " \
                      "own indicator column"
        return {"column": column, "kind": "ordinal", "order": list(order), "reason": reason}
    n = len(values)
    reason = f"{n} categories -> one-hot"
    if n > MAX_ONEHOT:
        reason += f"; the {MAX_ONEHOT - 1} most common get a column, the rest share 'infrequent'"
    if "unknown" in values:
        reason += "; 'unknown' kept as its own column"
    return {"column": column, "kind": "onehot", "categories": n, "reason": reason}


def _drop_reason(column: str, train: pd.Series, target: str | None,
                 drop: set[str]) -> str | None:
    """Why a column is not used as a model feature (None = it is used)."""
    if column == target:
        return "target column: kept apart, never a feature"
    if column in drop:
        return "dropped in the config"
    sample = train.dropna().astype(str).head(SAMPLE_ROWS)
    if sample.empty:
        return "empty column"
    basic = "numeric" if pd.api.types.is_numeric_dtype(train) else infer_basic_type(sample)
    semantic = detect_semantic(column, sample, basic)["semantic_type"]
    if semantic == "id":
        return "identifier: unique per row, carries no pattern a model can learn"
    if semantic in PII_TYPES:
        return f"personal data ({semantic}): privacy risk, mask it instead (src/privacy)"
    if basic == "datetime" or semantic == "date":
        return "date: needs its own features (year, month, ...), not a scaler"
    if basic == "text":
        return "free text with many different values"
    if basic == "numeric" and train.isna().mean() > MAX_MISSING:
        return f"{train.isna().mean():.0%} empty (a flag column usually carries this)"
    return None


def plan_columns(train: pd.DataFrame, target: str | None = None,
                 ordinal: dict[str, list[str]] | None = None,
                 scaling: dict[str, str] | None = None,
                 bins: dict[str, list[float]] | None = None,
                 drop: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Return (plan, dropped) for the TRAIN rows.

    plan:    one dict per feature column: {column, kind, reason, ...}
    dropped: [{column, reason}] for the columns that are not features
    ordinal / scaling / bins / drop: per-column choices that override the rules.
    """
    orders = {**ORDINAL_ORDERS, **(ordinal or {})}
    scaling = scaling or {}
    bad = {c: s for c, s in scaling.items() if s not in SCALERS}
    if bad:
        raise ValueError(f"unknown scaler {bad}; use one of {sorted(SCALERS)}")
    plan, dropped = [], []
    for c in train.columns:
        s = train[c]
        reason = _drop_reason(c, s, target, set(drop or []))
        if reason:
            dropped.append({"column": c, "reason": reason})
            continue
        if pd.api.types.is_numeric_dtype(s) and set(s.dropna().unique()) <= {0, 1}:
            plan.append({"column": c, "kind": "binary", "reason": "0/1 flag -> kept as it is"})
        elif pd.api.types.is_numeric_dtype(s):
            plan.append(_numeric_plan(c, s, scaling.get(c)))
        else:
            plan.append(_category_plan(c, s, orders.get(c)))
    for c, edges in (bins or {}).items():
        if c not in train.columns or not pd.api.types.is_numeric_dtype(train[c]):
            raise ValueError(f"bins: {c!r} is not a numeric column")
        plan.append({"column": c, "kind": "bins", "edges": [float(e) for e in edges],
                     "labels": bin_labels(edges),
                     "reason": "groups " + ", ".join(bin_labels(edges)) + " (lower edge "
                               "included), one-hot; added next to the column itself"})
    return plan, dropped


def bin_labels(edges: list[float]) -> list[str]:
    """[18, 30, 45] -> ['18-29', '30-44'] for whole numbers, ['0.5-1', ...] otherwise."""
    whole = all(float(e).is_integer() for e in edges)
    return [f"{lo:g}-{hi - 1:g}" if whole else f"{lo:g}-{hi:g}"
            for lo, hi in zip(edges[:-1], edges[1:])]
