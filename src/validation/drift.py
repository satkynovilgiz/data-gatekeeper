"""Drift: how far has a new dataset moved away from a reference dataset, per column?

PSI (Population Stability Index) compares the share of rows per bin:
    PSI = sum over bins of (cur% - ref%) x ln(cur% / ref%)
  numeric      10 bins cut at the reference's deciles (fewer when the reference has
               ties, e.g. pdays = 999 for 96% of rows)
  categorical  one bin per category (union of both files)
  missing      empty cells get their own bin when either file has any, so a column that
               suddenly has 5% missing values drifts even if the rest looks the same
  A share of 0 is replaced by EPS (0.0001) so ln() stays finite; this is the usual
  convention, and it means a new bin holding p% of rows adds about p x ln(p / EPS).

KS (Kolmogorov-Smirnov, scipy) is reported for numeric columns too: the largest gap between
the two cumulative distributions (0 = identical, 1 = no overlap). With thousands of rows
its p-value is tiny for any small difference, so the status is decided by PSI only.

Status by PSI: < 0.1 stable, 0.1 - 0.25 moderate, > 0.25 major (thresholds from the
contract's `drift:` section when one is given).

Privacy: for PII columns (detected, or `pii: true` in the contract) the report keeps PSI
and KS but no bin edges or category names; unseen values are masked with
src/privacy/mask.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.privacy.mask import mask_value
from src.profiling.semantic import numeric_parse_rate
from src.validation.contract import DRIFT_DEFAULTS, pii_columns

EPS = 1e-4
NUMERIC_SHARE = 0.95       # a column is numeric when >= 95% of reference values are numbers
MAX_CATEGORIES = 50        # more distinct values than this (and mostly unique): not compared
MISSING = "<missing>"
SHOWN_CATEGORIES = 10      # new / vanished categories listed in the report


def psi(ref_counts: np.ndarray | list, cur_counts: np.ndarray | list) -> float:
    """PSI of two aligned count vectors (same bins, same order)."""
    ref = np.asarray(ref_counts, dtype=float)
    cur = np.asarray(cur_counts, dtype=float)
    if ref.sum() == 0 or cur.sum() == 0:
        raise ValueError("PSI needs at least one row in each dataset")
    p = np.clip(ref / ref.sum(), EPS, None)
    q = np.clip(cur / cur.sum(), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def quantile_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Inner bin edges at the reference's quantiles (duplicates removed)."""
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    return np.unique(np.quantile(reference, qs))


def bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Rows per bin; bin i holds edges[i-1] <= x < edges[i] (outer bins open-ended)."""
    return np.bincount(np.searchsorted(edges, values, side="right"),
                       minlength=len(edges) + 1)


def status_for(value: float, moderate: float, major: float) -> str:
    """stable / moderate / major for one PSI value."""
    if value > major:
        return "major"
    if value >= moderate:
        return "moderate"
    return "stable"


def _missing_pct(s: pd.Series) -> float:
    return round(float(s.isna().mean() * 100), 2) if len(s) else 0.0


def numeric_drift(ref: pd.Series, cur: pd.Series, bins: int = 10,
                  show_bins: bool = True) -> dict:
    """PSI over reference-quantile bins (+ a missing bin) and the KS statistic."""
    r = pd.to_numeric(ref, errors="coerce")
    c = pd.to_numeric(cur, errors="coerce")
    r_vals, c_vals = r.dropna().to_numpy(), c.dropna().to_numpy()
    if len(r_vals) == 0 or len(c_vals) == 0:
        return {"kind": "numeric", "psi": None, "status": "skipped",
                "note": "no numeric values in one of the files"}
    edges = quantile_edges(r_vals, bins)
    rc, cc = bin_counts(r_vals, edges), bin_counts(c_vals, edges)
    r_miss, c_miss = int(r.isna().sum()), int(c.isna().sum())
    if r_miss or c_miss:
        rc, cc = np.append(rc, r_miss), np.append(cc, c_miss)
    # For a tiny KS distance on big samples (cons.conf.idx: D = 0.003, 28k vs 12k rows)
    # scipy's p-value routine overflows in an intermediate matrix power and warns, while
    # the p-value it returns (~1.0) is right. Silence that noise; keep None if not finite.
    with np.errstate(all="ignore"):
        ks = ks_2samp(r_vals, c_vals)
    pvalue = float(ks.pvalue) if np.isfinite(ks.pvalue) else None
    out = {"kind": "numeric", "psi": round(psi(rc, cc), 4),
           "ks_statistic": round(float(ks.statistic), 4), "ks_pvalue": pvalue,
           "bins_used": int(len(edges) + 1)}
    if show_bins:
        labels = ([f"< {edges[0]:g}"] if len(edges) else ["all"])
        labels += [f"[{a:g}, {b:g})" for a, b in zip(edges[:-1], edges[1:])]
        labels += [f">= {edges[-1]:g}"] if len(edges) else []
        if r_miss or c_miss:
            labels.append(MISSING)
        out["bins"] = [{"bin": lab, "ref_pct": round(a / rc.sum() * 100, 2),
                        "cur_pct": round(b / cc.sum() * 100, 2)}
                       for lab, a, b in zip(labels, rc, cc)]
    return out


def categorical_drift(ref: pd.Series, cur: pd.Series, pii_type: str | None = None) -> dict:
    """PSI over categories (+ a missing bin); new and vanished categories listed."""
    r = ref.astype("string").fillna(MISSING)
    c = cur.astype("string").fillna(MISSING)
    rv, cv = r.value_counts(), c.value_counts()
    cats = list(dict.fromkeys([*rv.index, *cv.index]))
    rc = np.array([rv.get(k, 0) for k in cats])
    cc = np.array([cv.get(k, 0) for k in cats])
    new = [k for k in cv.index if k not in rv.index and k != MISSING]
    gone = [k for k in rv.index if k not in cv.index and k != MISSING]

    def show(values: list[str]) -> list[str]:
        values = values[:SHOWN_CATEGORIES]
        return [mask_value(v, pii_type) for v in values] if pii_type else values

    out = {"kind": "categorical", "psi": round(psi(rc, cc), 4),
           "new_categories_count": len(new), "new_categories": show(new),
           "new_categories_rows_pct": round(float(cv[new].sum() / cv.sum() * 100), 2),
           "missing_categories_count": len(gone), "missing_categories": show(gone)}
    if not pii_type:
        out["categories"] = [{"category": k, "ref_pct": round(a / rc.sum() * 100, 2),
                              "cur_pct": round(b / cc.sum() * 100, 2)}
                             for k, a, b in zip(cats, rc, cc)]
    return out


def column_kind(ref: pd.Series, dtype: str | None = None) -> str:
    """numeric / categorical / skipped (free text, ids) for one reference column."""
    if dtype in ("integer", "number"):
        return "numeric"
    if dtype is None and numeric_parse_rate(ref) >= NUMERIC_SHARE:
        return "numeric"
    s = ref.dropna()
    n_unique = s.nunique()
    if n_unique <= MAX_CATEGORIES or n_unique <= 0.5 * len(s):
        return "categorical"
    return "skipped"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 contract: dict | None = None, reference_name: str = "reference",
                 current_name: str = "current") -> dict:
    """Compare every shared column of two DataFrames (read as text) -> report dict."""
    cfg = (contract or {}).get("drift", DRIFT_DEFAULTS)
    moderate, major, bins = cfg["psi_moderate"], cfg["psi_major"], cfg["bins"]
    dtypes = {c["name"]: c["rules"].get("dtype", (None,))[0]
              for c in (contract or {}).get("columns", [])}
    pii = {**pii_columns(reference, contract), **pii_columns(current, contract)}

    columns = []
    for name in dict.fromkeys([*map(str, reference.columns), *map(str, current.columns)]):
        entry: dict = {"column": name, "pii": name in pii}
        if name not in reference.columns or name not in current.columns:
            where = "reference" if name not in reference.columns else "current"
            entry.update(kind=None, psi=None, status="skipped", note=f"not in the {where} file")
            columns.append(entry)
            continue
        r, c = reference[name], current[name]
        entry["ref_missing_pct"], entry["cur_missing_pct"] = _missing_pct(r), _missing_pct(c)
        kind = column_kind(r, dtypes.get(name))
        if kind == "numeric":
            entry.update(numeric_drift(r, c, bins, show_bins=name not in pii))
        elif kind == "categorical":
            entry.update(categorical_drift(r, c, pii.get(name)))
        else:
            entry.update(kind="text", psi=None, status="skipped",
                         note="too many distinct values to compare (free text or id)")
        if entry.get("psi") is not None:
            entry["status"] = status_for(entry["psi"], moderate, major)
        columns.append(entry)

    columns.sort(key=lambda e: -1 if e["psi"] is None else e["psi"], reverse=True)
    counts = {s: sum(e["status"] == s for e in columns)
              for s in ("stable", "moderate", "major", "skipped")}
    return {
        "reference": {"name": reference_name, "rows": int(len(reference))},
        "current": {"name": current_name, "rows": int(len(current))},
        "thresholds": {"psi_moderate": moderate, "psi_major": major, "bins": bins,
                       "eps": EPS},
        "status": "major" if counts["major"] else ("moderate" if counts["moderate"]
                                                   else "stable"),
        "counts": counts,
        "major_columns": [e["column"] for e in columns if e["status"] == "major"],
        "pii_columns": sorted(pii),
        "columns": columns,
    }
