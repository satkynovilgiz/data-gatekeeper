"""Module 1 — Advanced Data Profiling & Metadata Intelligence.

Main entry points:
    profile_file(path)                         -> dict   (read a CSV + profile it)
    profile_dataset(df, dataset_name)          -> dict   (the profiling report)
    save_report(report, out_dir)               -> path to profiling_report.json
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .semantic import (PII_TYPES, date_parse_rate, detect_semantic,
                       infer_basic_type, numeric_parse_rate)

REPORT_VERSION = "1.1"
HISTOGRAM_BINS = 10
DOMINANT_VALUE_PCT = 90     # one value in more than 90% of rows -> suspicious

# Text that people type instead of leaving a cell empty ("hidden" missing values).
# Compared after strip() + lower(). pandas already turns "", "NA", "null"... into NaN
# when reading a CSV, but files and API uploads can still contain these.
PLACEHOLDERS = {"unknown", "unk", "n/a", "na", "none", "null", "nan", "?", "-", "--",
                "missing", "not available", "undefined", ""}

# Rules that mark a column as "suspicious" in the summary
SUSPICIOUS_RULES = {"id_stored_as_number", "dominant_value", "constant_column",
                    "placeholder_values"}


# ---------------------------------------------------------------- helpers
def _py(v):
    """Convert numpy/pandas scalars to plain Python so json.dump works."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def _value_types(series: pd.Series) -> dict:
    """Count Python types of the non-null values, e.g. {'float': 480, 'str': 15}.

    Values read from CSV are all strings, so we classify each one by what it
    *looks like* (number vs text) instead of trusting the Python type.
    """
    counts: dict[str, int] = {}
    for v in series.dropna():
        if isinstance(v, str):
            t = "number" if pd.notna(pd.to_numeric(v, errors="coerce")) else "text"
        elif isinstance(v, (int, float, np.integer, np.floating)):
            t = "number"
        else:
            t = type(v).__name__
        counts[t] = counts.get(t, 0) + 1
    return counts


def _histogram(s: pd.Series, bins: int = HISTOGRAM_BINS) -> dict:
    """Split the value range into equal-width bins and count values per bin.

    bin_edges has one more item than counts: bin i covers [edges[i], edges[i+1]).
    """
    counts, edges = np.histogram(s, bins=bins)
    return {"bin_edges": [_py(e) for e in edges], "counts": [int(c) for c in counts]}


def _numeric_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lo) | (s > hi)]
    return {
        "mean": _py(s.mean()), "std": _py(s.std()), "min": _py(s.min()),
        "p25": _py(q1), "median": _py(s.median()), "p75": _py(q3), "max": _py(s.max()),
        "skew": _py(s.skew()),
        "outliers_iqr": {"count": int(len(outliers)),
                         "pct": _py(len(outliers) / len(s) * 100),
                         "lower_fence": _py(lo), "upper_fence": _py(hi)},
        "histogram": _histogram(s),
    }


def _categorical_stats(s: pd.Series, top_n: int = 10) -> dict:
    vc = s.dropna().astype(str).value_counts()
    # Values that only differ by case/whitespace => inconsistent formatting
    norm = s.dropna().astype(str).str.strip().str.lower()
    variants = {}
    for key, grp in s.dropna().astype(str).groupby(norm):
        forms = sorted(set(grp))
        if len(forms) > 1:
            variants[key] = forms
    total = int(vc.sum())
    top = vc.head(top_n)
    return {
        "top_values": {k: int(v) for k, v in top.items()},
        # same top values as a share of non-null rows, e.g. {"married": 60.52}
        "top_values_pct": {k: round(v / total * 100, 2) for k, v in top.items()},
        # how many rows fall outside the top N (the "long tail")
        "other_values_count": total - int(top.sum()),
        "case_or_space_variants": variants,
    }


def _top_value(s: pd.Series) -> dict | None:
    """The most common value and its share of non-null rows, e.g. {"value": "999", "pct": 96.32}."""
    vc = s.dropna().astype(str).value_counts()
    if vc.empty:
        return None
    return {"value": vc.index[0], "pct": round(vc.iloc[0] / vc.sum() * 100, 2)}


def _placeholder_values(s: pd.Series) -> dict:
    """Count placeholder texts like 'unknown' or '?', e.g. {"unknown": 8597}."""
    text = s.dropna().astype(str)
    hits = text[text.str.strip().str.lower().isin(PLACEHOLDERS)]
    return {k: int(v) for k, v in hits.value_counts().items()}


def _date_formats(s: pd.Series) -> dict:
    """Group date strings by their shape, e.g. 9999-99-99 vs 99/99/9999."""
    shapes = s.dropna().astype(str).str.replace(r"\d", "9", regex=True)
    return {k: int(v) for k, v in shapes.value_counts().head(5).items()}


def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Cramér's V: how strongly two categorical columns are linked (0 = not at all, 1 = fully).

    Steps: count every (a, b) combination (contingency table), compare it with
    the counts we would expect if a and b were independent (chi-squared), then
    scale the result to 0..1.
    """
    table = pd.crosstab(a, b).to_numpy(dtype=float)   # rows with a missing value are dropped
    n = table.sum()
    k = min(table.shape) - 1
    if n == 0 or k == 0:
        return None                                    # need at least 2 categories on each side
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return round(float(np.sqrt(chi2 / n / k)), 3)


def categorical_association(df: pd.DataFrame, columns: list[dict],
                            max_categories: int = 50, strong: float = 0.5) -> dict:
    """Cramér's V for every pair of categorical columns (skips columns with too many categories)."""
    cats = [c["name"] for c in columns
            if c["inferred_type"] == "categorical" and 2 <= c["unique_count"] <= max_categories]
    matrix = {a: {b: 1.0 if a == b else None for b in cats} for a in cats}
    strong_pairs = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            v = cramers_v(df[a], df[b])
            matrix[a][b] = matrix[b][a] = v
            if v is not None and v >= strong:
                strong_pairs.append({"a": a, "b": b, "cramers_v": v})
    return {"method": "cramers_v", "matrix": matrix, "strong_pairs": strong_pairs}


# ---------------------------------------------------------- column profile
def profile_column(name: str, series: pd.Series) -> dict:
    n = len(series)
    missing = int(series.isna().sum())
    basic = infer_basic_type(series)
    semantic = detect_semantic(name, series, basic)
    nunique = int(series.nunique(dropna=True))
    vtypes = _value_types(series)

    col = {
        "name": name,
        "pandas_dtype": str(series.dtype),
        "inferred_type": basic,
        "semantic": semantic,
        "is_pii": semantic["semantic_type"] in PII_TYPES,
        "missing_count": missing,
        "missing_pct": round(missing / n * 100, 2) if n else 0.0,
        "unique_count": nunique,
        "unique_pct": round(nunique / max(n - missing, 1) * 100, 2),
        "value_types": vtypes,
        "is_mixed_type": len(vtypes) > 1,
        "top_value": _top_value(series),
        "placeholder_values": _placeholder_values(series),
    }
    if basic == "numeric" or numeric_parse_rate(series) >= 0.8:
        col["numeric_stats"] = _numeric_stats(series)
    if basic in ("categorical", "text"):
        col["categorical_stats"] = _categorical_stats(series)
    if semantic["semantic_type"] == "date":
        col["date_formats"] = _date_formats(series)
    return col


# ----------------------------------------------------------------- rules
def run_rules(df: pd.DataFrame, columns: list[dict]) -> list[dict]:
    """Profiling rule engine: flag suspicious columns, inconsistent formats, PII."""
    issues = []

    def add(col, rule, severity, detail):
        issues.append({"column": col, "rule": rule, "severity": severity, "detail": detail})

    for c in columns:
        name = c["name"]
        if c["missing_pct"] >= 50:
            add(name, "mostly_missing", "high", f"{c['missing_pct']}% missing")
        elif c["missing_pct"] >= 5:
            add(name, "missing_values", "medium", f"{c['missing_pct']}% missing")
        if c["unique_count"] <= 1:
            add(name, "constant_column", "medium", "only one distinct value")
        if c["is_mixed_type"]:
            add(name, "mixed_types", "high", f"value types found: {c['value_types']}")
        if c["is_pii"]:
            add(name, "potential_pii", "info",
                f"looks like {c['semantic']['semantic_type']} — handle as personal data")
        variants = c.get("categorical_stats", {}).get("case_or_space_variants", {})
        if variants:
            add(name, "inconsistent_category_format", "medium",
                f"{len(variants)} values written in several ways, e.g. "
                f"{next(iter(variants.values()))}")
        if len(c.get("date_formats", {})) > 1:
            add(name, "inconsistent_date_format", "medium",
                f"several date formats: {list(c['date_formats'])}")
        out = c.get("numeric_stats", {}).get("outliers_iqr", {})
        if out.get("pct", 0) >= 1:
            add(name, "outliers", "medium",
                f"{out['count']} values outside [{out['lower_fence']}, {out['upper_fence']}]")
        sem = c["semantic"]["semantic_type"]

        # --- suspicious-column rules (Sprint 2)
        if sem == "id" and c["inferred_type"] == "numeric":
            add(name, "id_stored_as_number", "medium",
                "identifier stored as a number — store as text (no maths on IDs, "
                "leading zeros are kept)")
        top = c["top_value"]
        if top and c["unique_count"] > 1 and top["pct"] > DOMINANT_VALUE_PCT:
            add(name, "dominant_value", "medium",
                f"{top['pct']}% of values are '{top['value']}' — almost constant, "
                f"or a placeholder/magic number")
        if c["placeholder_values"]:
            n_ph = sum(c["placeholder_values"].values())
            add(name, "placeholder_values", "medium",
                f"{n_ph} values ({round(n_ph / len(df) * 100, 2)}%) look like hidden "
                f"missing data: {c['placeholder_values']}")

        if sem in ("email", "phone", "postcode") and c["semantic"]["confidence"] < 0.99:
            bad = round((1 - c["semantic"]["confidence"]) * 100, 1)
            add(name, "invalid_format_values", "medium",
                f"~{bad}% of non-null values do not match the {sem} format")

    dup = int(df.duplicated().sum())
    if dup:
        add("<dataset>", "duplicate_rows", "high", f"{dup} exact duplicate rows")
    return issues


# --------------------------------------------------------------- dataset
def profile_dataset(df: pd.DataFrame, dataset_name: str = "dataset") -> dict:
    columns = [profile_column(c, df[c]) for c in df.columns]

    numeric_cols = [c["name"] for c in columns if "numeric_stats" in c]
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr().round(3) if len(numeric_cols) > 1 else pd.DataFrame()
    strong = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= 0.7:
                strong.append({"a": a, "b": b, "corr": float(v)})

    total_cells = df.size or 1
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": dataset_name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_cells": int(df.isna().sum().sum()),
            "missing_cells_pct": round(df.isna().sum().sum() / total_cells * 100, 2),
        },
        "columns": columns,
        "missing_matrix": {c: int(df[c].isna().sum()) for c in df.columns},
        "correlation": {
            "method": "pearson",
            "matrix": {a: {b: _py(corr.loc[a, b]) for b in corr.columns} for a in corr.index},
            "strong_pairs": strong,
        },
        "categorical_association": categorical_association(df, columns),
    }
    report["issues"] = run_rules(df, columns)
    report["summary"] = {
        "pii_columns": [c["name"] for c in columns if c["is_pii"]],
        "mixed_type_columns": [c["name"] for c in columns if c["is_mixed_type"]],
        "suspicious_columns": sorted({i["column"] for i in report["issues"]
                                      if i["rule"] in SUSPICIOUS_RULES}),
        "issue_counts": {
            sev: sum(1 for i in report["issues"] if i["severity"] == sev)
            for sev in ("high", "medium", "info")
        },
    }
    # numpy numbers -> plain Python, so the report can go straight to JSON / the API
    return json.loads(json.dumps(report, default=_py))


def detect_separator(path: str | Path) -> str:
    """Guess the CSV separator (, ; tab |) from the first lines of the file."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(64_000)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","   # could not tell (e.g. a single column) -> use the default


def read_csv_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV with every value kept as text (dtype=str).

    Keeping text means mixed types ("5" vs "five") are not hidden by pandas;
    the profiler decides the real type itself.
    """
    return pd.read_csv(path, sep=detect_separator(path), dtype=str,
                       keep_default_na=True, encoding_errors="replace")


def profile_file(path: str | Path, dataset_name: str | None = None) -> dict:
    """Read a CSV file and return its profiling report (dict)."""
    path = Path(path)
    df = read_csv_raw(path)
    return profile_dataset(df, dataset_name=dataset_name or path.name)


def save_report(report: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "profiling_report.json"
    path.write_text(json.dumps(report, indent=2, default=_py))
    return path
