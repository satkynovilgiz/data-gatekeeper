"""Module 1 — Advanced Data Profiling & Metadata Intelligence.

Main entry point:
    profile_dataset(df, dataset_name) -> dict   (the profiling report)
    save_report(report, out_dir)               -> path to profiling_report.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .semantic import (PII_TYPES, date_parse_rate, detect_semantic,
                       infer_basic_type, numeric_parse_rate)

REPORT_VERSION = "1.0"


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
    return {
        "top_values": {k: int(v) for k, v in vc.head(top_n).items()},
        "case_or_space_variants": variants,
    }


def _date_formats(s: pd.Series) -> dict:
    """Group date strings by their shape, e.g. 9999-99-99 vs 99/99/9999."""
    shapes = s.dropna().astype(str).str.replace(r"\d", "9", regex=True)
    return {k: int(v) for k, v in shapes.value_counts().head(5).items()}


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
    }
    report["issues"] = run_rules(df, columns)
    report["summary"] = {
        "pii_columns": [c["name"] for c in columns if c["is_pii"]],
        "mixed_type_columns": [c["name"] for c in columns if c["is_mixed_type"]],
        "issue_counts": {
            sev: sum(1 for i in report["issues"] if i["severity"] == sev)
            for sev in ("high", "medium", "info")
        },
    }
    return report


def save_report(report: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "profiling_report.json"
    path.write_text(json.dumps(report, indent=2, default=_py))
    return path
