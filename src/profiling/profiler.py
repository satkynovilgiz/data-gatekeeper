"""Module 1 — Advanced Data Profiling & Metadata Intelligence.

Main entry points:
    profile_file(path)                         -> dict   (read a CSV + profile it)
    profile_dataset(df, dataset_name)          -> dict   (the profiling report)
    save_report(report, out_dir)               -> path to profiling_report.json
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.quality.score import quality_score

from .semantic import (
    PII_TYPES,
    detect_semantic,
    infer_basic_type,
    numeric_parse_rate,
)
from .suggestions import cleaning_suggestions

REPORT_VERSION = "1.2"
HISTOGRAM_BINS = 10
DOMINANT_VALUE_PCT = 90     # one value in more than 90% of rows -> suspicious
SAMPLE_SEED = 42            # fixed seed: the same --sample N always picks the same rows

# Text that people type instead of leaving a cell empty ("hidden" missing values).
# Compared after strip() + lower(). pandas already turns "", "NA", "null"... into NaN
# when reading a CSV, but files and API uploads can still contain these.
PLACEHOLDERS = {"unknown", "unk", "n/a", "na", "none", "null", "nan", "?", "-", "--",
                "missing", "not available", "undefined", ""}

# Numbers that people use as a code for "no value" (e.g. pdays = 999 means
# "never contacted"). They only count as missing when they sit far outside the
# column's normal range — see _numeric_sentinels().
NUMERIC_SENTINELS = [999, 9999, 99999, -1, -99, -999, -9999]

# Rules that mark a column as "suspicious" in the summary
SUSPICIOUS_RULES = {"id_stored_as_number", "dominant_value", "constant_column",
                    "disguised_missing"}


# ---------------------------------------------------------------- helpers
def _py(v: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python so json.dump works.

    Used as json.dumps(default=_py); any other value is returned unchanged.
    """
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def _value_types(series: pd.Series) -> dict:
    """Count how many non-null values look like a number vs text, e.g. {'number': 480, 'text': 15}.

    Values read from CSV are all strings, so we classify each one by what it
    *looks like* instead of trusting the Python type.
    Sprint 3: one pd.to_numeric call for the whole column (vectorised) instead of
    one call per value. Profiling the 41k-row bank dataset went from ~5 s to ~2.5 s.
    """
    s = series.dropna()
    is_number = pd.to_numeric(s, errors="coerce").notna()
    counts = {"number": int(is_number.sum()), "text": int((~is_number).sum())}
    return {k: v for k, v in counts.items() if v}     # drop zero counts


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
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lo) | (s > hi)]
    else:
        # IQR = 0 means at least half the values are identical (e.g. previous = 0).
        # Then the fences collapse to one point and EVERY other value would be an
        # "outlier", so the IQR method is not usable: report no fences, no outliers.
        lo = hi = None
        outliers = s.iloc[:0]
    return {
        "mean": _py(s.mean()), "std": _py(s.std()), "min": _py(s.min()),
        "p25": _py(q1), "median": _py(s.median()), "p75": _py(q3), "max": _py(s.max()),
        "skew": _py(s.skew()),
        "outliers_iqr": {"count": int(len(outliers)),
                         "pct": round(len(outliers) / len(s) * 100, 2),
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


def _text_placeholders(s: pd.Series) -> dict:
    """Count placeholder texts like 'unknown' or '?', e.g. {"unknown": 8597}."""
    text = s.dropna().astype(str)
    hits = text[text.str.strip().str.lower().isin(PLACEHOLDERS)]
    return {k: int(v) for k, v in hits.value_counts().items()}


def _numeric_sentinels(s: pd.Series) -> dict:
    """Find "magic numbers" that mean missing, e.g. {"999": 39673} for pdays.

    A candidate from NUMERIC_SENTINELS only counts when it is clearly separate
    from the real values, so a genuine 999 in a column of salaries is NOT flagged:
      - positive (999, 9999): bigger than 2 x the largest other value
      - negative (-1, -999):  all other values are >= 0
    Returns the original text of the cells, so Module 2 knows exactly what to replace.
    """
    nums = pd.to_numeric(s, errors="coerce")
    found = {}
    for v in NUMERIC_SENTINELS:
        is_v = nums == v
        others = nums[~is_v].dropna()
        if not is_v.any() or others.empty:
            continue
        if (v > 0 and v > 2 * others.max()) or (v < 0 and others.min() >= 0):
            for text, n in s[is_v].astype(str).value_counts().items():
                found[text] = int(n)
    return found


def _disguised_missing(s: pd.Series, numeric: bool) -> dict:
    """Values that mean "missing" but are not empty: placeholder text + numeric sentinels."""
    values = _text_placeholders(s)
    if numeric:
        values.update(_numeric_sentinels(s))
    count = sum(values.values())
    return {"count": count,
            "pct": round(count / len(s) * 100, 2) if len(s) else 0.0,
            "values": values}


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
    is_numeric = basic == "numeric" or numeric_parse_rate(series) >= 0.8
    disguised = _disguised_missing(series, is_numeric)
    true_missing = missing + disguised["count"]

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
        # hidden missing values ("unknown", "?", 999, -1 ...) and the honest total
        "disguised_missing": disguised,
        "true_missing_count": true_missing,
        "true_missing_pct": round(true_missing / n * 100, 2) if n else 0.0,
    }
    if is_numeric:
        # stats WITHOUT the sentinels: pdays has mean ~962 with 999 included, which is meaningless
        col["numeric_stats"] = _numeric_stats(series[~series.isin(list(disguised["values"]))])
    if basic in ("categorical", "text"):
        col["categorical_stats"] = _categorical_stats(series)
    if semantic["semantic_type"] == "date":
        col["date_formats"] = _date_formats(series)
    return col


# ----------------------------------------------------------------- rules
def run_rules(df: pd.DataFrame, columns: list[dict]) -> list[dict]:
    """Profiling rule engine: flag suspicious columns, inconsistent formats, PII."""
    issues: list[dict] = []

    def add(col: str, rule: str, severity: str, detail: str) -> None:
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
        dis = c["disguised_missing"]
        # (if the dominant value is already known to be a missing code, the
        #  disguised_missing rule below explains it — no need for two issues)
        if (top and c["unique_count"] > 1 and top["pct"] > DOMINANT_VALUE_PCT
                and top["value"] not in dis["values"]):
            add(name, "dominant_value", "medium",
                f"{top['pct']}% of values are '{top['value']}' — almost constant, "
                f"or a placeholder/magic number")
        if dis["count"]:
            add(name, "disguised_missing", "high" if c["true_missing_pct"] >= 50 else "medium",
                f"{dis['count']} values ({dis['pct']}%) mean 'missing' but are not empty: "
                f"{dis['values']} — true missing {c['true_missing_pct']}% "
                f"(empty {c['missing_pct']}% + disguised {dis['pct']}%)")

        if sem in ("email", "phone", "postcode") and c["semantic"]["confidence"] < 0.99:
            bad = round((1 - c["semantic"]["confidence"]) * 100, 1)
            add(name, "invalid_format_values", "medium",
                f"~{bad}% of non-null values do not match the {sem} format")

    dup = int(df.duplicated().sum())
    if dup:
        add("<dataset>", "duplicate_rows", "high", f"{dup} exact duplicate rows")
    return issues


# --------------------------------------------------------------- dataset
def numeric_frame(df: pd.DataFrame, columns: list[dict]) -> pd.DataFrame:
    """Numeric columns as numbers, with disguised missing values (999, 'unknown') set to NaN.

    Used for the correlation matrix and the outlier chart, so they agree with numeric_stats.
    """
    out = {}
    for c in columns:
        if c.get("numeric_stats"):
            s = df[c["name"]].mask(df[c["name"]].isin(list(c["disguised_missing"]["values"])))
            out[c["name"]] = pd.to_numeric(s, errors="coerce")
    return pd.DataFrame(out, index=df.index)


def sample_rows(df: pd.DataFrame, n: int | None, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """Random sample of n rows (same rows every time thanks to the seed).

    Returns df unchanged if n is None or the file is already that small.
    """
    if n is None or n >= len(df):
        return df
    if n < 1:
        raise ValueError("sample size must be at least 1")
    return df.sample(n=n, random_state=seed)


def profile_dataset(df: pd.DataFrame, dataset_name: str = "dataset",
                    rows_in_file: int | None = None) -> dict:
    """Profile a DataFrame (read with dtype=str) and return the report dict.

    rows_in_file: total rows before sampling (only set when df is a sample).
    """
    start = time.perf_counter()
    columns = [profile_column(c, df[c]) for c in df.columns]

    num_df = numeric_frame(df, columns)      # disguised missing values -> NaN
    corr = num_df.corr().round(3) if num_df.shape[1] > 1 else pd.DataFrame()
    strong = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= 0.7:
                strong.append({"a": a, "b": b, "corr": float(v)})

    total_cells = df.size or 1
    disguised_cells = sum(c["disguised_missing"]["count"] for c in columns)
    real_missing = int(df.isna().sum().sum())
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": dataset_name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_cells": real_missing,
            "missing_cells_pct": round(real_missing / total_cells * 100, 2),
            "disguised_missing_cells": disguised_cells,
            "true_missing_cells_pct":
                round((real_missing + disguised_cells) / total_cells * 100, 2),
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
    report["quality_score"] = quality_score(report)
    report["cleaning_suggestions"] = cleaning_suggestions(columns, report["issues"])
    rows_in_file = rows_in_file or len(df)
    report["run"] = {
        # profiling time only (no file reading, no charts)
        "seconds": round(time.perf_counter() - start, 2),
        "rows_in_file": int(rows_in_file),
        "rows_profiled": int(len(df)),
        "sampled": bool(len(df) < rows_in_file),
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


def profile_file(path: str | Path, dataset_name: str | None = None,
                 sample: int | None = None) -> dict:
    """Read a CSV file and return its profiling report (dict).

    sample: profile only this many random rows (faster for big files).
    """
    path = Path(path)
    df = read_csv_raw(path)
    return profile_dataset(sample_rows(df, sample), dataset_name=dataset_name or path.name,
                           rows_in_file=len(df))


def save_report(report: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "profiling_report.json"
    path.write_text(json.dumps(report, indent=2, default=_py))
    return path
