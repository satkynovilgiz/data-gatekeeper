"""Data quality score (0-100) computed from a profiling report.

Module 1 adds it to profiling_report.json (score BEFORE cleaning).
Module 2 will profile the cleaned data and call the same function (score AFTER
cleaning), then quality_delta() shows how much cleaning improved the data.

Four dimensions, each 0-100 (100 = no problems):

  completeness = 100 - true missing cells %          (empty + disguised, e.g. "unknown", 999)
  uniqueness   = 100 - duplicate rows %
  validity     = % of columns with no validity issue   (mixed_types, invalid_format_values)
  consistency  = % of columns with no consistency issue
                 (inconsistent_category_format, inconsistent_date_format)

  overall = weighted average of the four (equal weights, 25% each)

Completeness and uniqueness are counted per cell / per row because missing values
and duplicates are counted that way. Validity and consistency are counted per
column because these problems are fixed per column (one "lowercase" step fixes
the whole column).
"""
from __future__ import annotations

WEIGHTS = {"completeness": 0.25, "uniqueness": 0.25, "validity": 0.25, "consistency": 0.25}

VALIDITY_RULES = {"mixed_types", "invalid_format_values"}
CONSISTENCY_RULES = {"inconsistent_category_format", "inconsistent_date_format"}


def _clean_columns_pct(report: dict, rules: set[str]) -> float:
    """% of columns that have none of the given issues."""
    n_cols = report["dataset"]["columns"]
    if n_cols == 0:
        return 100.0
    bad = {i["column"] for i in report["issues"] if i["rule"] in rules}
    return (n_cols - len(bad)) / n_cols * 100


def quality_score(report: dict) -> dict:
    """Return {"overall": 0-100, "dimensions": {...}, "weights": {...}} for a profiling report."""
    d = report["dataset"]
    rows = d["rows"]
    dims = {
        "completeness": 100 - d["true_missing_cells_pct"],
        "uniqueness": 100 - (d["duplicate_rows"] / rows * 100 if rows else 0),
        "validity": _clean_columns_pct(report, VALIDITY_RULES),
        "consistency": _clean_columns_pct(report, CONSISTENCY_RULES),
    }
    overall = sum(dims[k] * w for k, w in WEIGHTS.items())
    return {
        "overall": round(overall, 2),
        "dimensions": {k: round(v, 2) for k, v in dims.items()},
        "weights": dict(WEIGHTS),
    }


def quality_delta(before: dict, after: dict) -> dict:
    """Improvement between two quality_score() results, e.g. {"overall": +4.2, ...}.

    For Module 2: delta = score(after cleaning) - score(before cleaning).
    """
    delta = {"overall": round(after["overall"] - before["overall"], 2)}
    for k in before["dimensions"]:
        delta[k] = round(after["dimensions"][k] - before["dimensions"][k], 2)
    return delta
