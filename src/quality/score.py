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

Flag columns (Sprint 6): Module 2 ADDS 0/1 columns such as salary_outlier,
possible_duplicate, is_anomaly or previously_contacted. They are never missing, so
counting them made completeness look better without fixing anything. Pass their
names as flag_columns: they are left out of all four dimensions and get their own
small "flag_columns" section instead.
"""
from __future__ import annotations

WEIGHTS = {"completeness": 0.25, "uniqueness": 0.25, "validity": 0.25, "consistency": 0.25}

VALIDITY_RULES = {"mixed_types", "invalid_format_values"}
CONSISTENCY_RULES = {"inconsistent_category_format", "inconsistent_date_format"}


def _clean_columns_pct(report: dict, rules: set[str], skip: set[str]) -> float:
    """% of columns (not counting the skipped ones) that have none of the given issues."""
    bad = {i["column"] for i in report["issues"] if i["rule"] in rules} - skip
    n_cols = report["dataset"]["columns"] - len(skip)
    if n_cols <= 0:
        return 100.0
    return (n_cols - len(bad)) / n_cols * 100


def _completeness(report: dict, skip: set[str]) -> float:
    """100 - true missing cells % over the columns that are not skipped."""
    if not skip:
        return 100 - report["dataset"]["true_missing_cells_pct"]
    cols = [c for c in report["columns"] if c["name"] not in skip]
    cells = report["dataset"]["rows"] * len(cols)
    missing = sum(c["true_missing_count"] for c in cols)
    return 100 - (missing / cells * 100 if cells else 0)


def quality_score(report: dict, flag_columns: list[str] | None = None) -> dict:
    """Return {"overall": 0-100, "dimensions": {...}, "weights": {...}} for a profiling report.

    flag_columns: columns added by cleaning (0/1 flags); scored apart, not in "overall".
    """
    d = report["dataset"]
    rows = d["rows"]
    skip = set(flag_columns or []) & {c["name"] for c in report.get("columns", [])}
    dims = {
        "completeness": _completeness(report, skip),
        "uniqueness": 100 - (d["duplicate_rows"] / rows * 100 if rows else 0),
        "validity": _clean_columns_pct(report, VALIDITY_RULES, skip),
        "consistency": _clean_columns_pct(report, CONSISTENCY_RULES, skip),
    }
    overall = sum(dims[k] * w for k, w in WEIGHTS.items())
    result = {
        "overall": round(overall, 2),
        "dimensions": {k: round(v, 2) for k, v in dims.items()},
        "weights": dict(WEIGHTS),
    }
    if skip:
        flags = [c for c in report["columns"] if c["name"] in skip]
        cells = rows * len(flags)
        filled = cells - sum(c["true_missing_count"] for c in flags)
        result["flag_columns"] = {
            "columns": sorted(skip),
            "completeness": round(filled / cells * 100, 2) if cells else 100.0,
            "note": "added by cleaning; not part of overall",
        }
    return result


def quality_delta(before: dict, after: dict) -> dict:
    """Improvement between two quality_score() results, e.g. {"overall": +4.2, ...}.

    For Module 2: delta = score(after cleaning) - score(before cleaning).
    """
    delta = {"overall": round(after["overall"] - before["overall"], 2)}
    for k in before["dimensions"]:
        delta[k] = round(after["dimensions"][k] - before["dimensions"][k], 2)
    return delta
