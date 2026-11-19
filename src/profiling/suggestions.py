"""Turn profiling issues into cleaning suggestions — the hand-off to Module 2.

Each suggestion = {column, rule, severity, action, detail}
  action = a short code Module 2 can act on (e.g. "impute_median")
  detail = the same advice in plain English for people
"""
from __future__ import annotations

# Above this share of missing values, filling them in would mostly invent data,
# so we suggest a "was missing" flag column instead.
FLAG_INSTEAD_OF_IMPUTE_PCT = 50

ACTIONS = [
    "drop_column", "drop_duplicates", "impute_median", "impute_mode",
    "replace_with_nan_then_impute_median", "replace_with_nan_then_impute_mode",
    "replace_with_nan_and_add_flag", "convert_to_number", "convert_to_text",
    "standardise_case", "parse_dates_iso", "cap_outliers", "set_invalid_to_nan",
    "mask_pii", "review_values",
]


def _is_numeric(col: dict | None) -> bool:
    return bool(col and col.get("numeric_stats"))


def _suggest(issue: dict, col: dict | None) -> tuple[str, str]:
    """Return (action, detail) for one issue. col is None for dataset-level issues."""
    rule = issue["rule"]
    if rule == "duplicate_rows":
        return "drop_duplicates", "drop exact duplicate rows (keep the first copy)"
    if rule in ("mostly_missing", "constant_column"):
        return "drop_column", "drop the column: it carries almost no information"
    if rule == "missing_values":
        if _is_numeric(col):
            return "impute_median", "fill empty cells with the median (robust to outliers)"
        return "impute_mode", "fill empty cells with the most common value"
    if rule == "disguised_missing":
        values = list(col["disguised_missing"]["values"])
        if col["true_missing_pct"] >= FLAG_INSTEAD_OF_IMPUTE_PCT:
            return ("replace_with_nan_and_add_flag",
                    f"replace {values} with NaN and add a '{col['name']}_missing' 0/1 column; "
                    f"{col['true_missing_pct']}% missing is too much to fill in")
        if _is_numeric(col):
            return ("replace_with_nan_then_impute_median",
                    f"replace {values} with NaN, then fill with the median")
        return ("replace_with_nan_then_impute_mode",
                f"replace {values} with NaN, then fill with the most common value "
                f"(or keep '{values[0]}' as its own category)")
    if rule == "mixed_types":
        if col and col["inferred_type"] == "numeric":
            return "convert_to_number", "convert to numbers; text that is not a number becomes NaN"
        return "review_values", "check the values: numbers and text are mixed"
    if rule == "inconsistent_category_format":
        return "standardise_case", "strip spaces and lowercase, so 'HR' and 'hr ' become one value"
    if rule == "inconsistent_date_format":
        return "parse_dates_iso", "parse dates to YYYY-MM-DD"
    if rule == "outliers":
        return "cap_outliers", "cap values at the IQR fences (winsorise), or check them by hand"
    if rule == "invalid_format_values":
        return "set_invalid_to_nan", f"set values that do not match the {col['semantic']['semantic_type']} format to NaN"
    if rule == "potential_pii":
        return "mask_pii", "mask or hash this personal data before sharing"
    if rule == "id_stored_as_number":
        return "convert_to_text", "store the ID as text (keeps leading zeros)"
    # dominant_value and anything new: a person should look first
    return "review_values", f"review: {issue['detail']}"


def cleaning_suggestions(columns: list[dict], issues: list[dict]) -> list[dict]:
    """One suggestion per issue, most severe first (high, medium, info)."""
    by_name = {c["name"]: c for c in columns}
    order = {"high": 0, "medium": 1, "info": 2}
    out = []
    for issue in sorted(issues, key=lambda i: order[i["severity"]]):
        action, detail = _suggest(issue, by_name.get(issue["column"]))
        out.append({"column": issue["column"], "rule": issue["rule"],
                    "severity": issue["severity"], "action": action, "detail": detail})
    return out
