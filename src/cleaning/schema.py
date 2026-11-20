"""Schema inference — what each column SHOULD look like.

infer_schema(df, report) uses the Module 1 profiling report (types, meaning, missing
codes) plus the data itself (percentiles, categories) to describe every column:

    expected_type    numeric | categorical | date | text | id
    nullable         may the column have empty cells?
    numeric columns  typical_range = [p1, p99] of the data (for warnings)
                     valid_range   = business rule, e.g. age 0-120 (outside = impossible)
    categorical      allowed_values (after normalising case and spaces)
    date             format "YYYY-MM-DD"
    text             format, e.g. "email", with the regex Module 3 can check

The cleaner uses it to decide how to fix each column, and run_cleaning.py saves it as
schema.json so Module 3 can validate new data against it later.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.profiling.profiler import PLACEHOLDERS
from src.profiling.semantic import PATTERNS, name_words

SCHEMA_VERSION = "1.0"
DATE_FORMAT = "YYYY-MM-DD"

# Business rules: limits that real values can never break, matched on a WORD of the
# column name ("age" matches "age" and "customer_age", not "page_views").
# (low, high); None = no limit on that side.
BUSINESS_RULES: dict[str, tuple[float | None, float | None]] = {
    "age": (0, 120),
    "experience": (0, 70),
    "salary": (0, None),
    "income": (0, None),
    "price": (0, None),
    "amount": (0, None),
    "duration": (0, None),
    "pct": (0, 100),
    "percent": (0, 100),
}

# Text formats Module 3 can check with a regex (taken from Module 1's semantic detection)
TEXT_FORMATS = ("email", "phone", "postcode", "url", "ip_address")


def business_rule(column: str) -> tuple[str, list[float | None]] | None:
    """Return (rule word, [low, high]) if a business rule matches the column name, else None.

    Example: business_rule("age") -> ("age", [0, 120]).
    """
    for word in name_words(column):
        if word in BUSINESS_RULES:
            return word, list(BUSINESS_RULES[word])
    return None


def expected_type(col: dict) -> str:
    """Map a Module 1 column profile to one schema type: numeric|categorical|date|text|id."""
    sem = col["semantic"]["semantic_type"]
    basic = col["inferred_type"]
    if sem == "id":
        return "id"
    if basic == "datetime" or sem == "date":
        return "date"
    if basic == "numeric":
        return "numeric"
    if basic in ("categorical", "boolean"):
        return "categorical"
    return "text"


def category_key(value: str) -> str:
    """The form used to compare categories: no outer spaces, lowercase ('HR ' -> 'hr')."""
    return str(value).strip().lower()


def canonical_forms(series: pd.Series) -> dict[str, str]:
    """Map every written form of a category to ONE canonical form.

    1. Group values that are equal after strip + lowercase ("HR", "hr ", "Hr").
       The canonical form is the most common spelling in the group (ties: alphabetical),
       so "Engineering" wins over "ENGINEERING " and "engineering".
    2. Abbreviations: a rare value ending with "." whose stem starts exactly one other,
       more common category is merged into it ("Fin." -> "Finance").
    Returns {original value: canonical value}; values already canonical map to themselves.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return {}
    counts = s.str.strip().value_counts()            # stripped spelling -> count
    groups: dict[str, list[tuple[int, str]]] = {}
    for form, n in counts.items():
        groups.setdefault(category_key(form), []).append((int(n), form))
    # winner per group: highest count first, then alphabetical
    canon = {key: sorted(forms, key=lambda f: (-f[0], f[1]))[0][1] for key, forms in groups.items()}
    size = {key: sum(n for n, _ in forms) for key, forms in groups.items()}

    # abbreviation merge: "fin." -> the only other key that starts with "fin"
    for key in list(canon):
        if key.endswith(".") and len(key) > 2:
            stem = key[:-1]
            targets = [k for k in canon if k != key and k.startswith(stem) and size[k] > size[key]]
            if len(targets) == 1:
                canon[key] = canon[targets[0]]

    return {orig: canon[category_key(orig)] for orig in s.unique()}


def _column_schema(col: dict, series: pd.Series) -> dict:
    """Schema for one column from its profile (col) and its values (series)."""
    etype = expected_type(col)
    missing_codes = list(col["disguised_missing"]["values"])
    out: dict = {
        "name": col["name"],
        "expected_type": etype,
        "semantic_type": col["semantic"]["semantic_type"],
        "nullable": col["true_missing_count"] > 0,
        "missing_codes": missing_codes,  # values meaning "missing", e.g. "unknown", "999"
    }
    # real values only: no empty cells, no missing codes, no placeholder text
    real = series.dropna()
    real = real[~real.isin(missing_codes) & ~real.astype(str).map(category_key).isin(PLACEHOLDERS)]

    if etype == "numeric":
        nums = pd.to_numeric(real, errors="coerce").dropna()
        rule = business_rule(col["name"])
        valid = rule[1] if rule else [None, None]
        lo, hi = valid
        # describe only the possible values: age 230 must not widen the ranges
        inside = nums
        if lo is not None:
            inside = inside[inside >= lo]
        if hi is not None:
            inside = inside[inside <= hi]
        out["is_integer"] = bool(not inside.empty and (inside % 1 == 0).all())
        out["observed_range"] = ([float(inside.min()), float(inside.max())]
                                 if not inside.empty else None)
        out["typical_range"] = ([round(float(inside.quantile(0.01)), 4),
                                 round(float(inside.quantile(0.99)), 4)]
                                if not inside.empty else None)
        out["valid_range"] = valid
        out["range_source"] = f"business_rule:{rule[0]}" if rule else None
    elif etype == "categorical":
        out["allowed_values"] = sorted(set(canonical_forms(real).values()))
    elif etype == "date":
        out["format"] = DATE_FORMAT
    elif etype == "text" and out["semantic_type"] in TEXT_FORMATS:
        out["format"] = out["semantic_type"]
        out["pattern"] = PATTERNS[out["semantic_type"]].pattern
    if etype == "id":
        out["unique"] = bool(real.is_unique)
    return out


def infer_schema(df: pd.DataFrame, report: dict) -> dict:
    """Infer the expected schema of every column.

    Input:  df = the data (read as text), report = its Module 1 profiling report.
    Output: {"schema_version", "generated_at", "dataset", "columns": {name: column schema}}.
    Raises ValueError if the report does not describe the same columns as df.
    """
    profiles = {c["name"]: c for c in report["columns"]}
    missing = [c for c in df.columns if c not in profiles]
    if missing:
        raise ValueError(f"profiling report has no entry for columns {missing} — "
                         f"was it made from a different file?")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": report["dataset"]["name"],
        "columns": {c: _column_schema(profiles[c], df[c]) for c in df.columns},
    }


def save_schema(schema: dict, out_dir: str | Path) -> Path:
    """Write the schema as pretty JSON to <out_dir>/schema.json and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "schema.json"
    path.write_text(json.dumps(schema, indent=2))
    return path

