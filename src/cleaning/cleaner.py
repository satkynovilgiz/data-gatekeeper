"""Module 2 — core cleaning pipeline.

    clean_dataset(df, report, config)          -> (cleaned df, cleaning log dict)
    clean_file(input, out_dir, report=None)    -> cleaning log dict, and writes
        <out_dir>/cleaned_data.csv, cleaning_log.json, schema.json

The profiling report (Module 1) says WHAT is wrong (cleaning_suggestions); the schema
(schema.py) says what each column SHOULD be, so the steps know HOW to fix it.
The steps always run in this order (each one makes the next one easier):

     1. trim_whitespace           " HR " -> "HR"                      (all text)
     2. special_codes             pdays 999 -> NaN + previously_contacted yes/no
     3. disguised_missing_to_nan  "unknown", "?", "n/a", 999 ... -> NaN
     4. normalise_categories      "hr", "HR", "Fin." -> "HR", "Finance"
     5. convert_to_number         "42" -> 42, "five" -> NaN (flagged)
     6. parse_dates               "20/09/2023" -> "2023-09-20"
     7. invalid_format_to_nan     "not-an-email" -> NaN
     8. impossible_value_to_nan   age 230 -> NaN (outside the business rule 0-120)
     9. drop_duplicates           exact duplicate rows (keep the first)
    10. fuzzy_duplicates          "Jon Smith" ~ "John Smith" + same fields: flag (or merge)
    11. handle_outliers           IQR outliers: flag / cap / NaN; Isolation Forest: is_anomaly
    12. impute                    NaN -> median / mode, or an ML imputer (knn, regression,
                                  iterative), per column

Missing codes and bad values become NaN first (steps 2-8), duplicates go next (so
they do not bias the median), outliers are handled, and only then are the gaps
filled (step 12).
Rows are never deleted except exact duplicates, and the input file is only read.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.profiling.profiler import PLACEHOLDERS, profile_dataset, read_csv_raw, save_report
from src.profiling.semantic import PII_TYPES
from src.quality.score import quality_delta, quality_score

from .evaluation import evaluate_imputers, save_evaluation
from .fuzzy import FUZZY_THRESHOLD, MIN_AGREEMENT, find_fuzzy_duplicates
from .imputers import METHODS, describe, impute_columns, is_numeric
from .outliers import ANOMALY_SHARE, iqr_outliers, isolation_forest_rows, main_reason
from .schema import canonical_forms, category_key, infer_schema, save_schema

LOG_VERSION = "1.0"

DEFAULT_CONFIG: dict[str, Any] = {
    # imputation per column: "median" | "mean" | "mode" | "knn" | "regression" |
    # "iterative" | "none". Columns not listed use the defaults below by schema type
    # ("median" on a category means its mode). Dates, text, ids and personal data are
    # never filled in: an invented email or hire date would be fake data.
    "impute": {},
    # Default chosen by the Sprint 5 evaluation (hide 10% of known values, refill them):
    # Bank Marketing  NRMSE iterative 0.50 vs median 1.08, accuracy 65.8% vs 51.8%;
    # HR sample (columns unrelated by design) iterative 0.77 vs median 0.74, accuracy equal.
    # Real data usually has related columns, so iterative wins where it matters.
    "numeric_strategy": "iterative",
    "categorical_strategy": "iterative",
    # Columns the ML imputers must NOT look at. "y" is the Bank Marketing label (did the
    # client subscribe?): using it to fill other columns would leak the answer into the
    # inputs of a later model (target leakage).
    "exclude_features": ["y"],
    # Codes that carry a MEANING, so they are not missing values to fill in.
    # UCI Bank Marketing: pdays = days since the client was last contacted, and
    # 999 = "never contacted before". Imputing a number there would invent contacts.
    # Instead: new column previously_contacted (no when pdays was 999, else yes),
    # and pdays becomes NaN only for the never-contacted rows. pdays is not imputed.
    "special_codes": {
        "pdays": {"code": 999, "flag_column": "previously_contacted",
                  "code_means": "no", "otherwise": "yes"},
    },
    # Categorical columns where "unknown" carries information, so it is NOT a gap to fill.
    # UCI Bank Marketing: default = "has credit in default?"; 20.9% answer "unknown".
    # A client who does not say may be a risk signal, and imputing the mode would turn
    # all of them into "no" (it even made 2 different clients identical in Sprint 4).
    # In these columns every missing code (and empty cell) becomes the category "unknown".
    "keep_as_category": ["default"],
    # override or add business rules, e.g. {"age": [18, 100]}
    "valid_ranges": {},
    # ambiguous dates like 05/06/2020: True = 5 June (day first), False = May 6
    # (only used when no value in the column shows the order, e.g. 20/09/2023)
    "dayfirst": True,
    # Outliers (IQR, per column): "flag" (default: new <column>_outlier 0/1 column, value
    # kept), "cap" (clip to the IQR fences), "nan" (set to NaN, then imputed) or "none".
    # Per column in "outliers", e.g. {"salary": "cap"}. Rows are never deleted.
    "outlier_action": "flag",
    "outliers": {},
    # Isolation Forest marks this share of the most unusual rows in is_anomaly (0 = off)
    "anomaly_share": ANOMALY_SHARE,
    # Fuzzy duplicates: rows whose names are similar (rapidfuzz score >= threshold, 0-100)
    # AND whose other fields mostly agree. Default: flag both rows (possible_duplicate = 1).
    # Pairs with a name score >= fuzzy_merge_threshold are merged (later row removed);
    # None = never merge. Name columns are found automatically (Module 1 person_name).
    "fuzzy_threshold": FUZZY_THRESHOLD,
    "fuzzy_min_agreement": MIN_AGREEMENT,
    "fuzzy_merge_threshold": None,
    "fuzzy_key_columns": None,
    # hide 10% of known values and score every imputer on them (slower: off by default)
    "evaluate_imputers": False,
    "max_examples": 3,
    # New rows only (prediction mode, Sprint 9). Codes people use for "no value"; the
    # profiler's numeric sentinels found in a train column are added for that column.
    "common_sentinels": [999, 9999, -1, -999],
    # Sprint 10: a code becomes NaN on new rows when its share there jumps:
    #     new_share > max(k x train_share, train_share + min_abs)
    # Chosen once, before any evaluation, and never tuned on it:
    #   k = 5        "clearly more common than before", not a random wobble. The number of
    #                rows with a code is roughly Poisson, so a share 5x its usual level
    #                is very unlikely by chance once the code has a few dozen rows.
    #   min_abs      for rare codes k alone is not enough: duration = 999 is 2 of 33k train
    #   = 0.005      rows, so 2-3 of 8k new rows would already be 5x. Also requiring
    #                +0.5 percentage points (40 rows in an 8k batch) stops that. A very
    #                common code (pdays 999 = 96%) never triggers: 5 x 96% > 100%.
    "sentinel_rule": {"k": 5.0, "min_abs": 0.005},
    # What the train rows looked like (learn_from_train()); None when cleaning one file.
    "learned": None,
}

IMPUTE_STRATEGIES = {*METHODS, "none"}
# a feature column may be at most this share empty (pdays is 96% empty: no use as a feature)
MAX_FEATURE_MISSING = 0.5

# Which step carries out each Module 1 action. Actions not listed here are skipped
# with the reason in SKIPPED_ACTIONS, so the log accounts for every suggestion.
ACTION_STEPS = {
    "drop_duplicates": ["drop_duplicates"],
    "impute_median": ["impute"],
    "impute_mode": ["impute"],
    "replace_with_nan_then_impute_median": ["disguised_missing_to_nan", "impute"],
    "replace_with_nan_then_impute_mode": ["disguised_missing_to_nan", "impute"],
    "replace_with_nan_and_add_flag": ["disguised_missing_to_nan", "special_codes"],
    "convert_to_number": ["convert_to_number"],
    "standardise_case": ["normalise_categories"],
    "parse_dates_iso": ["parse_dates"],
    "set_invalid_to_nan": ["invalid_format_to_nan"],
    "cap_outliers": ["outlier_flag", "outlier_cap", "outlier_to_nan"],
}
OUTLIER_ACTIONS = {"flag", "cap", "nan", "none"}
SKIPPED_ACTIONS = {
    "mask_pii": "planned for Sprint 6 (PII protection)",
    "drop_column": "not automatic: dropping a column needs a person's decision",
    "review_values": "needs a person to look at the values",
    "convert_to_text": "nothing to do: the cleaner reads and writes IDs as text",
}


# ------------------------------------------------------------------ log
def _jsonable(v: Any) -> Any:
    """Turn a cell value into plain JSON: NaN -> None, numpy numbers -> Python numbers."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


@dataclass
class CleaningRun:
    """Everything one cleaning run shares: the data, the plan and the log of changes."""

    df: pd.DataFrame
    schema: dict
    config: dict
    suggestions: list[dict]
    steps: list[dict] = field(default_factory=list)
    no_impute: set[str] = field(default_factory=set)      # pdays, flag columns, ...
    added_columns: list[str] = field(default_factory=list)
    kept_as_category: dict[str, int] = field(default_factory=dict)  # column -> "unknown" cells
    outlier_stats: dict[str, int] = field(default_factory=dict)
    fuzzy: dict[str, Any] = field(default_factory=dict)
    sentinel_decisions: list[dict] = field(default_factory=list)   # step 8b, per code

    def cols(self, etype: str) -> list[str]:
        """Names of the columns whose schema expected_type is etype."""
        return [c for c, s in self.schema["columns"].items()
                if s["expected_type"] == etype and c in self.df.columns]

    def suggested(self, action: str) -> set[str]:
        """Columns for which Module 1 suggested this action."""
        return {s["column"] for s in self.suggestions if s["action"] == action}

    def record(self, step: str, column: str, rule: str, changed: pd.Series,
               before: pd.Series, after: pd.Series) -> None:
        """Log one step on one column: how many cells changed + a few before -> after examples.

        changed is a True/False mask over the rows; nothing is logged if no cell changed.
        "row" in an example is the row number in the input file (0 = first data row).
        """
        n = int(changed.sum())
        if n == 0:
            return
        rows = changed[changed].index[: self.config["max_examples"]]
        self.steps.append({
            "step": step, "column": column, "rule": rule,
            "cells_changed": n, "rows_changed": n,
            "examples": [{"row": int(r), "before": _jsonable(before.loc[r]),
                          "after": _jsonable(after.loc[r])} for r in rows],
        })


# ---------------------------------------------------------------- steps
def trim_whitespace(run: CleaningRun) -> None:
    """Step 1: remove spaces at the start and end of every text value (" HR " -> "HR")."""
    for c in run.df.columns:
        s = run.df[c]
        new = s.str.strip()
        run.record("trim_whitespace", c, "strip leading/trailing spaces",
                   s.notna() & (new != s), s, new)
        run.df[c] = new


def special_codes(run: CleaningRun) -> None:
    """Step 2: codes that MEAN something (pdays 999 = never contacted) become a flag column.

    The code cells become NaN and the column is added to no_impute, so nothing is invented.
    """
    for c, rule in run.config["special_codes"].items():
        if c not in run.df.columns:
            continue
        s = run.df[c]
        is_code = pd.to_numeric(s, errors="coerce") == rule["code"]     # "999" and "999.0"
        if not is_code.any():
            continue
        flag = pd.Series(np.where(is_code, rule["code_means"], rule["otherwise"]),
                         index=s.index).where(s.notna())
        new = s.mask(is_code)
        name = rule["flag_column"]
        run.df.insert(run.df.columns.get_loc(c) + 1, name, flag)
        run.added_columns.append(name)
        run.no_impute.update({c, name})
        run.record("special_codes", name,
                   f"new column: '{rule['code_means']}' when {c} = {rule['code']}, "
                   f"else '{rule['otherwise']}'", flag.notna(), s, flag)
        run.record("special_codes", c,
                   f"{rule['code']} means '{name} = {rule['code_means']}', not a number: "
                   f"set to NaN and NOT imputed", is_code, s, new)
        run.df[c] = new


def disguised_missing_to_nan(run: CleaningRun) -> None:
    """Step 3: values that mean "missing" ("unknown", "?", "n/a", 999 ...) become NaN.

    Uses the exact codes Module 1 found (schema missing_codes) plus the placeholder
    list for any column. Columns with >= 50% missing get a <column>_missing 0/1 flag
    instead of being filled in (Module 1 action replace_with_nan_and_add_flag).
    Columns in config keep_as_category keep "unknown" as a real category instead.
    """
    flag_cols = run.suggested("replace_with_nan_and_add_flag")
    for c in list(run.df.columns):
        if c in run.added_columns:
            continue
        s = run.df[c]
        codes = run.schema["columns"].get(c, {}).get("missing_codes", [])
        is_missing = s.notna() & (s.isin(codes) | s.map(category_key).isin(PLACEHOLDERS))
        if c in run.cols("categorical") and c in run.config["keep_as_category"]:
            _keep_unknown(run, c, is_missing | s.isna())
            continue
        new = s.mask(is_missing)
        run.record("disguised_missing_to_nan", c,
                   f"missing codes {sorted(set(s[is_missing]))} -> NaN", is_missing, s, new)
        run.df[c] = new
        if c in flag_cols and c not in run.no_impute:
            name = f"{c}_missing"
            flag = new.isna().astype(int)
            run.df.insert(run.df.columns.get_loc(c) + 1, name, flag)
            run.added_columns.append(name)
            run.no_impute.update({c, name})
            run.record("add_missing_flag", name, f"1 when {c} was missing, else 0 "
                       f"(too much missing to fill in)", flag == 1, s, flag)


def _keep_unknown(run: CleaningRun, column: str, is_missing: pd.Series) -> None:
    """Missing codes and empty cells in a keep_as_category column become one category "unknown".

    The column is added to no_impute: "unknown" is an answer, not a gap.
    """
    s = run.df[column]
    new = s.mask(is_missing, "unknown")
    run.record("keep_as_category", column, "missing codes and empty cells -> category "
               "'unknown' (kept, not imputed)", is_missing & (s != "unknown"), s, new)
    run.df[column] = new
    run.no_impute.add(column)
    run.kept_as_category[column] = int(is_missing.sum())


def normalise_categories(run: CleaningRun) -> None:
    """Step 4: one spelling per category: case/space variants and "Fin."-style abbreviations."""
    for c in run.cols("categorical"):
        s = run.df[c]
        new = s.map(canonical_forms(s))              # NaN stays NaN
        run.record("normalise_categories", c, "merge case, space and abbreviation variants",
                   s.notna() & (new != s), s, new)
        run.df[c] = new


def convert_to_number(run: CleaningRun) -> None:
    """Step 5: numeric columns become real numbers; text that is not a number becomes NaN."""
    for c in run.cols("numeric"):
        s = run.df[c]
        nums = pd.to_numeric(s, errors="coerce")
        bad = s.notna() & nums.isna()
        run.record("convert_to_number", c, "not a number -> NaN", bad, s, nums)
        run.df[c] = nums


def _dayfirst(values: pd.Series, default: bool) -> bool:
    """Decide the order of dd/mm vs mm/dd dates from the values themselves.

    If any first part is > 12 it must be the day (20/09/2023); if any second part is
    > 12 it must be the day (09/20/2023). If no value tells, use the config default.
    """
    parts = values.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}$").astype(float)
    if (parts[0] > 12).any():
        return True
    if (parts[1] > 12).any():
        return False
    return default


def parse_dates(run: CleaningRun) -> None:
    """Step 6: every date becomes YYYY-MM-DD; dates that cannot be read become NaN (flagged)."""
    for c in run.cols("date"):
        s = run.df[c]
        iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        other = s.notna() & iso.isna()
        if other.any():
            first = _dayfirst(s[other], run.config["dayfirst"])
            iso[other] = pd.to_datetime(s[other], format="mixed", dayfirst=first,
                                        errors="coerce")
        new = iso.dt.strftime("%Y-%m-%d").where(iso.notna())
        run.record("parse_dates", c, "reformat to YYYY-MM-DD",
                   other & new.notna(), s, new)
        run.record("parse_dates", c, "not a date -> NaN", s.notna() & new.isna(), s, new)
        run.df[c] = new


def invalid_format_to_nan(run: CleaningRun) -> None:
    """Step 7: text that breaks its format (e.g. "not-an-email" in an email column) -> NaN."""
    for c in run.cols("text"):
        pattern = run.schema["columns"][c].get("pattern")
        if not pattern:
            continue
        s = run.df[c]
        bad = s.notna() & ~s.astype(str).str.match(pattern)
        new = s.mask(bad)
        run.record("invalid_format_to_nan", c,
                   f"does not match the {run.schema['columns'][c]['format']} format -> NaN",
                   bad, s, new)
        run.df[c] = new


def impossible_value_to_nan(run: CleaningRun) -> None:
    """Step 8: numbers outside the valid range (business rule) -> NaN. The row is kept."""
    for c in run.cols("numeric"):
        col = run.schema["columns"][c]
        lo, hi = run.config["valid_ranges"].get(c, col.get("valid_range", [None, None]))
        if lo is None and hi is None:
            continue
        s = run.df[c]
        bad = ((s < lo) if lo is not None else False) | ((s > hi) if hi is not None else False)
        new = s.mask(bad)
        source = col.get("range_source") if c not in run.config["valid_ranges"] else "config"
        run.record("impossible_value_to_nan", c,
                   f"outside valid range [{lo}, {hi}] ({source}) -> NaN", bad, s, new)
        run.df[c] = new


def drop_duplicates(run: CleaningRun) -> None:
    """Step 9: remove rows that are exact copies of an earlier row (after steps 1-8).

    Examples show which earlier row each removed row was a copy of.
    """
    dup = run.df.duplicated(keep="first")
    n = int(dup.sum())
    if n == 0:
        return
    hashes = pd.util.hash_pandas_object(run.df, index=False)
    first_row = {h: i for i, h in hashes[~hashes.duplicated()].items()}
    examples = [{"row": int(r), "duplicate_of_row": int(first_row[hashes.loc[r]])}
                for r in dup[dup].index[: run.config["max_examples"]]]
    run.steps.append({"step": "drop_duplicates", "column": "<dataset>",
                      "rule": "exact duplicate row (keep the first copy)",
                      "cells_changed": 0, "rows_changed": n, "examples": examples})
    run.df = run.df[~dup]


def _strategy(run: CleaningRun, column: str) -> str:
    """The imputation strategy for a column: from config, else by schema type.

    Never for: pdays-like and flag columns (no_impute), dates, text, ids, personal data.
    """
    col = run.schema["columns"].get(column, {})
    if column in run.no_impute or col.get("semantic_type") in PII_TYPES:
        return "none"
    if column in run.config["impute"]:
        return run.config["impute"][column]
    if col.get("expected_type") == "numeric":
        return run.config["numeric_strategy"]
    if col.get("expected_type") == "categorical":
        return run.config["categorical_strategy"]
    return "none"


def feature_columns(run: CleaningRun) -> list[str]:
    """Columns the ML imputers may learn from: numbers and categories only.

    Not: ids, text, dates, personal data, excluded columns (the label "y"), or columns
    that are mostly empty.
    """
    out = []
    for c in run.cols("numeric") + run.cols("categorical"):
        col = run.schema["columns"][c]
        if (col["semantic_type"] in PII_TYPES or c in run.config["exclude_features"]
                or run.df[c].isna().mean() > MAX_FEATURE_MISSING):
            continue
        out.append(c)
    return out


def learning_frame(run: CleaningRun) -> pd.DataFrame:
    """A copy of the data where FLAGGED outliers are hidden (NaN), for the imputers to learn from.

    With the default action "flag", salary 9,999,999 stays in the output. If the imputers
    learned from it, MICE filled missing HR salaries with ~187,000 instead of ~80,000.
    So they learn from typical values only; the output keeps the flagged values unchanged.
    """
    df = run.df.copy()
    for c in outlier_columns(run):
        flag = f"{c}_outlier"
        if flag in df.columns:
            df[c] = df[c].mask(df[flag] == 1)
    return df


def impute(run: CleaningRun) -> None:
    """Step 12: fill the remaining NaN with each column's method (median/mode or ML).

    Columns that share an ML method are filled together in one call, so KNN and
    iterative see all of them at once.
    """
    by_method: dict[str, list[str]] = {}
    for c in run.df.columns:
        strategy = _strategy(run, c)
        if strategy != "none" and run.df[c].isna().any() and run.df[c].notna().any():
            by_method.setdefault(strategy, []).append(c)
    features = feature_columns(run)
    learn = learning_frame(run)
    for method, targets in by_method.items():
        filled = impute_columns(learn, targets, features, method)
        for c in targets:
            s = run.df[c]
            gaps = s.isna()
            new = s.copy() if not is_numeric(s) else s.astype(float)
            new[gaps] = filled.loc[gaps, c]                 # only the real gaps are filled
            if method in ("median", "mean", "mode"):
                # one value for every gap: show it, e.g. "median = 42" or "mode = Rome"
                label = method if is_numeric(s) or method == "mode" else "mode"
                value = _jsonable(new[gaps].iloc[0])
                if isinstance(value, float) and value.is_integer():
                    value = int(value)                  # "median = 35", not "35.0"
                rule = f"{label} = {value}"
            else:
                rule = f"{describe(method)} using {len(features)} feature columns"
            run.record("impute", c, rule, gaps, s, new)
            run.df[c] = new


def fuzzy_duplicates(run: CleaningRun) -> None:
    """Step 10: near-duplicate rows (similar name + mostly equal fields): flag or merge.

    Runs before imputation: filled-in values would make different people look alike.
    """
    cfg = run.config
    keys = cfg["fuzzy_key_columns"] or [
        c for c, s in run.schema["columns"].items()
        if s["semantic_type"] == "person_name" and c in run.df.columns]
    if not keys:
        run.fuzzy = {"skipped": "no name column to compare (set fuzzy_key_columns)"}
        return
    compare = [c for c in run.df.columns if c not in keys and c not in run.added_columns
               and run.schema["columns"].get(c, {}).get("expected_type") != "id"]
    pairs = find_fuzzy_duplicates(run.df, keys, compare, cfg["fuzzy_threshold"],
                                  cfg["fuzzy_min_agreement"])
    merge_at = cfg["fuzzy_merge_threshold"]
    to_merge = [p for p in pairs if merge_at is not None and p["name_similarity"] >= merge_at]
    to_flag = [p for p in pairs if p not in to_merge]
    n_ex = cfg["max_examples"]

    def example(p: dict) -> dict:
        return {"row": p["row_b"], "similar_to_row": p["row_a"],
                "names": f"{p['name_a']} ~ {p['name_b']}", "name_similarity":
                p["name_similarity"], "fields_equal": f"{p['fields_equal']}/{p['fields_compared']}"}

    removed = sorted({p["row_b"] for p in to_merge} - {p["row_a"] for p in to_merge})
    if removed:
        run.steps.append({"step": "fuzzy_merge", "column": "<dataset>",
                          "rule": f"name similarity >= {merge_at} and >= "
                                  f"{cfg['fuzzy_min_agreement']:.0%} of fields equal: "
                                  f"later row removed",
                          "cells_changed": 0, "rows_changed": len(removed),
                          "examples": [example(p) for p in to_merge[:n_ex]]})
        run.df = run.df.drop(index=removed)
    flag_rows = sorted({r for p in to_flag for r in (p["row_a"], p["row_b"])
                        if r in run.df.index})
    if flag_rows:
        run.df["possible_duplicate"] = run.df.index.isin(flag_rows).astype(int)
        run.added_columns.append("possible_duplicate")
        run.no_impute.add("possible_duplicate")
        run.steps.append({"step": "fuzzy_flag", "column": "possible_duplicate",
                          "rule": f"name similarity >= {cfg['fuzzy_threshold']} and >= "
                                  f"{cfg['fuzzy_min_agreement']:.0%} of fields equal: "
                                  f"both rows flagged, nothing removed",
                          "cells_changed": len(flag_rows), "rows_changed": len(flag_rows),
                          "examples": [example(p) for p in to_flag[:n_ex]]})
    run.fuzzy = {"key_columns": keys, "compared_columns": compare,
                 "pairs_found": len(pairs), "rows_flagged": len(flag_rows),
                 "rows_merged": len(removed), "pairs": pairs[:100]}


def outlier_columns(run: CleaningRun) -> list[str]:
    """Numeric columns checked for outliers: not added flags, not special-code columns (pdays)."""
    return [c for c in run.cols("numeric")
            if c not in run.added_columns and c not in run.no_impute]


def handle_outliers(run: CleaningRun) -> None:
    """Step 11: IQR outliers per column (flag / cap / nan) + Isolation Forest per row (flag).

    Both detectors look at the data BEFORE any outlier is changed, so they are independent.
    "nan" leaves a gap that the impute step fills; no row is ever deleted.
    """
    columns = outlier_columns(run)
    masks = {c: iqr_outliers(run.df[c]) for c in columns}
    anomalies = isolation_forest_rows(run.df, columns, run.config["anomaly_share"])
    any_iqr = pd.Series(False, index=run.df.index)

    for c, (is_out, fences) in masks.items():
        action = run.config["outliers"].get(c, run.config["outlier_action"])
        if fences is None or not is_out.any() or action == "none":
            continue
        any_iqr |= is_out
        lo, hi = fences
        s = run.df[c]
        where = f"outside IQR fences [{lo:g}, {hi:g}]"
        if action == "flag":
            name = f"{c}_outlier"
            flag = is_out.astype(int)
            run.df.insert(run.df.columns.get_loc(c) + 1, name, flag)
            run.added_columns.append(name)
            run.no_impute.add(name)
            run.record("outlier_flag", c, f"{where}: value kept, {name} = 1", is_out, s, flag)
        elif action == "cap":
            if run.schema["columns"][c].get("is_integer"):
                lo, hi = np.ceil(lo), np.floor(hi)          # whole-number columns stay whole
            new = s.clip(lo, hi)
            run.record("outlier_cap", c, f"{where}: capped to [{lo:g}, {hi:g}]", is_out, s, new)
            run.df[c] = new
        else:
            new = s.mask(is_out)
            run.record("outlier_to_nan", c, f"{where}: set to NaN, imputed later",
                       is_out, s, new)
            run.df[c] = new

    if anomalies.any():
        run.df["is_anomaly"] = anomalies.astype(int)
        run.added_columns.append("is_anomaly")
        run.no_impute.add("is_anomaly")
        reasons = pd.Series("", index=run.df.index)
        for r in anomalies[anomalies].index[: run.config["max_examples"]]:
            reasons[r] = main_reason(run.df, columns, r)
        run.record("isolation_forest", "is_anomaly",
                   f"Isolation Forest: {run.config['anomaly_share']:.0%} most unusual rows "
                   f"-> is_anomaly = 1 (values kept); before = the most unusual value",
                   anomalies, reasons, run.df["is_anomaly"])
    run.outlier_stats = {
        "iqr_cells": int(sum(m.sum() for m, f in masks.values() if f is not None)),
        "iqr_rows": int(any_iqr.sum()),
        "isolation_forest_rows": int(anomalies.sum()),
        "rows_found_by_both": int((any_iqr & anomalies).sum()),
    }


def learn_from_train(df: pd.DataFrame, numeric: list[str], sentinels: list[float],
                     ranges: dict[str, list[float]] | None = None,
                     extra_codes: dict[str, list[float]] | None = None) -> dict:
    """What new rows are compared with, per numeric column of the (cleaned) train rows:
    its [min, max], and the share of train rows holding each candidate sentinel.

    sentinels: codes checked in every column (config common_sentinels).
    extra_codes: {column: [codes]} the profiler flagged as numeric sentinels in that column.
    ranges: limits that replace the learned ones (e.g. a contract's min / max).
    """
    learned: dict[str, dict] = {"ranges": {}, "sentinel_share": {}}
    rows = len(df)
    for c in numeric:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        known = s.dropna()
        if known.empty:
            continue
        learned["ranges"][c] = [float(known.min()), float(known.max())]
        codes = list(dict.fromkeys([*sentinels, *(extra_codes or {}).get(c, [])]))
        learned["sentinel_share"][c] = {float(v): float((s == v).sum() / rows) if rows else 0.0
                                        for v in codes}
    for c, (lo, hi) in (ranges or {}).items():
        if c in learned["ranges"]:
            learned["ranges"][c] = [float(lo), float(hi)]
    return learned


def sentinel_jumped(train_share: float, new_share: float, k: float, min_abs: float) -> bool:
    """The Sprint 10 rule: is a code much more common in the new rows than in train?"""
    return new_share > max(k * train_share, train_share + min_abs)


def out_of_range_to_missing(run: CleaningRun) -> None:
    """Step 8b (new rows only): values that do not fit what the train rows taught -> NaN.

    Found by the Sprint 8 robustness test: a 999 in euribor3m (train range 0.6-5.0) stayed
    a number, and the iterative imputer then filled other gaps of that row from it. So:
      - a number outside the train range (or the contract's min / max) -> NaN    (Sprint 9)
      - a sentinel code (999, 9999, -1, -999, + the profiler's codes for that column)
        whose share in the new rows JUMPED compared with train -> NaN            (Sprint 10)
        Sprint 9 asked "did train ever have it?"; duration has two real 999s in 41k rows,
        so injected 999s passed. Now 2 rows in train vs 20% of new rows is a jump.
    Every code that occurs is logged in run.sentinel_decisions (train share, new share,
    decision). Does nothing when no train rows were learned (config learned=None).
    Business rules (age 0-120) were already applied by step 8.
    """
    learned = run.config.get("learned")
    if not learned:
        return
    rule = run.config["sentinel_rule"]
    rows = len(run.df)
    for c in run.cols("numeric"):
        if c not in learned["ranges"]:
            continue
        s = pd.to_numeric(run.df[c], errors="coerce")
        lo, hi = learned["ranges"][c]
        outside = s.notna() & ((s < lo) | (s > hi))
        sentinel = pd.Series(False, index=s.index)
        jumped = []
        for code, train_share in learned.get("sentinel_share", {}).get(c, {}).items():
            hit = s == code
            n = int(hit.sum())
            if n == 0:
                continue
            new_share = n / rows if rows else 0.0
            if bool(outside[hit].all()):
                decision = "missing: outside the train range"
            elif sentinel_jumped(train_share, new_share, rule["k"], rule["min_abs"]):
                decision = "missing: share jumped"
                sentinel |= hit & ~outside
                jumped.append(code)
            else:
                decision = "kept: share as in train"
            run.sentinel_decisions.append({
                "column": c, "code": code, "train_share": round(train_share, 6),
                "new_share": round(new_share, 6), "rows": n, "decision": decision})
        new = s.mask(outside | sentinel)
        run.record("out_of_range_to_missing", c, f"outside the train range [{lo:g}, {hi:g}] -> NaN",
                   outside, run.df[c], new)
        if jumped:
            run.record("out_of_range_to_missing", c, "sentinel share jumped vs train "
                       f"({', '.join(f'{v:g}' for v in jumped)}) -> NaN", sentinel, run.df[c], new)
        run.df[c] = new


# Steps 1-8b change each row on its own (no statistics learned across rows), so they can
# be applied to new rows with a schema learned elsewhere (Sprint 8 impact, per CV fold).
ROW_STEPS = [trim_whitespace, special_codes, disguised_missing_to_nan, normalise_categories,
             convert_to_number, parse_dates, invalid_format_to_nan, impossible_value_to_nan,
             out_of_range_to_missing]
STEPS = [*ROW_STEPS, drop_duplicates, fuzzy_duplicates, handle_outliers, impute]


# -------------------------------------------------------------- pipeline
def _whole_numbers_as_int(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns whose values are all whole become integers (42.0 -> 42; NaN allowed)."""
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s) and s.notna().any() and (s.dropna() % 1 == 0).all():
            df[c] = s.astype("Int64")
    return df


def _suggestion_status(run: CleaningRun) -> list[dict]:
    """For every Module 1 suggestion: done (by which steps) or skipped (and why)."""
    done = {(s["step"], s["column"]) for s in run.steps}
    out = []
    for sug in run.suggestions:
        action, col = sug["action"], sug["column"]
        steps = [st for st in ACTION_STEPS.get(action, []) if (st, col) in done]
        if steps:
            out.append({"column": col, "action": action, "status": "done", "by_steps": steps})
        elif action == "cap_outliers" and col in run.no_impute:
            out.append({"column": col, "action": action, "status": "skipped",
                        "reason": "values have a special meaning (special code column)"})
        elif col in run.kept_as_category:
            out.append({"column": col, "action": action, "status": "overridden",
                        "reason": "keep_as_category: 'unknown' is kept as its own category"})
        elif action in ACTION_STEPS:
            out.append({"column": col, "action": action, "status": "nothing_to_do",
                        "reason": "no matching values were left when the step ran"})
        else:
            out.append({"column": col, "action": action, "status": "skipped",
                        "reason": SKIPPED_ACTIONS.get(action, "not supported yet")})
    return out


def _summary(run: CleaningRun, rows_before: int, cols_before: int) -> dict:
    """Totals over all steps (a cell changed by two steps counts twice in cell_changes)."""
    def total(step: str, key: str = "cells_changed", rule_start: str = "") -> int:
        return sum(s[key] for s in run.steps
                   if s["step"] == step and s["rule"].startswith(rule_start))

    return {
        "rows_before": rows_before,
        "rows_after": int(len(run.df)),
        "columns_before": cols_before,
        "columns_after": int(run.df.shape[1]),
        "columns_added": list(run.added_columns),
        "duplicates_removed": total("drop_duplicates", "rows_changed"),
        "whitespace_trimmed": total("trim_whitespace"),
        "special_codes_to_flag": sum(s["cells_changed"] for s in run.steps
                                     if s["step"] == "special_codes"
                                     and s["column"] not in run.added_columns),
        "disguised_missing_converted": total("disguised_missing_to_nan"),
        "categories_normalised": total("normalise_categories"),
        "not_a_number_to_nan": total("convert_to_number"),
        "dates_reformatted": total("parse_dates", rule_start="reformat"),
        "not_a_date_to_nan": total("parse_dates", rule_start="not a date"),
        "invalid_format_to_nan": total("invalid_format_to_nan"),
        "impossible_values_fixed": total("impossible_value_to_nan"),
        "out_of_range_to_missing": total("out_of_range_to_missing"),
        "fuzzy_duplicates": {k: v for k, v in run.fuzzy.items()
                             if k not in ("pairs", "key_columns", "compared_columns")},
        "outliers": {**run.outlier_stats,
                     "flagged": total("outlier_flag"), "capped": total("outlier_cap"),
                     "set_to_nan": total("outlier_to_nan")},
        "kept_as_category": dict(run.kept_as_category),
        "values_imputed": total("impute"),
        "cell_changes": sum(s["cells_changed"] for s in run.steps
                            if s["column"] not in run.added_columns),
    }


def clean_dataset(df: pd.DataFrame, report: dict,
                  config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean a DataFrame (read as text) using its Module 1 profiling report.

    Input:  df, its profiling report, optional config (merged over DEFAULT_CONFIG).
    Output: (cleaned copy of df, cleaning log dict). df itself is never changed.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    bad = {c: v for c, v in cfg["impute"].items() if v not in IMPUTE_STRATEGIES}
    bad.update({k: cfg[k] for k in ("numeric_strategy", "categorical_strategy")
                if cfg[k] not in IMPUTE_STRATEGIES})
    bad_outliers = {c: a for c, a in {**cfg["outliers"], "*": cfg["outlier_action"]}.items()
                    if a not in OUTLIER_ACTIONS}
    if bad_outliers:
        raise ValueError(f"unknown outlier action {bad_outliers}; use one of "
                         f"{sorted(OUTLIER_ACTIONS)}")
    if bad:
        raise ValueError(f"unknown imputation strategy {bad}; use one of "
                         f"{sorted(IMPUTE_STRATEGIES)}")
    start = time.perf_counter()
    run = CleaningRun(df=df.copy(), schema=infer_schema(df, report), config=cfg,
                      suggestions=report.get("cleaning_suggestions", []))
    evaluation, eval_seconds = None, 0.0
    for step in STEPS:
        if step is impute and cfg["evaluate_imputers"]:
            # score the imputers on this data: cleaned, but nothing filled in yet
            t0 = time.perf_counter()
            features = feature_columns(run)
            evaluation = evaluate_imputers(       # without flagged outliers, as impute()
                learning_frame(run), [c for c in features if c not in run.no_impute], features)
            eval_seconds = time.perf_counter() - t0
        step(run)
    run.df = _whole_numbers_as_int(run.df)

    log = {
        "log_version": LOG_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": report["dataset"]["name"],
        "step_order": [s.__name__ for s in STEPS],
        "config": {k: v for k, v in cfg.items() if k != "max_examples"},
        "steps": run.steps,
        "suggestions": _suggestion_status(run),
        "summary": _summary(run, len(df), df.shape[1]),
    }
    # cleaning time only; the (optional) evaluation is timed on its own
    log["summary"]["seconds"] = round(time.perf_counter() - start - eval_seconds, 2)
    if run.fuzzy.get("pairs"):
        log["fuzzy_duplicates"] = run.fuzzy             # the pairs with their scores
    if evaluation is not None:
        log["imputer_evaluation"] = evaluation
    return run.df, log


def _schema_keep_unknown(schema: dict, columns: list[str]) -> None:
    """In keep_as_category columns, "unknown" is an allowed value, not a missing code.

    (Module 1 re-profiles the cleaned file and still calls "unknown" disguised missing.)
    """
    for c in columns:
        col = schema["columns"].get(c)
        if not col or col["expected_type"] != "categorical":
            continue
        col["missing_codes"] = [m for m in col["missing_codes"] if category_key(m) != "unknown"]
        col["allowed_values"] = sorted(set(col["allowed_values"]) | {"unknown"})
        col["unknown_is_category"] = True


def clean_file(input_path: str | Path, out_dir: str | Path, report: dict | None = None,
               config: dict | None = None) -> dict:
    """Clean a CSV file end to end and write the three outputs to out_dir.

    1. read the raw CSV (only read — the raw file is never written)
    2. no report given -> run Module 1 profiling first (saved as profiling_report.json)
    3. clean -> cleaned_data.csv
    4. profile the cleaned file -> quality score after cleaning + delta
    5. infer the schema from the CLEANED data -> schema.json (what good data looks like)
    6. cleaning_log.json
    Returns the cleaning log dict.
    """
    start = time.perf_counter()
    input_path, out_dir = Path(input_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_dir / "cleaned_data.csv"
    if cleaned_path.resolve() == input_path.resolve():
        raise ValueError("the output would overwrite the input file; choose another --output")

    raw = read_csv_raw(input_path)
    if report is None:
        report = profile_dataset(raw, dataset_name=input_path.name)
        save_report(report, out_dir)

    cleaned, log = clean_dataset(raw, report, config)
    cleaned.to_csv(cleaned_path, index=False)

    # Re-read the file exactly as a user of cleaned_data.csv would, then profile it
    cleaned_raw = read_csv_raw(cleaned_path)
    after = profile_dataset(cleaned_raw, dataset_name=cleaned_path.name)
    # columns added by cleaning (flags) are scored apart, so they cannot inflate the score
    before_score = quality_score(report)
    after_score = quality_score(after, flag_columns=log["summary"]["columns_added"])
    log["quality"] = {"before": before_score, "after": after_score,
                      "delta": quality_delta(before_score, after_score)}

    schema = infer_schema(cleaned_raw, after)
    schema["source"] = "inferred from cleaned_data.csv (the data after cleaning)"
    _schema_keep_unknown(schema, log["config"]["keep_as_category"])
    save_schema(schema, out_dir)

    if "imputer_evaluation" in log:
        # the full evaluation gets its own file + chart; the log keeps the averages
        full = log.pop("imputer_evaluation")
        files = save_evaluation(full, out_dir, title=input_path.name)
        log["imputer_evaluation"] = {
            "best": full.get("best"), "files": files,
            "methods": {m: {k: v for k, v in r.items() if k != "columns"}
                        for m, r in full.get("methods", {}).items()},
        }

    log["input"] = {"file": str(input_path), "rows": int(len(raw)), "columns": int(raw.shape[1])}
    log["outputs"] = {"cleaned_data": str(cleaned_path),
                      "schema": str(out_dir / "schema.json"),
                      "cleaning_log": str(out_dir / "cleaning_log.json")}
    log["summary"]["total_seconds"] = round(time.perf_counter() - start, 2)
    (out_dir / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=_jsonable))
    return log
