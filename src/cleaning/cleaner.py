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
    10. impute                    NaN -> median (numbers) / mode (categories)

Missing codes and bad values become NaN first (steps 2-8), duplicates go next (so
they do not bias the median), and only then are the gaps filled (step 10).
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
from .imputers import METHODS, describe, impute_columns, is_numeric
from .schema import canonical_forms, category_key, infer_schema, save_schema

LOG_VERSION = "1.0"

DEFAULT_CONFIG: dict[str, Any] = {
    # imputation per column: "median" | "mean" | "mode" | "knn" | "regression" |
    # "iterative" | "none". Columns not listed use the defaults below by schema type
    # ("median" on a category means its mode). Dates, text, ids and personal data are
    # never filled in: an invented email or hire date would be fake data.
    "impute": {},
    "numeric_strategy": "median",
    "categorical_strategy": "mode",
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
    # hide 10% of known values and score every imputer on them (slower: off by default)
    "evaluate_imputers": False,
    "max_examples": 3,
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
}
SKIPPED_ACTIONS = {
    "cap_outliers": "planned for Sprint 5 (outlier handling); impossible values are "
                    "already set to NaN by impossible_value_to_nan",
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


def impute(run: CleaningRun) -> None:
    """Step 10: fill the remaining NaN with each column's method (median/mode or ML).

    Columns that share an ML method are filled together in one call, so KNN and
    iterative see all of them at once.
    """
    by_method: dict[str, list[str]] = {}
    for c in run.df.columns:
        strategy = _strategy(run, c)
        if strategy != "none" and run.df[c].isna().any() and run.df[c].notna().any():
            by_method.setdefault(strategy, []).append(c)
    features = feature_columns(run)
    for method, targets in by_method.items():
        filled = impute_columns(run.df, targets, features, method)
        for c in targets:
            s, new = run.df[c], filled[c]
            gaps = s.isna()
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


STEPS = [trim_whitespace, special_codes, disguised_missing_to_nan, normalise_categories,
         convert_to_number, parse_dates, invalid_format_to_nan, impossible_value_to_nan,
         drop_duplicates, impute]


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
            evaluation = evaluate_imputers(
                run.df, [c for c in features if c not in run.no_impute], features)
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
    before_score, after_score = quality_score(report), after["quality_score"]
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
