# Module 2 — Cleaning & Transformation (Sprint 4: core pipeline)

## What it does
Takes a raw CSV and its Module 1 `profiling_report.json` and produces:

| Output | What is in it |
|---|---|
| `cleaned_data.csv` | the cleaned data (comma separated, dates `YYYY-MM-DD`, empty cell = missing) |
| `cleaning_log.json` | every change: step, column, rule, how many cells/rows, before → after examples, summary, quality score before → after |
| `schema.json` | what every column **should** look like: type, ranges, allowed values, format (for Module 3) |

The raw input file is **only read, never written**. The CLI refuses an `--output` that would overwrite it.

```
raw.csv ─► Module 1 ─► profiling_report.json ─┐
   │                    (cleaning_suggestions) │
   └──────────────────────────────────────────►┴─► infer_schema ─► 10 cleaning steps ─► cleaned_data.csv
                                                                                    ├─► cleaning_log.json
                         profile cleaned_data.csv again (quality AFTER, schema) ────┴─► schema.json
```

## How to run it
```bash
# with a profiling report you already have
python run_profiling.py --input data/raw/bank_marketing.csv --output results/bank
python run_cleaning.py  --input data/raw/bank_marketing.csv \
                        --profile results/bank/profiling_report.json --output results/bank

# without --profile: Module 1 runs first (its report is saved next to the outputs)
python run_cleaning.py --input data/raw/hr_employees_messy.csv --output results/hr

# imputation per column: median | mean | mode | none  (repeat --impute)
python run_cleaning.py --input data/raw/bank_marketing.csv --output results/bank \
                       --impute default=none --impute age=mean
```
From Python:
```python
from src.cleaning import clean_file, clean_dataset, infer_schema
log = clean_file("data/raw/hr_employees_messy.csv", "results/hr")             # writes the 3 files
cleaned_df, log = clean_dataset(raw_df, report, {"impute": {"age": "mean"}})  # in memory only
```

## How the report drives the cleaning
- `cleaning_suggestions` (Module 1) = **what** is wrong: the to-do list.
- `schema` (`src/cleaning/schema.py`) = **how** each column should look, so the steps know how to fix it.
- At the end, the log's `suggestions` section marks every suggestion as `done` (and by which steps),
  `nothing_to_do` or `skipped` with a reason. Nothing is silently ignored:

| Module 1 action | Sprint 4 |
|---|---|
| `drop_duplicates`, `impute_median`, `impute_mode`, `replace_with_nan_then_impute_*`, `replace_with_nan_and_add_flag`, `convert_to_number`, `standardise_case`, `parse_dates_iso`, `set_invalid_to_nan` | ✅ done |
| `cap_outliers` | skipped → Sprint 5 (outliers); impossible values are already handled |
| `mask_pii` | skipped → Sprint 6 |
| `drop_column`, `review_values` | skipped: they need a person's decision |
| `convert_to_text` | nothing to do: IDs are read and written as text |

## The schema (`schema.json`)
`infer_schema(df, report)` describes each column:

| Field | Meaning | Example |
|---|---|---|
| `expected_type` | `numeric`, `categorical`, `date`, `text` or `id` | `age` → numeric |
| `nullable` | may the column have empty cells | `pdays` → true |
| `missing_codes` | values that mean "missing" | `["unknown"]`, `["999"]` |
| `typical_range` | p1–p99 of the real values (for **warnings**, not errors) | salary `[38114, 241852]` |
| `valid_range` + `range_source` | a business rule: outside = **impossible** | age `[0, 120]`, `business_rule:age` |
| `observed_range`, `is_integer` | min/max of the possible values; whole numbers only? | |
| `allowed_values` | categories after merging spelling variants | `["Engineering", "Finance", "HR", ...]` |
| `format` (+ `pattern`) | `YYYY-MM-DD` for dates; `email`, `phone`, ... with the regex | |
| `unique` | for `id` columns | `employee_id` → true |

How the type is chosen: `id` and `date` from Module 1's meaning; `numeric` when ≥ 80% of the **real**
values are numbers (missing codes like `unknown` removed first); email/phone/URL-like values are always
`text` (never filled in with the most common email); otherwise Module 1's `categorical` / `text`.

**Two ranges on purpose.** `duration = 4918` s in the bank data is rare but real; `age = 230` is impossible.
Only `valid_range` (business rules, in `BUSINESS_RULES`, matched on whole words of the column name:
`age` 0–120, `experience` 0–70, `salary`/`income`/`price`/`amount`/`duration` ≥ 0, `pct`/`percent` 0–100)
turns a value into NaN. The p1–p99 `typical_range` would wrongly delete 2% of good values.

The cleaner uses a schema of the **raw** data to decide how to clean. The saved `schema.json` is
inferred again from the **cleaned** data, so it describes good data (e.g. `employee_id` is `unique: true`
only after the duplicates are gone). That is what Module 3 will validate against.

## The steps, in order
| # | Step | What it does | Example |
|---|---|---|---|
| 1 | `trim_whitespace` | strip spaces at both ends of every value | `"ENGINEERING "` → `"ENGINEERING"` |
| 2 | `special_codes` | codes with a **meaning** → a flag column (see pdays below) | `pdays 999` → NaN + `previously_contacted = no` |
| 3 | `disguised_missing_to_nan` | Module 1's missing codes + placeholder text → NaN. Columns with ≥ 50% missing get a `<col>_missing` 0/1 flag instead of imputation | `"unknown"`, `"?"`, `"n/a"` → NaN |
| 4 | `normalise_categories` | one spelling per category: same after strip + lowercase → the most common spelling; a rare `"Xxx."` abbreviation → the only category starting with `xxx` | `"hr"` → `"HR"`, `"Fin."` → `"Finance"` |
| 5 | `convert_to_number` | numeric columns → real numbers; text that is not a number → NaN (flagged in the log) | `"five"` → NaN |
| 6 | `parse_dates` | every date → `YYYY-MM-DD`; day/month order read from the values (a first part > 12 must be the day); unreadable → NaN | `"20/09/2023"` → `"2023-09-20"` |
| 7 | `invalid_format_to_nan` | text that breaks its format → NaN | `"not-an-email"` → NaN |
| 8 | `impossible_value_to_nan` | outside `valid_range` → NaN, **the row is kept** | `age 230` → NaN |
| 9 | `drop_duplicates` | rows that are exact copies of an earlier row (keep the first) | |
| 10 | `impute` | fill the remaining NaN: **median** for numbers, **mode** for categories; configurable per column (`median`, `mean`, `mode`, `none`). Dates, text and IDs are never filled in (an invented email is fake data). Whole-number columns get a rounded median | `age` NaN → 42 |

Finally, numeric columns that only hold whole numbers are written as integers (`51.0` → `51`).

### Why this order
- **Missing codes → NaN first (2–3)**, so `"unknown"` is not normalised as a category or imputed as a value.
- **Normalise before de-duplicating (4–8 before 9)**: `"HR"` and `"hr "` rows are the same record.
- **De-duplicate before imputing (9 before 10)**: duplicates would pull the median and mode toward the copied rows.
- **Bad values → NaN before imputing (5–8 before 10)**: `"five"` and `230` become gaps that the same step fills.
- **Impute last**, so every gap is filled once, from the cleanest data.

## Decision: `pdays = 999` is NOT a missing value
In UCI Bank Marketing, `pdays` = days since the client was last contacted by a previous campaign, and
**`999` = "never contacted before"** (96.3% of rows). Module 1 correctly marks it as a disguised missing
value, but the information is real: "never contacted" is a fact about the client.

- ❌ Imputing (median 6) would claim 39,673 clients were contacted 6 days ago: invented history.
- ❌ Keeping 999 as a number would destroy every average and correlation (mean 962 instead of 6).
- ✅ **New column `previously_contacted`** (`no` where `pdays` was 999, else `yes`), placed right after `pdays`.
  `pdays` becomes NaN **only** for the never-contacted rows; real values (0–27) are kept. Neither column is imputed.

It is configured in `DEFAULT_CONFIG["special_codes"]`, so the same idea works for other datasets:
```python
"special_codes": {"pdays": {"code": 999, "flag_column": "previously_contacted",
                            "code_means": "no", "otherwise": "yes"}}
```
Note: after cleaning, Module 1 reports `pdays` as `mostly_missing` (96% empty). That is expected. The
empty cells now mean "not applicable", and completeness can't reach 100% for this dataset.

## `cleaning_log.json` (version 1.0)
```text
{
  log_version: "1.0", generated_at, dataset,
  step_order: [ "trim_whitespace", ..., "impute" ],
  config:  { impute, numeric_strategy, categorical_strategy, special_codes, valid_ranges, dayfirst },
  steps: [ {                                   # one entry per step x column that changed something
      step, column, rule,                      # rule = what exactly, e.g. "median = 42"
      cells_changed, rows_changed,             # drop_duplicates: cells 0, rows = rows removed
      examples: [ {row, before, after} ]       # up to 3; row = data row in the input file (0 = first)
  } ],                                         # drop_duplicates examples: {row, duplicate_of_row}
  suggestions: [ {column, action, status: done | nothing_to_do | skipped, by_steps | reason} ],
  summary: { rows_before, rows_after, columns_before, columns_after, columns_added,
             duplicates_removed, whitespace_trimmed, special_codes_to_flag,
             disguised_missing_converted, categories_normalised, not_a_number_to_nan,
             dates_reformatted, not_a_date_to_nan, invalid_format_to_nan,
             impossible_values_fixed, values_imputed,
             cell_changes,                     # sum over steps: a cell changed twice counts twice
             seconds,                          # cleaning only
             total_seconds },                  # + reading, profiling the result, schema, writing
  quality: { before: quality_score, after: quality_score,
             delta: { overall, completeness, uniqueness, validity, consistency } },
  input: { file, rows, columns },
  outputs: { cleaned_data, schema, cleaning_log }
}
```
Example step:
```json
{"step": "impossible_value_to_nan", "column": "age",
 "rule": "outside valid range [0, 120] (business_rule:age) -> NaN",
 "cells_changed": 10, "rows_changed": 10,
 "examples": [{"row": 22, "before": 230.0, "after": null}, ...]}
```

## Quality score before → after
The "after" score uses the **same** function as Module 1 (`src/quality/score.py`): `cleaned_data.csv` is
read back and profiled like any new file, then `quality_delta(before, after)` is computed.

| | Bank Marketing (41,188 × 21) | HR messy sample (515 × 10) |
|---|---|---|
| Rows before → after | 41,188 → 41,176 | 515 → 500 |
| Duplicates removed | 12 | 15 |
| Disguised missing → NaN | 12,718 (`unknown` in 6 columns) | 0 |
| `pdays 999` → `previously_contacted` | 39,673 | – |
| Text → number failures / bad dates / invalid emails | 0 / 0 / 0 | 15 (`"five"`) / 0 / 15 |
| Dates reformatted / categories merged | 0 / 0 | 51 / 19 |
| Impossible values → NaN | 0 | 10 (`age 230`) |
| Values imputed | 12,716 | 88 |
| **Quality score** | **98.48 → 98.90 (+0.42)** | **88.86 → 99.83 (+10.97)** |
| completeness / uniqueness / validity / consistency Δ | +1.68 / +0.03 / 0 / 0 | +0.95 / +2.91 / +20 / +20 |
| Time: cleaning / whole CLI | 1.5 s / 8.3 s | 0.04 s / 0.8 s |

Most of the CLI time on the bank data is profiling the cleaned file again (≈ 5 s) to get the "after" score.

## Known limitations (next sprints)
- **Imputation can create duplicates.** In the bank data, 2 rows differed only in `default`
  (`unknown` vs `no`). After imputing the mode `no` they are identical. They are different clients,
  so they are kept (that's why duplicates are removed *before* imputing).
- **Mode imputation hides information.** `default = unknown` is 20.9% of the bank rows and probably means
  something ("the client didn't say"). Filling it with `no` is what Sprint 4 asks for; Sprint 5 compares
  ML imputers and the option to keep `unknown` as its own category. Use `--impute default=none` to keep NaN.
- **Outliers are not changed yet** (salary 9,999,999 stays: it is possible, just very unlikely) → Sprint 5.
- **PII is not masked yet** (`full_name`, `email`, `phone`) → Sprint 6.
- Business rules are a small fixed list matched on column names. A column named in another language
  gets no rule (set it with `valid_ranges` in the config).
- Ambiguous dates (`05/06/2020`) follow the order found elsewhere in the column; if no value tells, day first.
