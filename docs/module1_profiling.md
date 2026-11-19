# Module 1 — Profiling & Metadata Intelligence (v1.2, complete)

## What it does
Reads any CSV and describes it **before** cleaning: column types, meaning (email, phone, date, ID,
credit card, ...), missing values **including disguised ones** (`unknown`, `999`), distributions,
outliers, correlations, personal data (PII), suspicious columns, a **data quality score**, and a
list of **cleaning suggestions** for Module 2.

Three ways to use it: **CLI**, **Python function**, **HTTP API**. All three return the same
`profiling_report.json`, described by a real JSON Schema:
[`profiling_report.schema.json`](profiling_report.schema.json).

## 1. CLI
```bash
python run_profiling.py --input data/raw/bank_marketing.csv --output results/bank
python run_profiling.py --input data/raw/bank_marketing.csv --output results/quick --sample 5000
```
- The separator (`,` `;` tab `|`) is detected automatically.
- `--sample N` profiles **N random rows** (seed 42, so the same rows every time). Use it for very big files.
- Each step's time is logged (`Profiling took 2.55 s`).
- PNG charts are saved in `<output>/charts/`.

## 2. Python
```python
from src.profiling import profile_file, save_report
from src.quality import quality_score, quality_delta

report = profile_file("data/raw/bank_marketing.csv")            # -> dict
report = profile_file("big.csv", sample=10_000)                 # random 10k rows
save_report(report, "results/")                                 # -> results/profiling_report.json
```

| Function | Input | Output |
|---|---|---|
| `profile_file(path, dataset_name=None, sample=None)` | path to a CSV | `dict` (report) |
| `profile_dataset(df, dataset_name, rows_in_file=None)` | `DataFrame` (read with `dtype=str`) | `dict` (report) |
| `read_csv_raw(path)` | path | `DataFrame`, all values as text, separator auto-detected |
| `sample_rows(df, n)` | df, n | random n rows (seed 42), or df unchanged if smaller |
| `save_report(report, out_dir)` | `dict`, folder | `Path` to JSON |
| `run_rules(df, columns)` | df + column profiles | `list[dict]` (issues) |
| `cleaning_suggestions(columns, issues)` | column profiles + issues | `list[dict]` |
| `quality_score(report)` (`src/quality/score.py`) | a report | `{overall, dimensions, weights}` |
| `quality_delta(before, after)` | two scores | improvement per dimension (for Module 2) |

## 3. HTTP API (FastAPI)
```bash
uvicorn src.api.profiling_api:app --reload
# interactive docs: http://127.0.0.1:8000/docs
curl -F "file=@data/raw/bank_marketing_sample.csv" http://127.0.0.1:8000/profile -o profiling_report.json
```
| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | – | `{"status": "ok"}` |
| POST | `/profile` | form field `file` = a `.csv` file (max 50 MB) | the report (JSON). `400` for a non-CSV, empty or unreadable file; `413` if too big; `422` if `file` is missing |

## Disguised missing values (Sprint 3)
Real data often hides "no value" inside a normal-looking value. pandas sees these cells as full.

| Kind | Examples | How it is detected |
|---|---|---|
| Placeholder text | `unknown`, `unk`, `n/a`, `none`, `null`, `-`, `?`, `missing`, blank `"  "` | `strip().lower()` is in a fixed list |
| Numeric sentinel | `999`, `9999`, `99999`, `-1`, `-99`, `-999`, `-9999` | only when **far outside** the real values (below) |

A numeric sentinel only counts when it is clearly separate from the other values:
- **positive** (`999`): bigger than **2 ×** the largest other value. `pdays`: other values are 0–27, so 999 is a code. A salary column with values 500–5000 keeps its real 999.
- **negative** (`-1`): every other value is **≥ 0**. In `emp.var.rate`, negative numbers are real, so -1 is kept.

Per column the report shows `disguised_missing = {count, pct, values}` and
`true_missing_pct = missing_pct + disguised pct`. Numeric stats, histograms, correlation and the
outlier chart are computed **without** the sentinels (`pdays` mean: 962 with them, **6.0** without).

## Data quality score (Sprint 3)
`src/quality/score.py` computes a score from 0 to 100 (100 = no problems found) from a report.
Module 1 stores it as `quality_score` (score **before** cleaning). Module 2 will profile the cleaned data, call the
same function and use `quality_delta(before, after)` to show the improvement.

| Dimension | Formula | Counted per |
|---|---|---|
| **completeness** | `100 − true_missing_cells_pct` (empty + disguised cells / all cells) | cell |
| **uniqueness** | `100 − duplicate_rows / rows × 100` | row |
| **validity** | `% of columns` with no `mixed_types` or `invalid_format_values` issue | column |
| **consistency** | `% of columns` with no `inconsistent_category_format` or `inconsistent_date_format` issue | column |

```
overall = 0.25 × completeness + 0.25 × uniqueness + 0.25 × validity + 0.25 × consistency
```
Why some dimensions are per column: format problems are fixed per column (one "lowercase" step
repairs the whole column), so "how many columns are OK" is the honest measure.

Example (Bank Marketing, 41,188 rows): completeness 93.94, uniqueness 99.97, validity 100,
consistency 100 → **overall 98.48**. The synthetic HR dataset scores **88.86** (mixed types, case variants).

## Cleaning suggestions (hand-off to Module 2)
For every issue, `cleaning_suggestions` gives an `action` code (for the program) and a `detail` (for people), most severe first.

| Rule | Action |
|---|---|
| `duplicate_rows` | `drop_duplicates` |
| `mostly_missing`, `constant_column` | `drop_column` |
| `missing_values` | `impute_median` (numeric) / `impute_mode` (other) |
| `disguised_missing` (< 50% true missing) | `replace_with_nan_then_impute_median` / `..._mode` |
| `disguised_missing` (≥ 50% true missing) | `replace_with_nan_and_add_flag` (too much to fill in) |
| `mixed_types` | `convert_to_number` (numeric) / `review_values` |
| `inconsistent_category_format` | `standardise_case` |
| `inconsistent_date_format` | `parse_dates_iso` (YYYY-MM-DD) |
| `outliers` | `cap_outliers` (winsorise at the IQR fences) |
| `invalid_format_values` | `set_invalid_to_nan` |
| `potential_pii` | `mask_pii` |
| `id_stored_as_number` | `convert_to_text` |
| `dominant_value` | `review_values` |

## Charts (CLI)
```
<output>/charts/
├── missing_heatmap.png          # dataset
├── correlation_heatmap.png      # dataset (needs ≥ 2 numeric columns)
├── outlier_distribution.png     # dataset (standardised boxplots)
├── hist_<column>.png            # one per numeric column (without sentinels)
└── bar_<column>.png             # one per categorical column, top 10 values, disguised values in orange
```
All file names are listed in `report["charts"]` (`histograms` and `bar_charts` map column → file).

## `profiling_report.json` (version 1.2)
The full contract is [`profiling_report.schema.json`](profiling_report.schema.json); `tests/test_schema.py` checks every report against it.
```text
{
  report_version: "1.2",
  generated_at: ISO datetime,
  dataset: { name, rows, columns, memory_kb, duplicate_rows, total_missing_cells, missing_cells_pct,
             disguised_missing_cells, true_missing_cells_pct },                       # NEW 1.2
  columns: [ {
      name, pandas_dtype,
      inferred_type: numeric | datetime | categorical | text | boolean | empty,
      semantic: { semantic_type, confidence (0-1), evidence },
      is_pii, missing_count, missing_pct, unique_count, unique_pct,
      value_types: { "number": n, "text": n }, is_mixed_type,
      top_value: { value, pct } | null,
      disguised_missing: { count, pct, values: { "unknown": n, "999": n } },        # NEW 1.2 (was placeholder_values)
      true_missing_count, true_missing_pct,                                         # NEW 1.2
      numeric_stats?:     { mean, std, min, p25, median, p75, max, skew,            # without sentinels
                            outliers_iqr: { count, pct, lower_fence, upper_fence }, # fences null when IQR = 0
                            histogram: { bin_edges[11], counts[10] } },
      categorical_stats?: { top_values, top_values_pct, other_values_count, case_or_space_variants },
      date_formats?:      { "9999-99-99": n, ... }
  } ],
  missing_matrix: { column: missing_count },
  correlation:             { method: "pearson",   matrix, strong_pairs: [ {a, b, corr} ] },        # |r| >= 0.7
  categorical_association: { method: "cramers_v", matrix, strong_pairs: [ {a, b, cramers_v} ] },   # V >= 0.5
  issues: [ { column, rule, severity: high | medium | info, detail } ],
  summary: { pii_columns, mixed_type_columns, suspicious_columns, issue_counts },
  quality_score: { overall, dimensions: { completeness, uniqueness, validity, consistency }, weights },  # NEW 1.2
  cleaning_suggestions: [ { column, rule, severity, action, detail } ],                                # NEW 1.2
  run: { seconds, rows_in_file, rows_profiled, sampled },                                              # NEW 1.2
  charts?: { missing_heatmap, correlation_heatmap?, outlier_distribution?,
             histograms: { column: file }, bar_charts: { column: file } }                              # CLI only
}
```

## Semantic types and PII
| Semantic type | How it is detected | PII? |
|---|---|---|
| `email`, `phone`, `postcode`, `url` | ≥ 80% of values match a regex | yes (not url) |
| `credit_card` | 13–19 digits **and** the Luhn checksum passes (so random numbers are not flagged) | yes |
| `ip_address` | valid IPv4, each part 0–255 (checked before phone) | yes |
| `date_of_birth` | column name contains `dob`, `birth`, `birthday` | yes |
| `person_name`, `address` | column name hints (`name`, `surname`, `street`, `city`) | yes |
| `id` | name has the **word** `id`/`uuid`/`code`/`key` (so `cons.price.idx` is not an ID) | no |
| `date` | values parse as dates | no |

## Rules (profiling rule engine)
| Rule | Severity | When |
|---|---|---|
| `mostly_missing` | high | ≥ 50% empty |
| `missing_values` | medium | ≥ 5% empty |
| `disguised_missing` ⭐ (new) | high if true missing ≥ 50%, else medium | placeholder text or numeric sentinel found |
| `mixed_types` | high | numbers and text in the same column |
| `duplicate_rows` | high | exact duplicate rows |
| `inconsistent_category_format` | medium | same value written differently (`HR`, `hr `) |
| `inconsistent_date_format` | medium | more than one date shape |
| `outliers` | medium | ≥ 1% values outside 1.5 × IQR (skipped when IQR = 0) |
| `invalid_format_values` | medium | some emails / phones / postcodes don't match the format |
| `constant_column` | medium | only one distinct value |
| `potential_pii` | info | any PII semantic type above |
| `id_stored_as_number` ⭐ | medium | ID column read as a number (no maths on IDs; leading zeros get lost) |
| `dominant_value` ⭐ | medium | one value in **> 90%** of rows, and it is not already a known missing code |

⭐ = "suspicious column" rules, listed in `summary.suspicious_columns` (together with `constant_column`).

## Performance
| Bank Marketing (41,188 × 21) | Sprint 2 | Sprint 3 |
|---|---|---|
| profiling only | 5.1 s | **2.55 s** |
| full CLI (read + profile + 24 charts + save) | – | **3.55 s** |
| `--sample 5000` profiling | – | 0.42 s |

The speed-up comes from checking "number or text?" for a whole column in one `pd.to_numeric` call instead of one call per value.

## Known limitations
- **Sampling can miss rare things.** The 5,000-row sample finds 0 of the 12 duplicate rows. Use the full file for the final report.
- Numeric sentinels come from a fixed list. A code like `88` would not be found.
- Validity and consistency are per column, so one bad value counts the same as many.
- The planned one-page HTML report is moved to Module 4 (presentation layer).

## How Module 2 should use it
- `cleaning_suggestions` = the ordered to-do list (`action` codes are fixed in the schema).
- `disguised_missing.values` = the exact text to turn into NaN before imputing.
- `inferred_type` / `semantic` = how to convert each column (to number, date, category).
- `is_pii` = columns to mask or protect.
- `quality_score` = the "before" score. Profile the cleaned file and call `quality_delta(before, after)`.
