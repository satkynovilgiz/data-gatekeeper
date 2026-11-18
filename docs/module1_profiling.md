# Module 1 — Profiling & Metadata Intelligence

## What it does
Reads any CSV and describes it **before** cleaning: column types, meaning (email, phone, date, ID,
credit card, ...), missing and hidden-missing values, distributions, outliers, correlations,
personal data (PII), suspicious columns, and a list of problems to fix.

Three ways to use it: **CLI**, **Python function**, **HTTP API**. All three return the same
`profiling_report.json`, described by a real JSON Schema:
[`profiling_report.schema.json`](profiling_report.schema.json).

## 1. CLI
```bash
python run_profiling.py --input data/raw/bank_marketing.csv --output results/bank
```
The separator (`,` `;` tab `|`) is detected automatically. The CLI also draws PNG charts in `results/.../charts/`.

## 2. Python
```python
from src.profiling import profile_file, save_report

report = profile_file("data/raw/bank_marketing.csv")   # -> dict
save_report(report, "results/")                         # -> results/profiling_report.json
```

| Function | Input | Output |
|---|---|---|
| `profile_file(path, dataset_name=None)` | path to a CSV | `dict` (report) |
| `profile_dataset(df, dataset_name)` | `pandas.DataFrame` (read with `dtype=str`) | `dict` (report) |
| `read_csv_raw(path)` | path | `DataFrame`, all values as text, separator auto-detected |
| `save_report(report, out_dir)` | `dict`, folder | `Path` to JSON |
| `profile_column(name, series)` | one column | `dict` |
| `run_rules(df, columns)` | df + column profiles | `list[dict]` (issues) |
| `cramers_v(a, b)` | two columns | `float` 0..1 or `None` |

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

## `profiling_report.json` (version 1.1)
The full contract is [`profiling_report.schema.json`](profiling_report.schema.json); `tests/test_schema.py` checks every report against it.
```text
{
  report_version: "1.1",
  generated_at: ISO datetime,
  dataset: { name, rows, columns, memory_kb, duplicate_rows, total_missing_cells, missing_cells_pct },
  columns: [ {
      name, pandas_dtype,
      inferred_type: numeric | datetime | categorical | text | boolean | empty,
      semantic: { semantic_type, confidence (0-1), evidence },
      is_pii, missing_count, missing_pct, unique_count, unique_pct,
      value_types: { "number": n, "text": n }, is_mixed_type,
      top_value: { value, pct } | null,                 # NEW: most common value
      placeholder_values: { "unknown": n, ... },        # NEW: hidden missing values
      numeric_stats?:     { mean, std, min, p25, median, p75, max, skew,
                            outliers_iqr: { count, pct, lower_fence, upper_fence },
                            histogram: { bin_edges[11], counts[10] } },          # NEW
      categorical_stats?: { top_values, top_values_pct, other_values_count,     # NEW: pct + tail
                            case_or_space_variants },
      date_formats?:      { "9999-99-99": n, ... }
  } ],
  missing_matrix: { column: missing_count },
  correlation:             { method: "pearson",   matrix, strong_pairs: [ {a, b, corr} ] },        # |r| >= 0.7
  categorical_association: { method: "cramers_v", matrix, strong_pairs: [ {a, b, cramers_v} ] },   # NEW, V >= 0.5
  issues: [ { column, rule, severity: high | medium | info, detail } ],
  summary: { pii_columns, mixed_type_columns, suspicious_columns, issue_counts },
  charts?: { missing_heatmap, correlation_heatmap, outlier_distribution }   # CLI only
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
| `mostly_missing` | high | ≥ 50% missing |
| `missing_values` | medium | ≥ 5% missing |
| `mixed_types` | high | numbers and text in the same column |
| `duplicate_rows` | high | exact duplicate rows |
| `inconsistent_category_format` | medium | same value written differently (`HR`, `hr `) |
| `inconsistent_date_format` | medium | more than one date shape |
| `outliers` | medium | ≥ 1% values outside 1.5 × IQR |
| `invalid_format_values` | medium | some emails / phones / postcodes don't match the format |
| `constant_column` | medium | only one distinct value |
| `potential_pii` | info | any PII semantic type above |
| `id_stored_as_number` ⭐ | medium | ID column read as a number (no maths on IDs; leading zeros get lost) |
| `dominant_value` ⭐ | medium | one value in **> 90%** of rows (almost constant, or a magic number like `999`) |
| `placeholder_values` ⭐ | medium | text like `unknown`, `?`, `n/a`, `-`, `none` = hidden missing data |

⭐ = "suspicious column" rules, listed in `summary.suspicious_columns` (together with `constant_column`).

## Known limitations (planned for Sprint 3)
- Profiling 41k rows takes ~10 s; the per-value type check is slow.
- When one value dominates (`pdays = 999`), IQR = 0, so every other value counts as an "outlier".
- Placeholders are reported but not yet counted in `missing_pct`.

## How Module 2 should use it
- `issues` = the to-do list for cleaning.
- `inferred_type` / `semantic` = how to convert each column (to number, date, category).
- `placeholder_values` = values to turn into real missing values (NaN).
- `is_pii` = columns to mask or protect.
