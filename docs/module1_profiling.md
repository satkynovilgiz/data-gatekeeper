# Module 1 — Profiling & Metadata Intelligence

## What it does
Reads any CSV and describes it **before** cleaning: column types, meaning (email, phone, date, ID, name),
missing values, unique values, distributions, outliers, correlations, and problems to fix.

## How to run
```bash
python run_profiling.py --input <file.csv> --output results/
```

## Python API (for Module 2 and Module 4)
```python
import pandas as pd
from src.profiling.profiler import profile_dataset, save_report

df = pd.read_csv("data.csv", dtype=str)       # read raw, as text
report = profile_dataset(df, dataset_name="data.csv")   # -> dict
save_report(report, "results/")               # -> results/profiling_report.json
```

| Function | Input | Output |
|---|---|---|
| `profile_dataset(df, dataset_name)` | `pandas.DataFrame` | `dict` (report) |
| `save_report(report, out_dir)` | `dict`, folder | `Path` to JSON |
| `profile_column(name, series)` | column | `dict` (one column) |
| `run_rules(df, columns)` | df + column profiles | `list[dict]` (issues) |

## `profiling_report.json` schema
```text
{
  report_version: str,
  generated_at: ISO datetime,
  dataset: { name, rows, columns, memory_kb, duplicate_rows,
             total_missing_cells, missing_cells_pct },
  columns: [ {
      name, pandas_dtype,
      inferred_type: "numeric" | "datetime" | "categorical" | "text" | "boolean" | "empty",
      semantic: { semantic_type, confidence (0-1), evidence },
      is_pii: bool,
      missing_count, missing_pct, unique_count, unique_pct,
      value_types: { "number": int, "text": int },
      is_mixed_type: bool,
      numeric_stats?:     { mean, std, min, p25, median, p75, max, skew,
                            outliers_iqr: { count, pct, lower_fence, upper_fence } },
      categorical_stats?: { top_values, case_or_space_variants },
      date_formats?:      { "9999-99-99": count, ... }
  } ],
  missing_matrix: { column: missing_count },
  correlation: { method, matrix, strong_pairs: [ {a, b, corr} ] },
  issues: [ { column, rule, severity: "high"|"medium"|"info", detail } ],
  summary: { pii_columns, mixed_type_columns, issue_counts },
  charts: { missing_heatmap, correlation_heatmap, outlier_distribution }
}
```

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
| `potential_pii` | info | email, phone, name, address, postcode |

## How Module 2 should use it
- `issues` = the to-do list for cleaning.
- `inferred_type` / `semantic` = how to convert each column (to number, date, category).
- `is_pii` = columns to mask or protect.
