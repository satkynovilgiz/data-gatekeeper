# Sprint 3 Summary — Module 1 complete + hand-off to Module 2

**Goal:** finish Module 1. Find hidden missing values, add a quality score and cleaning suggestions for Module 2, make profiling faster, and present the module.

## What I built
| Feature | Where |
|---|---|
| Disguised missing values: placeholder text (`unknown`, `n/a`, `-`, `?`, blank) + numeric sentinels (`999`, `-1`, …) only when far outside the real values; `true_missing_pct` per column | `src/profiling/profiler.py` |
| New rule `disguised_missing` (high when ≥ 50% truly missing); stats, correlation and charts ignore sentinels | `profiler.py` |
| Fixed: outlier rule when IQR = 0 (no more "every non-zero value is an outlier") | `profiler.py` |
| Histogram per numeric column + top-10 bar chart per categorical column (24 charts on the bank data) | `src/profiling/visualize.py` |
| Data quality score 0–100: completeness, uniqueness, validity, consistency + `quality_delta()` for Module 2 | `src/quality/score.py` |
| `cleaning_suggestions`: one action code per issue (`impute_median`, `standardise_case`, `drop_duplicates`, `mask_pii`, …) | `src/profiling/suggestions.py` |
| Speed: vectorised type check; `--sample N`; timing logged and stored in `report.run` | `profiler.py`, `run_profiling.py` |
| JSON Schema v1.2 + tests: **44 → 70**, all passing | `docs/profiling_report.schema.json`, `tests/` |
| Module 1 presentation | `docs/module1_presentation.md` |

## Results on the full Bank Marketing dataset (41,188 rows × 21 columns)
- **Empty cells 0.0%, but 52,391 disguised missing cells → true missing 6.06%** in 7 columns.
- `pdays = 999` in **96.32%** of rows (high severity). `default = unknown` in **20.87%**.
- **12 duplicate rows**, **no PII** (correct for anonymised data), **13 issues** (2 high, 11 medium).
- **Quality score 98.48 / 100**: completeness 93.94, uniqueness 99.97, validity 100, consistency 100. The synthetic HR data scores 88.86.
- After removing the 999s, `pdays` has a real mean of **6.0** days instead of a fake 962.
- **Run time:** profiling **2.5 s** (was 5.1 s in Sprint 2); full CLI with 24 charts **3.5 s**; `--sample 5000` profiling **0.4 s**.

## What I learned
1. Missing data hides in plain sight. A rule for magic numbers needs context: 999 is a code only when it is far from the real values.
2. One wrong statistic spreads: sentinels distorted the mean, the outliers, the correlation and the charts until they were removed everywhere.
3. Vectorising (one pandas call per column) halved the run time without changing any result. I checked the output before and after.
4. A sample is fast but can miss rare things: 5,000 rows found 0 of the 12 duplicates.
5. A shared scoring function lets Module 2 measure improvement in the same way.

## Plan for Sprint 4 (Module 2 — cleaning basics)
- Read `profiling_report.json` and run `cleaning_suggestions` in order, starting with high severity
- Replace disguised missing values with NaN; fix types; strip and lowercase categories; drop duplicates
- Write `cleaning_log.json` (column, action, rows affected) and `cleaned_data.csv`
- Profile the cleaned file and report `quality_delta(before, after)`
- Tests for each cleaning step
