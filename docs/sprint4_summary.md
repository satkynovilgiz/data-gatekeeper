# Sprint 4 Summary — Module 2 starts: core cleaning pipeline

**Goal:** turn Module 1's profiling report into action. Infer what each column should look like,
clean the data step by step, log every change, and measure the quality before and after.

## What I built
| Feature | Where |
|---|---|
| Schema inference: expected type, p1–p99 typical range, business-rule valid range (age 0–120), allowed categories, formats → `schema.json` for Module 3 | `src/cleaning/schema.py` |
| 10-step cleaning pipeline in a fixed order (missing codes → NaN, normalise, convert, de-duplicate, impute last) | `src/cleaning/cleaner.py` |
| `pdays = 999` ("never contacted") → new column `previously_contacted`, not imputed | `cleaner.py` (`special_codes`) |
| `cleaning_log.json`: every change with counts and before → after examples; every Module 1 suggestion marked done / skipped with a reason; quality before → after | `cleaner.py` |
| CLI `run_cleaning.py` (`--profile` optional, `--impute COL=STRATEGY`) | `run_cleaning.py` |
| Tests **76 → 110**, all passing; ruff clean | `tests/test_cleaning_*.py` |
| Fixed a Module 1 bug: 5-digit salaries detected as postcodes | `src/profiling/semantic.py` |

## Results
| | Bank Marketing (41,188 × 21) | HR messy sample (515 × 10) |
|---|---|---|
| Rows before → after | 41,188 → 41,176 | 515 → 500 |
| Duplicates removed | 12 | 15 |
| Disguised missing → NaN | 12,718 (`unknown`) + 39,673 `pdays 999` → flag | 0 |
| Impossible values fixed | 0 | 10 (`age = 230`) |
| Other fixes | – | 19 category variants, 51 dates → ISO, 15 `"five"`, 15 invalid emails |
| Values imputed | 12,716 | 88 |
| **Quality score** | **98.48 → 98.90 (+0.42)** | **88.86 → 99.83 (+10.97)** |
| Δ completeness / uniqueness / validity / consistency | +1.68 / +0.03 / 0 / 0 | +0.95 / +2.91 / +20 / +20 |
| Run time: cleaning / whole CLI | 1.5 s / 8.3 s | 0.04 s / 0.8 s |

The bank data only gains +0.42 because it was already clean in format (validity and consistency were
100). Its empty `pdays` cells now honestly mean "not applicable". The HR data shows the full pipeline
at work: +20 in validity and consistency.

## What I learned
1. **Order matters.** Missing codes must become NaN before normalising, and duplicates must go before
   imputing, or the median is biased toward the copied rows.
2. **Not every "missing" value is missing.** `pdays = 999` is a fact ("never contacted"). Turning it into
   a yes/no column keeps the information without inventing contact dates.
3. **Rare ≠ impossible.** A 4,918-second call is rare but real; age 230 is impossible. Only business rules
   delete values; the p1–p99 range only describes the data.
4. **Running the next module tests the previous one.** Re-profiling the cleaned file found a Module 1
   bug (salaries as postcodes), and new tests found two cleaner bugs: a small email column treated
   as a category, and one `"unknown"` turning `age` into text.
5. **Cleaning can create problems too.** Imputing `unknown → no` made 2 different bank clients look
   like duplicates. The log makes such side effects visible.

## Plan for Sprint 5 — ML imputation + outliers
- KNN, regression and iterative imputation (scikit-learn), chosen per column in the config
- Compare with median / mode: hide known values, impute them, measure the error
- Outlier handling (cap or flag), e.g. salary 9,999,999 in the HR data
- Option to keep `unknown` as its own category when it carries meaning (bank `default`: 20.9%)
