# Sprint 5 Summary — Module 2 part 2: ML-based cleaning

**Goal:** fill missing values with machine learning and prove with numbers which method is best;
handle outliers and fuzzy duplicates; fix the Sprint 4 review point that `unknown` can be meaningful.

## What I built
| Feature | Where |
|---|---|
| `unknown` kept as its own category in `default` (configurable `keep_as_category`); 0 duplicate rows now | `src/cleaning/cleaner.py` |
| ML imputers: KNN (k = 5), regression (linear / logistic), iterative (MICE); one-hot + z-score encoding; never IDs, PII or the label `y` | `src/cleaning/imputers.py` |
| Imputer evaluation: hide 10% of known values, refill, RMSE / MAE / NRMSE / accuracy / time → JSON + chart | `src/cleaning/evaluation.py` |
| Outliers: IQR per column + Isolation Forest per row; flag (default) / cap / nan; outliers hidden from the imputers | `src/cleaning/outliers.py` |
| Fuzzy duplicates with rapidfuzz: similar name **and** ≥ 80% equal fields; flag, optional merge | `src/cleaning/fuzzy.py` |
| CLI: `--imputer`, `--outliers`, `--fuzzy-threshold`, `--fuzzy-merge`, `--evaluate-imputers` | `run_cleaning.py` |
| Tests **110 → 167**, all passing; ruff clean | `tests/` |

## Results
**Imputer evaluation, Bank Marketing** (9,000 hidden cells in 5,000 complete rows):

| | median / mode | knn | regression | **iterative** |
|---|---|---|---|---|
| numeric error (mean NRMSE, 1.0 = guessing the average) ↓ | 1.063 | 0.563 | 0.561 | **0.478** |
| categorical accuracy ↑ | 51.8% | 62.4% | 67.2% | **67.5%** |
| time | 0.02 s | 3.2 s | 2.0 s | 9.2 s |

On the HR sample every method scores ≈ 0.93: its columns are generated independently, so there is
nothing to learn. → **Default: iterative** (it wins where columns are related, and doesn't lose where they aren't).

| | Bank Marketing (41,188 × 21) | HR messy sample (523 × 10) |
|---|---|---|
| Rows before → after | 41,188 → 41,176 | 523 → 508 |
| Fuzzy duplicate pairs | skipped (no name column) | 8 flagged (8/8 planted, 0 false) |
| Outliers (IQR cells / Isolation Forest rows / both) | 6,283 / 412 / 226, flagged | 7 / 6 / 5, flagged |
| `unknown` kept as category | 8,597 (`default`) | – |
| Values imputed (iterative) | 4,120 | 89 |
| **Quality score** | **98.48 → 98.91 (+0.43)** | **88.87 → 99.87 (+11.00)** |
| Run time (whole CLI) | 59 s (13 s with `--imputer median`) | 4.2 s with evaluation |

## What I learned
1. **ML imputation only helps when columns are related.** The bank's economic indicators move together
   (euribor3m error 1.26 → 0.10); in the random HR data no method beats the median.
2. **Measure, don't assume.** The evaluation found a bug I could not see: MICE predicted one-hot columns
   from each other in a circle and always chose the same category (46% → 100% after drop-first encoding).
3. **Outliers poison the models.** A few salaries of 9,999,999 made MICE fill missing salaries with
   ≈ 187,000. Hiding flagged outliers while the imputers learn brought this back to ≈ 81,000. Median and
   KNN were robust; linear models were not.
4. **A name match is not enough.** "Omar Garcia" appears 11 times as 11 people. Requiring the other fields
   to agree gave 0 false pairs at every threshold, so the threshold only controls how many typos we catch.
5. **Better accuracy has a price.** Iterative takes ≈ 49 s on 41k rows instead of ≈ 2 s.

## Plan for Sprint 6 — transformation + cleaning API + final
- Scaling (standard, min-max, robust) and encoding (one-hot, ordinal) for ML-ready output
- Mask or hash PII (`full_name`, `email`, `phone`)
- Cleaning API `POST /clean`; faster iterative imputation on big files; optional `--imputer auto`
- Final tests, Module 2 documentation and presentation
