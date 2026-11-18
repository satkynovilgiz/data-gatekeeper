# Sprint 2 Summary — Module 1 (Profile) on real data

**Goal:** take the Sprint 1 profiler from synthetic data to a real finance dataset, and make it usable as an API.

## What I built
| Feature | Where |
|---|---|
| Real dataset: UCI Bank Marketing (41,188 rows × 21 cols, CC BY 4.0), download script + 1k sample | `scripts/download_bank_marketing.py`, `docs/dataset.md` |
| `profile_file(path)` with automatic separator detection (`,` `;` tab) | `src/profiling/profiler.py` |
| FastAPI app: `POST /profile` (CSV upload → JSON report), `GET /health` | `src/api/profiling_api.py` |
| Histograms (numeric) + top categories with % (categorical) | `profiler.py` |
| Cramér's V for categorical pairs (pure numpy) | `profiler.py` |
| PII: credit cards (with Luhn checksum), IPv4 addresses, date of birth | `src/profiling/semantic.py` |
| Suspicious-column rules: ID stored as number, one value > 90%, placeholder text (`unknown`, `?`) | `profiler.py` |
| Real JSON Schema for the report + validation tests | `docs/profiling_report.schema.json` |
| Tests: **6 → 44**, all passing | `tests/` |

## Results on the real dataset (41,188 rows)
- **Missing cells: 0.0%** by pandas, but **12,718 hidden missing values** written as `unknown` (1.47% of all cells) in 6 columns. The worst is `default`, with **8,597 rows (20.9%)** unknown.
- **12 exact duplicate rows** (the only high-severity issue).
- **`pdays = 999` in 96.3% of rows**: a "magic number" meaning "never contacted", flagged by the new `dominant_value` rule.
- **7 suspicious columns:** `default`, `education`, `housing`, `job`, `loan`, `marital`, `pdays`.
- **PII columns: none.** This is correct, because the data is anonymised.
- **Correlations:** `emp.var.rate`, `euribor3m` and `nr.employed` are almost the same signal (Pearson r = 0.91–0.97). Cramér's V: `housing`–`loan` = 0.71, because the same 990 clients are `unknown` in both.
- **Imbalanced target:** only 11.3% `yes`.
- Total: **14 issues** (1 high, 13 medium). The report passes the JSON Schema; profiling takes ~10 s.

## What I learned
1. **"0% missing" can be a lie.** Real data hides missing values as text (`unknown`) or magic numbers (`999`).
2. **Detection by name alone is fragile.** `cons.price.idx` was tagged as an ID because it contains "id"; matching whole words fixed it. A test now protects that fix.
3. **A checksum beats a regex.** The Luhn check separates real card numbers from random 16-digit numbers.
4. **A schema is a contract.** It lets Module 2 trust the report, and tests catch accidental changes.
5. **Statistics have blind spots.** When one value dominates, IQR = 0 and every other value looks like an "outlier".

## Plan for Sprint 3 (finish Module 1)
- Speed up profiling (vectorised type check, sampling for big files)
- Count placeholders as missing; detect numeric magic numbers (`999`, `-1`)
- Fix the outlier rule when IQR = 0
- One-page HTML report built from the JSON
- Module 1 presentation, and hand-off to Module 2 (Clean)
