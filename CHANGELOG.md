# Changelog

All notable changes to Data Gatekeeper. Format based on [Keep a Changelog](https://keepachangelog.com/);
versions follow the `report_version` of `profiling_report.json`.

## [2.0.0] — 2026-11-26 — Sprint 12: final release (transform by evidence, one-command demo, final docs)

### Changed (behaviour)
- **Skewed columns are only scaled by default** (`src/transform/plan.py`, `POWER = {"default": False, "off": set()}`).
  Decided by a rule committed before measuring (`scripts/ablate_power_transform.py`): power transforms stay on only if
  turning them off hurts LogReg on both Bank and Adult. It hurt Bank (PR-AUC −0.0191, all from `duration`) but helped
  Adult (+0.0322, from `capital-gain`), and trees were unchanged. Opt back in with `POWER["default"] = True`; per-column
  `POWER["off"]`. Effect: on Adult the gatekeeper no longer hurts LogReg (ROC-AUC B − A −0.0079 → −0.0002) and its TPR
  gap for sex is back to raw (0.1053 vs 0.1055); on Bank, LogReg PR-AUC 0.6193 → 0.6004.

### Added
- `make demo` / `make demo-full` / `scripts/demo.sh` (`--full`, `--fixture`, `--local`): Docker image if installed →
  downloads with checksums → pipeline on Bank and Adult → two `report.html`. The CI Docker job runs the fixture demo
  inside the image.
- The Bank downloader checks a pinned zip SHA-256 and writes `bank_marketing.csv.sha256`.
- `scripts/check_docs_commands.py`: README / docs commands parsed or run, links, README numbers vs sprint reports,
  plan status. `scripts/independent_decision_test.py` (UCI Mushroom; SUCCESS by criteria fixed before download).
- `docs/architecture.md`, `docs/final_presentation.md`, `docs/demo_script.md`, `docs/sprint-12.md`; a new README top
  section (demo command, screenshot, key results, diagram). The image ships `pipeline_adult.yaml`.
- Tests 440 → 449.

## [1.10] — 2026-11-26 — Sprint 11: decisions for a person, HTML report, fairness measurement

### Added
- **Decisions for a person** (`src/profiling/decisions.py`, `gatekeeper decisions [--target] [--write]`): "top code or
  sentinel?" and "informative missing?" with counts, label rates in vs rest and a two-proportion test ("clearly
  different" = > 5 points, p < 0.01, ≥ 30 rows; fixed before any run). A `decisions:` block in a contract or config
  (`keep_as_value`, `treat_as_missing`, `keep_as_category`, `impute`) is obeyed by the cleaner and logged with
  `set_by: default | config`. `contracts/adult_decisions.yaml`. Options `decisions` for the pipeline's clean / impact
  steps and `impact --decisions`.
- **Fairness measurement** (`src/impact/fairness.py`, `impact --fairness sex,race`): out-of-fold group rates, TPR, FPR,
  ROC-AUC, Wilson intervals, small-group flags, gaps A vs B. Measurement only.
- **HTML report** (`src/pipeline/html.py`, `gatekeeper report`), written by every `gatekeeper run`; **logging** to the
  console and `run.log`; the profile step writes `decisions.json`.
- **API**: `POST /run` (CSV + JSON config, 120 s limit) and `GET /runs/{id}/report.html`.
- `src/cleaning/imputers.py`: `ITERATIVE` settings and `make_iterative()`; `skip_complete` available as an option.
- Scripts: `evaluate_imputer_speed.py` (rule fixed before measuring), `ablate_adult_transform.py`.
- `docs/sprint-11.md`, `docs/figures/report_adult.png`. Tests 407 → 440.

### Fixed
- The pipeline's clean step dropped the confirmed decisions when an `imputer` option was also set.

### Not changed (on purpose)
- The iterative imputer default: the pre-set rule picked `skip_complete` (same in-table accuracy, Adult fold 349.8 s →
  140.5 s), but the Bank robustness study showed its cost on dirty new rows (HGB, all kinds 20%: 0.8126 → 0.7590).
- The transform default: the Adult ablation shows the power transform causes LogReg's remaining gap
  (C − A −0.0079 → −0.0002 without it); it needs a second dataset before a change.

## [1.9] — 2026-11-26 — Sprint 10: close the known gaps (sentinels by frequency, anomalies, UCI Adult)

### Changed
- **Step 8b judges sentinels by frequency**: a candidate code (999, 9999, −1, −999 + the profiler's numeric sentinels of
  the column) becomes NaN on new rows when `new_share > max(5 × train_share, train_share + 0.005)` (config
  `sentinel_rule`, fixed before any evaluation). `learn_from_train` stores `sentinel_share` instead of `seen_sentinels`;
  every code's decision is logged (`CleaningRun.sentinel_decisions`, `RowCleaner.sentinel_log_`).
  Bank, HGB, sentinels at 20%: ROC-AUC 0.8498 → 0.9141; clean CV unchanged.

### Added
- Anomaly methods `zscore_mixture` (1–3 Gaussian bumps per column by BIC, distance to the nearest bump) and
  `iforest_with_categories` (frequency or one-hot encoding; `gatekeeper anomalies --encoding`). The Sprint 9 methods are
  unchanged. Report: `categorical_features`, `settings.mixture_components`, `mixture_z` in reasons.
- **UCI Adult**: `scripts/download_adult.py` (pinned zip SHA-256, train + test joined, `--split`), `docs/dataset.md`,
  `contracts/adult.yaml`, `pipeline_adult.yaml`, `scripts/diagnose_adult_impact.py` (ablation).
- `docs/sprint-10.md`. Tests 389 → 407.

### Fixed
- Transform: a known ordinal order (Bank's `education`) is only used when ≥ 50% of the values are in it; before, UCI
  Adult's `education` became an empty column.

### Findings
- On Adult the gatekeeper makes both models worse (ROC-AUC B − A: LogReg −0.0093, HGB −0.0053, CIs exclude 0): it
  imputes `capital-gain = 99999` (a top code; all 244 rows are `>50K`) and the informative `?`. Keeping both closes the
  HGB gap. The quality score still rises 99.74 → 100.00.

## [1.8] — 2026-11-25 — Sprint 9: Module 3 complete (sentinel fix, anomalies, one report)
New files: `reports/<dataset>_validation_report.json` (schema `docs/validation_report.schema.json`),
`reports/<dataset>_anomalies.json`. `<dataset>_contract.json` is still written.

### Fixed
- **Unseen sentinels on new rows** (found by the Sprint 8 robustness test). New cleaning step 8b
  `out_of_range_to_missing` (only on rows cleaned with what the train rows taught, `learn_from_train`): values
  outside the train or contract range (`contract_ranges`) and common sentinels (999, 9999, −1, −999) that the
  train column never had become NaN before imputation, logged per column. `FoldImputer` falls back to median /
  mode for rows whose inputs are still out of range (`fallbacks_`). Single-file cleaning is unchanged.
  HGB, all corruption kinds at 20%: ROC-AUC 0.6368 → 0.7508 (raw baseline 0.6906).

### Added
- **Anomaly detection** (`src/validation/anomaly.py`): robust z (1.4826 × MAD), IQR distance, Isolation Forest
  `score_samples`; top-k explanations (3 columns with value, median, robust z; leave-one-column-out for the forest).
  Never uses the label, IDs or PII; rows are reported by position.
- `src/cleaning/outliers.py`: `robust_scale`, `fill_for_forest`, `isolation_forest_model` (shared; `is_anomaly`
  flags unchanged).
- Planted-anomaly demo + evaluation (`src/validation/anomaly_inject.py`, `scripts/make_anomaly_demo.py`,
  `scripts/evaluate_anomalies.py`).
- **One validation report** (`src/validation/report.py`): `gatekeeper validate --reference --anomalies
  --fail-on-anomaly-share`, `gatekeeper anomalies`, `gatekeeper impact --variants`.
- Pipeline step `anomalies` (after quality, before drift; gate only with `fail_on_share`).
- API: `POST /validate`, `POST /drift`, `POST /anomalies` (Sprint 6 upload limits; contract ≤ 256 KB).
- `docs/module3_presentation.md`, `docs/sprint-09.md`. Tests 351 → 389.

### Known gap
- Sentinels are judged by "ever seen in train": `duration` has two real 999s, so injected 999s pass there
  (fold 0, sentinels 20%: 0.8529, or 0.9166 if they were missing). Planned for Sprint 10.

## [1.7] — 2026-11-25 — Sprint 8: downstream impact + one-command pipeline
New files: `reports/<name>_impact.json/.md`, `reports/<name>_robustness.png`,
`reports/run_<UTC time>/summary.json/.md`.

### Added
- **`gatekeeper impact`** (`src/impact/`): A raw / B gatekeeper / C median-imputer with 5-fold stratified CV
  (seed 42), LogisticRegression + HistGradientBoosting, ROC-AUC / PR-AUC / F1 / Brier as mean ± std, paired
  fold differences with 95% t-interval and Wilcoxon; everything fitted on the train rows of each fold only.
- **Robustness test**: seeded injectors (missing, `999` / `?` sentinels, decimal-comma numbers, duplicate
  rows) at 0 / 5 / 10 / 20% on the test folds; table + line chart.
- **`FoldImputer`** and `encode(like=...)` (`src/cleaning/imputers.py`): the Sprint 5 imputers as fit / transform.
- **`gatekeeper run --config pipeline.yaml`** (`src/pipeline/run.py`): profile → validate → clean → mask_pii
  → quality → drift → impact; contract gate exit 1, `--skip STEP`, summary JSON + markdown per run.
- `pipeline.yaml`, `tests/fixtures/pipeline_ci.yaml`, `scripts/diagnose_combined_corruption.py`.
- Docker image ships `pipeline.yaml`; CI runs the fixture pipeline locally and in Docker.
- Tests 297 → 351.

### Changed
- `cleaner.ROW_STEPS` names the row-wise cleaning steps 1–8 (no behaviour change).

### Findings (Bank Marketing)
No ROC-AUC gain from cleaning (LogReg −0.0002, HGB +0.0001, CIs include 0); LogReg PR-AUC +0.0197 and Brier
−0.0013 (CIs exclude 0), the same with a median imputer. With combined corruption the iterative imputer
amplifies unrecognised `999` sentinels (HGB 0.6368 vs 0.6906 at 20%). See `docs/sprint-08.md`.

## [1.6] — 2026-11-25 — Sprint 7: data contracts + drift gate (Module 3 starts)
New files: `contracts/<name>.yaml`, `reports/<dataset>_contract.json`, `reports/<current>_drift.json`.

### Added
- **Data contracts** (`src/validation/contract.py`, `contracts/example.yaml`): per column dtype, required,
  nullable / max_missing_pct, allowed_values, min / max, regex, unique, pii; table min_rows,
  max_duplicate_rows_pct, extra_columns; severity `error` / `warn` per rule.
  `contract init` drafts a contract from a reference file with the profiler (no PII values written).
- **Drift** (`src/validation/drift.py`): PSI over 10 reference-decile bins (+ a missing bin) and the KS
  statistic for numeric columns; PSI over categories with new / vanished categories. Thresholds
  (0.1 / 0.25) configurable under `drift:`. PII columns: no bin edges or categories, unseen values masked.
- **CLI `gatekeeper.py`**: `contract init`, `validate`, `drift`; exit codes 0 pass, 1 fail, 2 bad input.
- `scripts/make_drift_demo.py` (seed 42), `Dockerfile`, CI jobs `contract-gate` and `docker`.
- Tests 219 → 297.

### Demo (Bank Marketing)
Clean split: 97/97 rules pass, 21/21 columns stable. Shifted file: 2 error rules fail, major drift on
exactly `duration` (PSI 8.26), `contact` (0.58), `age` (0.31). See `docs/sprint-07.md`.

## [1.5] — 2026-11-21 — Sprint 6: transformation, PII masking, API (Module 2 complete)
`profiling_report.json` stays at 1.2, `cleaning_log.json` at 1.0 (one new field:
`quality.after.flag_columns`). New files: `transform_report.json`, `transform_pipeline.joblib`,
`pii_report.json`, `masked_data.csv`.

### Fixed
- **Quality score no longer rewards added columns.** The 0/1 flag columns that cleaning adds
  (`<col>_outlier`, `is_anomaly`, `possible_duplicate`, `previously_contacted`, `<col>_missing`) are never
  empty, so they raised completeness without fixing anything, and could change validity and consistency.
  `quality_score(report, flag_columns=...)` now leaves them out of all four dimensions and reports
  them apart (`flag_columns.completeness`). Profiling a raw file is unchanged.

  | "after" score | Sprint 5 (flags counted) | Sprint 6 (flags left out) |
  |---|---|---|
  | Bank Marketing | 98.91 (+0.43) | **98.60 (+0.12)** |
  | HR sample | 99.87 (+11.00) | **99.83 (+10.96)** |

### Added
- **Transformation** (`src/transform/`, `run_transform.py`). Per column, decided on the train rows, with a
  written reason:
  - one-hot for categories (`unknown` kept as its own column, rare values grouped once there are more than 15)
  - ordinal for `education` (values outside the order get an indicator column)
  - log1p or Yeo-Johnson for |skew| > 1 (whichever is more symmetric), skew before and after reported
  - standard, robust (> 5% outliers) or minmax (≤ 10 distinct values) scaling
  - optional bins (`--bins age=18,30,45,60,100`)
  - IDs, PII, dates, free text and mostly-empty columns dropped, with a reason
  - one scikit-learn `ColumnTransformer` **fitted on the train rows only**, saved with joblib, plus `transform_report.json`
- **PII detection and masking** (`src/privacy/`, `run_masking.py`).
  - Detection uses column name + content: email, phone, SSN, credit card (Luhn), IPv4, person name, street address, date of birth, zip code. Each result has a confidence score.
  - Actions per column: `mask` (`j***@g***.com`), `hash` (HMAC-SHA256, salt from env `GATEKEEPER_PII_SALT`, consistent so joins still work), `drop`, `generalize` (DOB → age band, zip → `660**`, IP → `/24`) and `keep`.
  - The report, logs and errors never contain a data value.
- **Synthetic `data/raw/customers_pii.csv`** (300 rows, seeded). Every PII value comes from a reserved fake range. The HR file is byte-identical.
- **API `src/api/app.py`**:
  - Endpoints: `GET /health`, `POST /profile`, `/clean`, `/transform`, `/mask-pii`, with JSON output, or `?output=csv` to download the data.
  - Upload rules: `.csv` only, ≤ 20 MB (checked from `Content-Length` before the body is read), ≤ 200,000 rows, parsed in memory.
  - Errors: clear 400 / 413 / 422 / 503; a plain 500 with no stack trace, and only the error type is logged.
  - `make api` runs it.
- `scripts/make_figures.py` → `docs/figures/` (quality scores, imputer comparison, PII columns).
- `docs/module2_summary.md` (presentation material), `docs/sprint6_summary.md`.
- `pyproject.toml` `[project]` dependencies (incl. `joblib`, `fastapi`, `uvicorn`, `python-multipart`);
  `joblib` in `requirements.txt`; CI runs `ruff check .` before the tests.
- Tests: 167 → 219.

### Changed
- `src/api/profiling_api.py` is now an alias of `src/api/app.py`. The old `uvicorn` command still works, and
  `tests/test_api.py` imports the new module.
- `profiler.sniff_separator(text)` split out of `detect_separator(path)`, so the API never needs a temp file.

## [1.4] — 2026-11-20 — Sprint 5: Module 2 ML-based cleaning
`profiling_report.json` stays at 1.2, `cleaning_log.json` and `schema.json` at 1.0 (new fields only).
New file: `imputer_evaluation.json` 1.0.

### Added
- **ML imputation** (`src/cleaning/imputers.py`): `knn` (k = 5), `regression` (linear / logistic) and
  `iterative` (MICE, 10 rounds) next to the median / mode baseline. Categories are one-hot encoded
  and decoded; numbers are z-scored, clipped to the known range and rounded for whole-number columns.
  Set globally (`numeric_strategy`, `categorical_strategy`, `--imputer`) or per column (`impute`).
  Never imputed or used as features: IDs, text, dates, personal data, the label (`exclude_features: ["y"]`).
- **Imputer evaluation** (`src/cleaning/evaluation.py`, `--evaluate-imputers`): hide 10% of the known
  values (seed 42), refill them with every method, report RMSE, MAE, NRMSE, accuracy and run time;
  `imputer_evaluation.json` + a comparison chart.
- **Outliers** (`src/cleaning/outliers.py`): IQR per column + Isolation Forest per row (`is_anomaly`).
  Actions per column or `--outliers`: `flag` (default, `<col>_outlier`), `cap` (IQR fences), `nan`.
- **Fuzzy duplicates** (`src/cleaning/fuzzy.py`, rapidfuzz): similar names (`token_sort_ratio` ≥ 85)
  **and** ≥ 80% of the other fields equal → `possible_duplicate = 1`; optional merge (`--fuzzy-merge`).
- **`keep_as_category: ["default"]`**: `unknown` in Bank Marketing `default` is kept as its own category
  instead of being imputed; the suggestion is logged as `overridden`, `schema.json` allows `unknown`.
- CLI flags: `--imputer`, `--outliers`, `--fuzzy-threshold`, `--fuzzy-merge`, `--evaluate-imputers`.
- HR sample data: 8 re-typed (near-duplicate) employees appended; the original 515 rows are unchanged.
- `rapidfuzz` in `requirements.txt`. Tests: 110 → 167.

### Changed
- **Default imputer is now `iterative`** (was median / mode), chosen by the evaluation:
  Bank Marketing NRMSE 0.48 vs 1.06, accuracy 67.5% vs 51.8%. Cleaning the full bank file takes
  ≈ 49 s instead of ≈ 2 s; use `--imputer median` for quick runs.
- Flagged outliers are hidden from the imputers while they learn (the output keeps them).
- Personal data (Module 1 PII types, e.g. a `city` column) is never imputed.
- Default fuzzy threshold 85 (finds 8/8 planted pairs on the HR sample, 90 finds 6/8; 0 false pairs).

### Fixed
- Sprint 4: imputing `default = unknown` → `no` made 2 different bank clients identical; now 0 duplicates.
- Iterative imputation of categories always chose the same value (full one-hot columns predicted each
  other); fixed with drop-first encoding (46% → 100% on the test data).
- False "divide by zero in matmul" warnings from numpy 2 + macOS Accelerate are silenced around the
  scikit-learn calls (results checked: finite, same as `einsum`).

## [1.3] — 2026-11-19 — Sprint 4: Module 2 core cleaning pipeline
`profiling_report.json` stays at version 1.2. New files: `cleaning_log.json` 1.0 and `schema.json` 1.0.

### Added
- **Schema inference** (`src/cleaning/schema.py`) → `schema.json`: expected type (numeric,
  categorical, date, text, id), `typical_range` (p1–p99), `valid_range` from business rules
  (age 0–120, salary ≥ 0, …), allowed categories, date / email / phone formats, `unique` for IDs.
- **Cleaning pipeline** (`src/cleaning/cleaner.py`), 10 steps in a fixed order: trim spaces, special
  codes, disguised missing → NaN, normalise categories (case, spaces, `Fin.` → `Finance`), text →
  number, dates → `YYYY-MM-DD`, invalid formats → NaN, impossible values → NaN, drop duplicates,
  median / mode imputation (configurable per column).
- **`pdays = 999` → new column `previously_contacted`** (yes/no); `pdays` is NaN only for the
  never-contacted rows and is not imputed.
- **`cleaning_log.json`**: every step with cells/rows changed and before → after examples, the status
  of every Module 1 suggestion (done / skipped + reason), a summary and the quality score
  before → after with the delta per dimension.
- **CLI `run_cleaning.py`** (`--input`, `--profile`, `--output`, `--impute COL=STRATEGY`); runs
  Module 1 profiling first when no `--profile` is given. The raw file is never modified.
- Tests: 76 → 110 (`tests/test_cleaning_schema.py`, `test_cleaning_steps.py`, `test_cleaning_pipeline.py`).
- Docs: `docs/module2_cleaning.md`, `docs/sprint4_summary.md`; Module 2 sprint plan (Sprints 4–6).

### Fixed
- Module 1: numeric columns of 5-digit numbers (e.g. cleaned salaries `82114`) were detected as US
  postcodes (PII + invalid format). Value patterns for phone and postcode now skip numeric columns;
  a column *named* `zip` / `postcode` is still found.

## [Unreleased] — repo polish before Sprint 4
### Added
- MIT `LICENSE`.
- GitHub Actions CI (`.github/workflows/tests.yml`): `pytest` on every push and pull request (Python 3.11).
- README badges (tests, Python, license), Makefile (`install`, `test`, `lint`, `profile-sample`).
- ruff linter config in `pyproject.toml`; type hints and docstrings are now required in `src/`.
- Tests for the CLI on a tiny CSV and for the API (v1.2 sections, broken CSV, upload limit).
- Issue templates (bug, feature) and a pull request template.

### Fixed
- All ruff findings (unused imports, long lines, `raise ... from e` in the API).
- Histogram crash with numpy 2 on a constant column of huge numbers (e.g. one card number).

## [1.2] — 2026-11-18 — Sprint 3: Module 1 complete
### Added
- **Disguised missing values**: placeholder text (`unknown`, `n/a`, `-`, `?`, blank) and numeric
  sentinels (`999`, `-1`, …) counted only when far outside the real values.
  New column fields `disguised_missing`, `true_missing_count`, `true_missing_pct`; dataset fields
  `disguised_missing_cells`, `true_missing_cells_pct`.
- **Data quality score** 0–100 (`src/quality/score.py`): completeness, uniqueness, validity,
  consistency; `quality_delta()` for Module 2's before/after comparison.
- **`cleaning_suggestions`**: one action code per issue (`impute_median`, `standardise_case`,
  `parse_dates_iso`, `drop_duplicates`, `mask_pii`, …), the hand-off to Module 2.
- Histogram per numeric column and top-10 bar chart per categorical column (`report.charts`).
- `--sample N` CLI flag / `profile_file(sample=N)`; `report.run` with timing and sampling info.
- Module 1 presentation and Sprint 3 summary in `docs/`.

### Changed
- Rule `placeholder_values` replaced by `disguised_missing` (high when ≥ 50% truly missing).
- `dominant_value` is not raised when the dominant value is a known missing code (e.g. `pdays = 999`).
- Numeric stats, correlation and the outlier chart ignore sentinels (`pdays` mean 962 → 6.0).
- Profiling is ~2× faster (41,188 rows: 5.1 s → 2.5 s) by checking value types per column, not per value.

### Fixed
- Outlier rule when IQR = 0: no longer flags every non-mode value (e.g. `previous`).

## [1.1] — 2026-11-17 — Sprint 2: real data + API
### Added
- Real finance dataset: UCI Bank Marketing (41,188 × 21), download script, 1,000-row sample, `docs/dataset.md`.
- `profile_file(path)` with automatic separator detection (`,` `;` tab `|`).
- FastAPI app: `POST /profile` (CSV upload → JSON report) and `GET /health`.
- Histograms and top categories with percentages; Cramér's V between categorical columns.
- Stronger PII detection: credit cards (Luhn checksum), IPv4 addresses, date of birth.
- Suspicious-column rules: `id_stored_as_number`, `dominant_value`, `placeholder_values`.
- JSON Schema for the report (`docs/profiling_report.schema.json`) and schema tests (6 → 44 tests).

### Fixed
- `cons.price.idx` no longer detected as an ID (ID keywords matched as whole words).

## [1.0] — 2026-11-16 — Sprint 1: project setup + profiler v1
### Added
- Repository structure (`src/profiling`, `src/cleaning`, `src/validation`, `tests`, `docs`).
- Messy synthetic HR dataset generator (`scripts/make_sample_data.py`).
- Profiler v1: basic and semantic types (email, phone, date, ID), missing values, IQR outliers,
  Pearson correlation, profiling rule engine.
- CLI `run_profiling.py` with missing, correlation and outlier charts.
- First 6 unit tests and `docs/module1_profiling.md`.
