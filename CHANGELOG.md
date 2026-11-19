# Changelog

All notable changes to Data Gatekeeper. Format based on [Keep a Changelog](https://keepachangelog.com/);
versions follow the `report_version` of `profiling_report.json`.

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
