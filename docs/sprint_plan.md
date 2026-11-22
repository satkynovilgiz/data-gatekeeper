# Sprint Plan — Data Gatekeeper (12 sprints, ~1 per week)

```
Sprints 1–3   Module 1  Profile     -> profiling_report.json
Sprints 4–6   Module 2  Clean       -> cleaned_data.csv + cleaning_log.json
Sprints 7–9   Module 3  Validate    -> validation_report.json (+ anomaly detection)
Sprints 10–12 Module 4  Automate    -> one CLI pipeline, YAML config, Docker, CI, demo
```

Status: ✅ done · 🔄 in progress · ⏳ planned

---

## Module 1 — Profile

### Sprint 1 ✅ — Project setup + profiler v1
- Repo structure (`src/profiling`, `src/cleaning`, `src/validation`, `tests`, `docs`)
- Messy sample HR dataset generator (`scripts/make_sample_data.py`)
- Profiler v1: types, semantic types (email, phone, date, ID), missing values, outliers, correlations, rule engine
- CLI `run_profiling.py` + charts (missing heatmap, correlation heatmap, outliers)
- 6 unit tests + `docs/module1_profiling.md`

**Deliverable:** `python run_profiling.py --input data/raw/hr_employees_messy.csv` produces `profiling_report.json` and charts.

### Sprint 2 ✅ — Real dataset + profiling API
- Pick a real public finance dataset (UCI Bank Marketing) + download script + `docs/dataset.md`
- `profile_file(path)` function + FastAPI `POST /profile` endpoint
- Better profiling: histograms, top categories, Cramér's V, stronger PII detection, suspicious-column rules
- Real JSON Schema for the report + a test that validates against it
- Tests for every new feature, updated docs

**Deliverable:** the profiler runs on the real dataset (CLI and API); `pytest -q` is green; `docs/sprint2_summary.md`.

### Sprint 3 ✅ — Polish Module 1 + hand-off to Module 2
- ✅ Speed: vectorised the per-value type check (41k rows: 5.1 s → 2.5 s) + `--sample N` for big files
- ✅ Disguised missing values: placeholder text (`unknown`, `?`, `-`) + numeric sentinels (`999`, `-1`), `true_missing_pct`
- ✅ Fixed the outlier rule when IQR = 0; stats/correlation ignore sentinels
- ✅ Histogram per numeric column, top-10 bar chart per categorical column
- ✅ Data quality score 0–100 (`src/quality/score.py`) — reused by Module 2 for the before/after delta
- ✅ `cleaning_suggestions` with action codes = the input for Module 2
- ✅ Module 1 presentation (`docs/module1_presentation.md`) + `docs/sprint3_summary.md`
- ➡️ Moved: one-page HTML report → Module 4 (Sprint 12 demo); full type-hint pass → ongoing

**Deliverable:** Module 1 v1.2 + presentation; the report is the "to-do list" input for Module 2.

---

## Module 2 — Clean

Module 2 = **Advanced Data Cleaning & Transformation**.
Sprint 4 builds the core pipeline, Sprint 5 adds ML imputation, Sprint 6 adds transformation,
the cleaning API and the final tests + presentation.

### Sprint 4 ✅ — Core cleaning pipeline
- Schema inference engine (`src/cleaning/schema.py`) → `schema.json`: expected type
  (numeric, categorical, date, text, id), expected range (p1–p99 + business rules like age 0–120),
  expected format (YYYY-MM-DD, allowed categories). Module 3 will validate against it.
- Cleaning pipeline (`src/cleaning/cleaner.py`) driven by `cleaning_suggestions`:
  disguised missing → NaN, `pdays = 999` → `previously_contacted` (not imputed),
  trim + normalise categories, parse dates to ISO, text → numbers, invalid formats → NaN,
  impossible values → NaN, drop duplicates, median / mode imputation (configurable per column)
- `cleaning_log.json`: every step with counts and before → after examples, summary,
  quality score before → after + delta (reuses `src/quality/score.py`)
- CLI `run_cleaning.py` (runs Module 1 first when no `--profile` is given)
- Tests for every step, the raw file is never modified; results on both datasets

**Deliverable:** `cleaned_data.csv` + `cleaning_log.json` + `schema.json` for Bank Marketing and HR.
**Result:** quality score Bank 98.48 → 98.90, HR 88.86 → 99.83; 110 tests. See `docs/sprint4_summary.md`.

### Sprint 5 ✅ — ML-based cleaning
- ✅ `unknown` kept as its own category where it carries meaning (`keep_as_category: ["default"]`);
  the 2 identical rows from Sprint 4 are gone
- ✅ KNN, regression and iterative (MICE) imputation (scikit-learn), per column or globally;
  categories encoded / decoded; IDs and PII never imputed; the label `y` never used as a feature
- ✅ Imputer evaluation: hide 10% of known values, RMSE / MAE / accuracy / time, JSON + chart →
  `iterative` is the new default
- ✅ Outliers: IQR (per column) + Isolation Forest (per row); flag (default) / cap / nan;
  flagged outliers hidden from the imputers
- ✅ Fuzzy duplicates with rapidfuzz (CadetX brief): flag by default, optional merge
- ✅ CLI flags `--imputer`, `--outliers`, `--fuzzy-threshold`, `--fuzzy-merge`, `--evaluate-imputers`

**Deliverable:** the pipeline can choose an imputer per column, with an accuracy comparison.
**Result:** Bank iterative NRMSE 0.48 vs median 1.06; HR 8/8 fuzzy pairs found; quality Bank 98.48 → 98.91,
HR 88.87 → 99.87; 167 tests. See `docs/sprint5_summary.md`.

### Sprint 6 ✅ — Transformation + PII masking + API + final
- ✅ Quality score fix: flag columns added by cleaning are scored apart
- ✅ Scaling (standard, robust, min-max) and encoding (one-hot, ordinal) per column with a reason;
  log / Yeo-Johnson for skewed columns; optional bins; fitted on train only, saved with joblib
- ✅ PII detection (9 types) + mask / salted hash / drop / generalize, PII report without values
- ✅ API: `/health`, `/profile`, `/clean`, `/transform`, `/mask-pii` with upload limits
- ✅ Module 2 summary + charts, tests 167 → 219, ruff in CI
- ➡️ Moved to Sprint 7+: faster iterative imputation on big files, optional `--imputer auto`,
  date features (year / month) in the transform

**Deliverable:** Module 2 complete: raw → clean → ML-ready, by CLI and API.
**Result:** Bank quality 98.48 → 98.60 and HR 88.87 → 99.83 (honest score); Bank 26 → 63 model columns,
fitted on 32,940 train rows only; 9/9 fake PII columns found, 0 in Bank; 219 tests. See `docs/sprint6_summary.md`.

---

## Module 3 — Validate + anomaly detection

### Sprint 7 ⏳ — Rule-based validation
- Define expectations (not null, in range, allowed values, unique, regex)
- Run them on the cleaned data → pass/fail per rule
- `validation_report.json` + overall quality score

**Deliverable:** rule checks with a JSON report.

### Sprint 8 ⏳ — Anomaly detection (ML)
- Isolation Forest on numeric columns (scikit-learn)
- Z-score / IQR as a simple baseline to compare
- Explain the top anomalies (which columns made them unusual)

**Deliverable:** anomaly section in `validation_report.json`.

### Sprint 9 ⏳ — Schema + drift checks
- Auto-generate expectations from a "good" dataset
- Detect drift between two versions of a dataset (new categories, shifted means)
- Module 3 docs + presentation

**Deliverable:** Module 3 complete: clean data gets a pass/fail and an anomaly list.

---

## Module 4 — Automate

### Sprint 10 ⏳ — One pipeline
- `pipeline.py --input ... --output ...` runs Profile → Clean → Validate
- YAML config file for thresholds and cleaning options
- Logging (Python `logging`) to console + file

**Deliverable:** one command runs the whole system.

### Sprint 11 ⏳ — Docker + CI
- `Dockerfile` (CLI + API)
- GitHub Actions: run `pytest` on every push / pull request
- API endpoint for the full pipeline

**Deliverable:** `docker run data-gatekeeper --input data.csv` works; CI badge is green.

### Sprint 12 ⏳ — Final demo
- End-to-end demo on the real dataset
- Final README, architecture diagram, docs check
- Final presentation + recorded demo

**Deliverable:** final presentation and a working, documented system.
