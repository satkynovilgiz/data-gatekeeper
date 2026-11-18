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

### Sprint 2 🔄 — Real dataset + profiling API
- Pick a real public finance dataset (UCI Bank Marketing) + download script + `docs/dataset.md`
- `profile_file(path)` function + FastAPI `POST /profile` endpoint
- Better profiling: histograms, top categories, Cramér's V, stronger PII detection, suspicious-column rules
- Real JSON Schema for the report + a test that validates against it
- Tests for every new feature, updated docs

**Deliverable:** the profiler runs on the real dataset (CLI and API); `pytest -q` is green; `docs/sprint2_summary.md`.

### Sprint 3 ⏳ — Polish Module 1 + hand-off to Module 2
- Speed: vectorise the per-value type check (41k rows takes ~10 s now) + sampling for big files
- Hidden missing values: count placeholders (`unknown`) in `missing_pct`; detect numeric magic numbers (`999`, `-1`)
- Fix outlier rule when IQR = 0 (e.g. `pdays`, `previous` flag every non-mode value)
- Simple HTML report (one page) built from the JSON
- Clean up the code, add type hints and docstrings everywhere
- Module 1 presentation

**Deliverable:** Module 1 v2 + presentation; the report is the "to-do list" input for Module 2.

---

## Module 2 — Clean

### Sprint 4 ⏳ — Cleaning basics
- Read `profiling_report.json` and decide actions per column
- Fix types (text → number / date), trim spaces, normalise case
- Remove exact duplicate rows
- Write `cleaning_log.json` (every change: column, action, rows affected)

**Deliverable:** `cleaned_data.csv` + `cleaning_log.json` for the sample dataset.

### Sprint 5 ⏳ — Missing values + outliers
- Imputation strategies: median (numeric), mode (categorical), "unknown" flag
- Outlier handling: cap (winsorise) or flag, configurable
- Standardise date formats to ISO `YYYY-MM-DD`
- Tests for each cleaning step

**Deliverable:** cleaning engine handles missing values and outliers with a log of every change.

### Sprint 6 ⏳ — PII protection + before/after report
- Mask or hash PII columns found by Module 1
- Before/after comparison (profile the cleaned data again)
- Cleaning options in a small config dict
- Module 2 docs + presentation

**Deliverable:** Module 2 complete: raw → clean with a before/after quality score.

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
