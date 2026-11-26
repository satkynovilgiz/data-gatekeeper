# Sprint Plan — Data Gatekeeper (12 sprints, ~1 per week)

```
Sprints 1–3   Module 1  Profile     -> profiling_report.json
Sprints 4–6   Module 2  Clean       -> cleaned_data.csv + cleaning_log.json
Sprints 7–9   Module 3  Validate    -> validation_report.json (contract + drift + anomalies), impact
Sprints 10–12 Module 4  Finish      -> sentinel-by-frequency, HTML report, POST /run, final demo
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

## Module 3 — Validate + anomaly detection ✅ (complete, Sprint 9)

### Sprint 7 ✅ — Data contracts + drift gate
- ✅ YAML data contracts: dtype, required, nullable / max_missing_pct, allowed values, min / max, regex,
  unique; table rules; severity error / warn per rule; `contract init` drafts one from a reference file
- ✅ `gatekeeper validate` → pass/fail table + `reports/<dataset>_contract.json`, exit codes 0 / 1 / 2
- ✅ Drift pulled forward from Sprint 9: PSI + KS per column, new categories, PII never shown
- ✅ Dockerfile + CI contract gate pulled forward from Sprint 11

**Result:** clean Bank split passes 97/97 rules, 21/21 columns stable; the shifted file fails 2 rules
with major drift on exactly the 3 changed columns; 297 tests. See `docs/sprint-07.md`.

### Sprint 8 ✅ — Downstream impact + one-command pipeline
- ✅ `gatekeeper impact`: A raw / B gatekeeper / C median imputer, 5-fold CV, two models, paired CIs,
  fitted per fold only (a spy test proves it)
- ✅ Robustness to dirty data: seeded corruption at 0 / 5 / 10 / 20%, table + chart
- ✅ `gatekeeper run --config pipeline.yaml` (pulled forward from Sprint 10), Docker + CI
- ➡️ Moved to Sprint 9: anomaly detection section (Isolation Forest explanations), the sentinel fix
  found by the robustness test, per-step ablation

**Result:** no ROC-AUC gain on Bank Marketing; LogReg PR-AUC +0.0197 [+0.0108, +0.0285]; much more robust to
missing / malformed values, but unrecognised `999` sentinels hurt with the iterative imputer; 351 tests.
See `docs/sprint-08.md`.

### Sprint 9 ✅ — Finish Module 3: sentinel fix, anomalies, one validation report
(Schema checks and drift were already done in Sprint 7, so this sprint did the rest of Module 3.)
- ✅ **Sentinel fix**: step 8b `out_of_range_to_missing` on new rows (train / contract range + unseen
  common sentinels) and a median / mode fallback in `FoldImputer`; robustness study re-run before / after
- ✅ **Anomaly detection** (`src/validation/anomaly.py`, reusing `src/cleaning/outliers.py`): robust z, IQR,
  Isolation Forest, top-k explanations (leave-one-column-out); evaluation on 1% planted anomalies of
  3 kinds with settings committed before the first run
- ✅ **One report**: `validation_report.json` + JSON Schema, `gatekeeper anomalies`,
  `validate --reference --anomalies`, pipeline step `anomalies`, API `/validate /drift /anomalies`
- ✅ **Wrap-up**: `docs/module3_presentation.md`, `docs/sprint-09.md`
- ➡️ Moved to Sprint 10: sentinel rule by frequency (the fix lets `duration = 999` through because
  the train rows contain two real 999s); categories in the forest

**Result:** HGB with all corruption kinds at 20%: 0.6368 → 0.7508 (A: 0.6906), clip-edge fills 72–74% → 0–4.5%;
anomalies: forest ROC-AUC 0.84 overall, but precision@k ≤ 0.21 for every method on real data; 389 tests.
See `docs/sprint-09.md` and `docs/module3_presentation.md`.

---

## Module 4 — Finish (what is really left)
The one-command pipeline, YAML config, Docker and CI were pulled forward into Sprints 7–8, so Module 4 is about
closing the known gaps and presenting the system.

### Sprint 10 ✅ — Close the known gaps
- ✅ Sentinel rule by frequency (`new_share > max(5 × train_share, train_share + 0.005)`), logged per code;
  robustness re-run Sprint 8 → 9 → 10
- ✅ `zscore_mixture` (1–3 bumps by BIC) and `iforest_with_categories` next to the Sprint 9 methods
- ✅ UCI Adult: downloader with a pinned checksum, contract, pipeline, impact + robustness, ablation
- ✅ Fixed: the Bank ordinal order for `education` wiped out Adult's `education`
- ➡️ Moved to Sprint 11: top codes vs sentinels and informative `?` made visible as decisions; faster iterative
  imputation; a first fairness check on Adult; transform ablation for LogReg on Adult

**Result:** HGB sentinels 20% 0.8264 → 0.9141, clean data unchanged; anomaly precision@k 0.10 → 0.59 (mixture z);
on Adult the gatekeeper **hurts** (LogReg ROC-AUC −0.0093, HGB −0.0053), explained by `capital-gain = 99999` and `?`
being imputed, while it is far more robust on dirty data; 407 tests. See `docs/sprint-10.md`.

### Sprint 11 🔄 — Decisions a person can see + a report people can read
- 🔄 Decisions for a person: "top code or sentinel?" and "informative missing?" with evidence (label rates),
  set per column in the contract / config, obeyed and logged by the cleaner; `gatekeeper decisions`
- 🔄 Faster iterative imputation, chosen by speed AND the Sprint 5 accuracy evaluation
- 🔄 A first fairness check on Adult (measurement only): group rates and gaps, A vs B
- 🔄 `gatekeeper report` → one self-contained HTML page; `run` writes it; logging to run.log; `POST /run`
- 🔄 (if time) LogReg transform ablation on Adult

**Deliverable:** a person sees the uncertain decisions and one readable page per run.

### Sprint 12 ⏳ — Final demo
- End-to-end demo on the real dataset (Docker image, one command)
- Final README, architecture diagram, docs check
- Final presentation + recorded demo

**Deliverable:** final presentation and a working, documented system.
