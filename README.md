# Data Gatekeeper — Automated Data Cleaning & Validation System

[![tests](https://github.com/satkynovilgiz/data-gatekeeper/actions/workflows/tests.yml/badge.svg)](https://github.com/satkynovilgiz/data-gatekeeper/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A backend engine that **profiles, cleans, and validates** any dataset before it is used.
CadetX AI Engineering & Data Science internship project (12 sprints, see [`docs/sprint_plan.md`](docs/sprint_plan.md)).

```
raw.csv ──► Module 1: Profile ──► Module 2: Clean ──► Module 3: Validate ──► Module 4: Automate
            profiling_report.json   cleaned_data.csv    validation_report.json   CLI + Docker
                                    cleaning_log.json
```

| Module | Status | Output |
|---|---|---|
| 1. Profiling & Metadata Intelligence | ✅ **complete** (v1.2, Sprint 3): CLI + API + JSON Schema, disguised missing, quality score, cleaning suggestions | `profiling_report.json` + charts |
| 2. Cleaning & Transformation | ⏳ next | `cleaned_data.csv`, `cleaning_log.json` |
| 3. AI Validation & Anomaly Detection | ⏳ | `validation_report.json` |
| 4. Integration, CLI & Docker | ⏳ | `python pipeline.py --input ... --output ...` |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_bank_marketing.py    # real finance dataset (UCI Bank Marketing, 41k rows)
python run_profiling.py --input data/raw/bank_marketing.csv --output results/bank   # ~3.5 s
python run_profiling.py --input data/raw/bank_marketing.csv --output results/quick --sample 5000
pytest -q                                     # run the tests
```

No internet? Use the committed 1,000-row sample `data/raw/bank_marketing_sample.csv`,
or the synthetic HR data (`python scripts/make_sample_data.py`). Dataset details: [`docs/dataset.md`](docs/dataset.md).

### Profiling API
```bash
uvicorn src.api.profiling_api:app --reload        # http://127.0.0.1:8000/docs
curl -F "file=@data/raw/bank_marketing_sample.csv" \
     http://127.0.0.1:8000/profile -o profiling_report.json
```
The report format is a JSON Schema: [`docs/profiling_report.schema.json`](docs/profiling_report.schema.json).

CLI output:
```
results/bank/
├── profiling_report.json     # stats, issues, quality_score, cleaning_suggestions
└── charts/
    ├── missing_heatmap.png
    ├── correlation_heatmap.png
    ├── outlier_distribution.png
    ├── hist_<column>.png     # one per numeric column
    └── bar_<column>.png      # one per categorical column (top 10)
```

### Module 1 results on Bank Marketing (41,188 rows × 21 columns)
| | |
|---|---|
| Empty cells (what pandas sees) | 0.0% |
| **Disguised missing** (`unknown`, `pdays = 999`) | **52,391 cells → true missing 6.06%** in 7 columns |
| Duplicate rows | 12 |
| PII columns | none (anonymised data) |
| Issues | 13 (2 high, 11 medium) → 13 cleaning suggestions |
| **Quality score** | **98.48 / 100** (completeness 93.94, uniqueness 99.97, validity 100, consistency 100) |
| Run time | 2.5 s profiling, 3.5 s with charts |

Details: [`docs/module1_profiling.md`](docs/module1_profiling.md) · slides: [`docs/module1_presentation.md`](docs/module1_presentation.md)

## Project structure

```
data-gatekeeper/
├── data/raw/                 # input datasets (keep large files out of git — use DVC / Git LFS)
├── docs/                     # sprint plan, dataset, module docs, JSON schema
├── scripts/                  # dataset download + sample data generator
├── src/
│   ├── api/
│   │   └── profiling_api.py  #   FastAPI: POST /profile
│   ├── profiling/            # Module 1
│   │   ├── semantic.py       #   type + meaning detection (email, phone, card, IP, DOB, id, PII)
│   │   ├── profiler.py       #   profile_file / profile_dataset + rule engine -> report dict
│   │   ├── suggestions.py    #   issues -> cleaning suggestions (hand-off to Module 2)
│   │   └── visualize.py      #   backend charts (PNG)
│   ├── quality/
│   │   └── score.py          #   data quality score 0-100 (shared by Module 1 and 2)
│   ├── cleaning/             # Module 2 (to build)
│   └── validation/           # Module 3 (to build)
├── tests/                    # pytest
├── run_profiling.py          # Module 1 CLI
└── requirements.txt
```

## Workflow
- Work on a **branch per sprint/feature** → small commits → **pull request** → review → merge to `main`.
- Weekly meeting + rotating Scrum Master (see `docs/team.md`).
- Each module gets its own short presentation / doc in `docs/`.
