# Data Gatekeeper — Automated Data Cleaning & Validation System

A backend engine that **profiles, cleans, and validates** any dataset before it is used.
CadetX AI Engineering & Data Science internship project (12 sprints, see [`docs/sprint_plan.md`](docs/sprint_plan.md)).

```
raw.csv ──► Module 1: Profile ──► Module 2: Clean ──► Module 3: Validate ──► Module 4: Automate
            profiling_report.json   cleaned_data.csv    validation_report.json   CLI + Docker
                                    cleaning_log.json
```

| Module | Status | Output |
|---|---|---|
| 1. Profiling & Metadata Intelligence | ✅ v1.1 (Sprint 2): CLI + API + JSON Schema | `profiling_report.json` + charts |
| 2. Cleaning & Transformation | ⏳ next | `cleaned_data.csv`, `cleaning_log.json` |
| 3. AI Validation & Anomaly Detection | ⏳ | `validation_report.json` |
| 4. Integration, CLI & Docker | ⏳ | `python pipeline.py --input ... --output ...` |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_bank_marketing.py    # real finance dataset (UCI Bank Marketing, 41k rows)
python run_profiling.py --input data/raw/bank_marketing.csv --output results/bank
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
├── profiling_report.json
└── charts/
    ├── missing_heatmap.png
    ├── correlation_heatmap.png
    └── outlier_distribution.png
```

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
│   │   └── visualize.py      #   backend charts (PNG)
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
