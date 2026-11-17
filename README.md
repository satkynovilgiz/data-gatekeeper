# Data Gatekeeper — Automated Data Cleaning & Validation System

A backend engine that **profiles, cleans, and validates** any dataset before it is used.
Built by a team of 3 for the CadetX AI Engineering & Data Science programme.

```
raw.csv ──► Module 1: Profile ──► Module 2: Clean ──► Module 3: Validate ──► Module 4: Automate
            profiling_report.json   cleaned_data.csv    validation_report.json   CLI + Docker
                                    cleaning_log.json
```

| Module | Status | Output |
|---|---|---|
| 1. Profiling & Metadata Intelligence | ✅ v1 working | `profiling_report.json` + charts |
| 2. Cleaning & Transformation | ⏳ next | `cleaned_data.csv`, `cleaning_log.json` |
| 3. AI Validation & Anomaly Detection | ⏳ | `validation_report.json` |
| 4. Integration, CLI & Docker | ⏳ | `python pipeline.py --input ... --output ...` |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/make_sample_data.py           # creates a messy test dataset
python run_profiling.py --input data/raw/hr_employees_messy.csv --output results/
pytest -q                                     # run the tests
```

Output:
```
results/
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
├── docs/                     # module documentation (JSON schema, how to use)
├── scripts/                  # helper scripts (sample data generator)
├── src/
│   ├── profiling/            # Module 1
│   │   ├── semantic.py       #   type + meaning detection (email, phone, date, id, PII)
│   │   ├── profiler.py       #   profiling engine + rule engine -> report dict
│   │   └── visualize.py      #   backend charts (PNG)
│   ├── cleaning/             # Module 2 (to build)
│   └── validation/           # Module 3 (to build)
├── tests/                    # pytest
├── run_profiling.py          # Module 1 CLI
└── requirements.txt
```

## Team workflow
- Each person works on their **own branch** → opens a **pull request** → a teammate **reviews** → merge to `main`.
- Weekly meeting + rotating Scrum Master (see `docs/team.md`).
- Each module gets its own short presentation / doc in `docs/`.
