# Architecture — Data Gatekeeper v2.0

One CSV goes in; a person gets a clear answer (can this file be used?), a cleaned file, and one
readable page. Every step is an ordinary Python function in `src/`, reached through four doors
(CLI, pipeline YAML, HTTP API, Docker) that all call the same code.

## The flow of one run

```mermaid
flowchart TD
    CSV[/"CSV file"/] --> P["1 · profile<br/>src/profiling/profiler.py<br/>types, missing, disguised missing, PII hints"]
    P --> D["decisions for a person<br/>src/profiling/decisions.py<br/>top code or sentinel? informative missing?"]
    P --> V{"2 · contract validate<br/>src/validation/contract.py"}
    V -- "an error rule fails" --> STOP["exit 1: stopped at the gate"]
    V -- passes --> C["3 · clean<br/>src/cleaning/cleaner.py (+ imputers, outliers, fuzzy, schema)<br/>obeys the confirmed decisions"]
    D -. "decisions: in contract / config" .-> C
    C --> M["4 · PII mask<br/>src/privacy/detect.py, mask.py"]
    M --> Q["5 · quality score<br/>src/quality/score.py"]
    Q --> A["6 · anomalies<br/>src/validation/anomaly.py<br/>robust z, mixture z, IQR, Isolation Forest + reasons"]
    A --> DR["7 · drift vs a reference<br/>src/validation/drift.py (PSI + KS)"]
    DR --> I["8 · impact (optional)<br/>src/impact/impact.py, fairness.py, corruption.py<br/>raw vs gatekeeper, 5-fold CV, robustness, group gaps"]
    I --> OUT
    STOP --> OUT
    subgraph OUT ["outputs in reports/run_&lt;time&gt;/"]
        R1["report.html<br/>src/pipeline/html.py"]
        R2["summary.json / summary.md · run.log"]
        R3["validation_report.json (gatekeeper validate)<br/>src/validation/report.py"]
        R4["cleaned_data.csv · cleaning_log.json · schema.json<br/>masked_data.csv · decisions.json"]
    end
```

The model-ready transform (`src/transform/plan.py`, `pipeline.py`: encoders, scalers, power transforms,
fitted on train rows only) is used by `run_transform.py`, `POST /transform` and inside the impact study.

## Entry points (the four doors)

```mermaid
flowchart LR
    CLI["gatekeeper.py<br/>contract init · validate · drift · anomalies<br/>decisions · impact · run · report"] --> CORE
    YAML["pipeline.yaml / pipeline_adult.yaml<br/>gatekeeper run --config"] --> RUN["src/pipeline/run.py"] --> CORE
    API["src/api/app.py (FastAPI)<br/>/profile /clean /transform /mask-pii<br/>/validate /drift /anomalies /run"] --> CORE
    DOCKER["Dockerfile · scripts/demo.sh · CI"] --> CLI
    CORE["src/profiling · src/cleaning · src/privacy · src/quality<br/>src/validation · src/transform · src/impact"]
```

| Door | Use it for | Example |
|---|---|---|
| CLI | one task at a time, in a terminal or CI | `python gatekeeper.py validate new.csv --contract contracts/bank_marketing.yaml` |
| Pipeline YAML | the whole flow with one command, gates, one report | `python gatekeeper.py run --config pipeline_adult.yaml --skip impact` |
| HTTP API | other programs; upload a file, get JSON or the page | `curl -F file=@data.csv -F config=@config.json http://127.0.0.1:8000/run` |
| Docker / demo | the same commands without a Python setup | `make demo` |

## Where things live

| Folder | What it does | Built in |
|---|---|---|
| `src/profiling/` | profile a file, semantic types, disguised missing, cleaning suggestions, decisions for a person, charts | Sprints 1–3, 11 |
| `src/quality/` | quality score 0–100 (completeness, uniqueness, validity, consistency) | Sprint 3, 6 |
| `src/cleaning/` | cleaning steps 1–12 (+ 8b for new rows), imputers (median, KNN, regression, MICE; fit/transform), outliers, fuzzy duplicates, schema | Sprints 4–5, 9–11 |
| `src/transform/` | per-column encoding, power transform, scaling; one scikit-learn pipeline fitted on train rows | Sprint 6, 12 |
| `src/privacy/` | find PII (9 types) and mask / hash / drop / generalise it | Sprint 6 |
| `src/validation/` | data contracts, drift (PSI + KS), anomalies, the combined validation report | Sprints 7, 9, 10 |
| `src/impact/` | does cleaning help a model? (A / B / C CV), corruption injectors, robustness, fairness measurement | Sprints 8, 11 |
| `src/pipeline/` | `gatekeeper run` (steps, gates, logging) and `report.html` | Sprints 8, 11 |
| `src/api/` | FastAPI app with upload limits | Sprints 2, 6, 9, 11 |
| `scripts/` | dataset downloads (checksums), demo, every evaluation and ablation behind a reported number | all |
| `contracts/` | data contracts and confirmed decisions (YAML) | Sprints 7, 10, 11 |
| `docs/` | sprint reports, presentations, JSON schemas, this page | all |

## Rules the design keeps
- **Fit on train rows only.** Every learned statistic (schema, imputer, scaler, forest) is fitted inside a fold
  or on a reference, then only applied to new rows. A test spies on every `fit()` in the impact study.
- **No raw personal data in any output.** Contract and drift reports mask PII values, anomalies never use PII
  columns, the HTML page shows only those reports, and logs hold names, counts and timings.
- **A person decides what only a person can know.** Uncertain codes are listed with evidence (`decisions`); the
  cleaner obeys the confirmed choice and logs who made it.
- **Exit codes mean one thing.** 0 = the data passed, 1 = the data failed a gate, 2 = a bad input or a crash.
