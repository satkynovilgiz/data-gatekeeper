# Sprint 7 Summary — Module 3 part 1: data contracts + drift gate

**Goal:** a new dataset either PASSES or FAILS a written data contract, and drift is measured
against a reference dataset. Exit codes make both usable in CI.

## What I built
| Feature | Where |
|---|---|
| YAML data contracts: dtype, required, nullable / max_missing_pct, allowed_values, min / max, regex, unique; table min_rows, max_duplicate_rows_pct, extra columns; severity `error` / `warn` per rule | `src/validation/contract.py`, `contracts/example.yaml` |
| `contract init` drafts a contract from a reference file with the existing profiler (no PII values written) | `src/validation/contract.py` |
| Drift: PSI over 10 reference-decile bins + missing bin, KS (scipy) for numeric; PSI over categories with new / vanished categories; thresholds from the contract | `src/validation/drift.py` |
| PII masking reused (`src/privacy/mask.py`): no raw PII values in validate or drift output | both modules |
| CLI `gatekeeper.py contract init / validate / drift`, exit codes 0 / 1 / 2, JSON in `reports/` | `gatekeeper.py` |
| Reproducible demo files (seed 42) | `scripts/make_drift_demo.py` |
| Dockerfile + CI jobs `contract-gate` and `docker` | `Dockerfile`, `.github/workflows/tests.yml` |
| Tests **219 → 297**, all passing; ruff clean | `tests/test_contract.py`, `test_drift.py`, `test_gatekeeper_cli.py` |

## Commands
```bash
python scripts/make_drift_demo.py                    # data/demo/: reference 70%, current 30%, shifted
python gatekeeper.py contract init data/demo/bank_reference.csv --name bank_marketing
python gatekeeper.py validate data/demo/bank_current.csv --contract contracts/bank_marketing.yaml
python gatekeeper.py drift data/demo/bank_reference.csv data/demo/bank_current_shifted.csv \
       --contract contracts/bank_marketing.yaml
```
Exit codes: `validate` 0 = pass, 1 = an error rule failed, 2 = bad contract or input.
`drift` 0 = no major drift, 1 = at least one major column, 2 = bad input.

## Demo on Bank Marketing (real outputs)
Split: stratified on `y`, seed 42 → reference 28,831 rows, current 12,357 rows (y = yes 11.27% / 11.26%).
A random split, not a time split: the file is in time order and the economic columns change over time.
Shifted file = current + 3 changes: `duration` seconds → minutes (/60, 1 decimal), 3,612 `telephone`
→ `cellular` (80%), 618 ages emptied (5%). The contract was used exactly as `init` drafted it.

| | current (no changes) | current_shifted |
|---|---|---|
| Contract (97 rules) | **PASS**: 97 passed, 0 failed, exit 0 | **FAIL**: 95 passed, 2 failed (error), exit 1 |
| Failed rules | – | `age` max_missing_pct 5.00% (618 rows) > 0%; `duration` dtype: 10,921 of 12,357 not integer |
| Drift | 21 stable, 0 major, exit 0 | 3 major, 18 stable, exit 1 |
| Highest PSI | age 0.0011 (KS 0.0053) | see below |

Top drifted columns, shifted file:

| column | PSI | KS | status | why |
|---|---|---|---|---|
| duration | 8.2614 | 0.9623 | MAJOR | 99.98% of values now fall in the lowest reference bin (< 59) |
| contact | 0.5806 | – | MAJOR | cellular 63.48% → 92.69%, telephone 36.52% → 7.31% |
| age | 0.3138 | 0.0069 | MAJOR | missing 0.0% → 5.0% |
| job (next) | 0.0010 | – | stable | unchanged |

Major drift appears on exactly the three columns we changed; every other column stays below 0.002.

**Contract and drift catch different things.**
- The contract did **not** catch the `contact` mix change: both values are still allowed.
- `duration` failed the contract only because minutes have decimals. In an extra run with **whole** minutes,
  the contract **passed (exit 0)** while drift still gave PSI 8.2614: values in minutes stay inside the 0–5410 range.

## Limitations
- `age` is major only because of the missing bin: 0.05 × ln(0.05 / 0.0001) ≈ 0.31 of its 0.3138. The
  present values did not move (KS 0.0069). A different EPS would change this number.
- Missing = empty cells. `unknown` and `pdays = 999` are ordinary values for contracts (allowed / in range).
- Columns with ties get fewer than 10 bins (`pdays`, `previous`). Free-text and id columns are skipped by drift.
- KS p-values are tiny for any difference on large files, so the status uses PSI only.
- `init` drafts from what it saw (range + 10%, every observed category): a person must review it.
- The Dockerfile and the CI `docker` job were **not run**: Docker is not installed on this machine,
  and the branch is not pushed. The `contract-gate` steps were run locally and gave 0 / 1 / 2.
- The image installs `requirements.txt`, which still includes dev tools (pytest, ruff).

## Next steps (Sprints 8–9)
- Anomaly detection section; validate cleaned data against `schema.json` from Module 2.
- Optional `missing_values: [unknown]` in contracts; a `--fail-on moderate` flag for drift.
- An API endpoint for validate / drift.
