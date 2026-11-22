# Dataset — UCI Bank Marketing (finance domain)

| | |
|---|---|
| **Name** | Bank Marketing (`bank-additional-full.csv`) |
| **Domain** | Finance / retail banking |
| **Source** | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/222/bank+marketing |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): free to use and share, with credit |
| **Citation** | Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306 |
| **Size** | 41,188 rows × 21 columns, ~5.8 MB, `;` separated |

## What it is
Phone marketing campaigns of a Portuguese bank (May 2008 – Nov 2010).
Each row is one call to a client. The target `y` = did the client open a term deposit (`yes`/`no`).
Columns: client data (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`),
the call (`contact`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`)
and economic indicators (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`).

## Why this dataset (why it is "messy enough")
- **Hidden missing values**: missing data is written as the text `unknown` (e.g. `default` is 20.9% `unknown`), so pandas does not see it as missing.
- **Magic number**: `pdays = 999` means "never contacted before" (96.3% of rows).
- **Near-constant columns**: `pdays`, `previous`, `poutcome` are dominated by one value.
- **Duplicate rows**: 12 exact duplicates.
- **Imbalanced target**: only ~11% `yes`.
- **Non-standard format**: `;` separator and dots in column names (`emp.var.rate`).
- No personal data (anonymised), which is good for privacy. It also gives a useful test: the PII detector should find **nothing**.

## Files in this repo
| File | In git? | How to get it |
|---|---|---|
| `data/raw/bank_marketing.csv` | ❌ no (`.gitignore`, too big) | `python scripts/download_bank_marketing.py` |
| `data/raw/bank_marketing_sample.csv` | ✅ yes (1,000 random rows, seed 42, ~140 KB) | `python scripts/download_bank_marketing.py --sample` |
| `data/raw/hr_employees_messy.csv` | ✅ yes (synthetic, Sprint 1) | `python scripts/make_sample_data.py` |
| `data/raw/customers_pii.csv` | ✅ yes (synthetic, Sprint 6: **fake** PII for the privacy module) | `python scripts/make_sample_data.py` |
