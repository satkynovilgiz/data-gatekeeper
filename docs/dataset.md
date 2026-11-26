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

---

# Second dataset (Sprint 10) — UCI Adult (Census Income)

| | |
|---|---|
| **Name** | Adult, also called "Census Income" (`adult.data` + `adult.test`) |
| **Domain** | Census / income |
| **Source** | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/2/adult (zip: https://archive.ics.uci.edu/static/public/2/adult.zip) |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): free to use and share, with credit |
| **Citation** | Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20 |
| **Size** | 48,842 rows × 15 columns (32,561 train + 16,281 test, joined), ~5 MB |
| **Checksum** | zip SHA-256 `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` (checked by the script); `adult.csv` SHA-256 `6f8f2babc5ee744afd03f6d978d8d6b3e3b0aae240d931c4976a9cce7af0d347` (written to `data/raw/adult.csv.sha256`) |
| **Target** | `income`: `>50K` (11,687 rows, 23.93%) or `<=50K` (37,155) |

## What it is
One row per adult from the 1994 US Census database: `age`, `workclass`, `fnlwgt` (census sampling weight),
`education` / `education-num` (the same information twice, as text and as a number), `marital-status`,
`occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`.
The task is to predict whether a person earns more than $50K a year.

## Why it is "messier" than Bank Marketing
- **Real missing values written as `?`**: `workclass` 2,799, `occupation` 2,809, `native-country` 857 rows
  (counted in `adult.csv`). pandas does not see them as missing.
- **52 exact duplicate rows**.
- **Top-coded value**: `capital-gain = 99999` (244 rows) is a cap, not a real amount; 91.74% of rows have
  `capital-gain = 0` and 95.33% `capital-loss = 0`.
- **Redundant columns**: `education` and `education-num` say the same thing.
- **Format quirks in the published files** (fixed by the download script, see below): a space after every
  comma, a junk first line `|1x3 Cross validator` in `adult.test`, empty lines at its end, and labels
  written `>50K.` with a dot only in the test file.
- **Sensitive attributes**: `sex`, `race`, `native-country` (and `age`, `marital-status`). This project uses them
  like any other column and does **no fairness analysis yet**, which is a limitation, not a decision about fairness.

## How the file is made
`python scripts/download_adult.py` downloads the zip, **stops if the SHA-256 differs**, joins train and test,
and removes only the format quirks above (separator spaces, the junk line, empty lines, the label's dot).
Every data problem (`?`, duplicates, 99999) is left for the gatekeeper to find.
`--split` also writes the two official parts to `data/demo/adult_train.csv` / `adult_test.csv`.

| File | In git? | How to get it |
|---|---|---|
| `data/raw/adult.csv` (+ `.sha256`) | ❌ no (`.gitignore`) | `python scripts/download_adult.py` |
| `data/demo/adult_train.csv`, `adult_test.csv` | ❌ no | `python scripts/download_adult.py --split` |
| `contracts/adult.yaml` | ✅ yes | drafted by `gatekeeper contract init data/demo/adult_train.csv --name adult` |
