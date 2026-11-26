# Demo video script (3–4 minutes)

Every command and number below is real: from the Sprint 12 demo run (`make demo`, 287 s) and the sprint reports.
Before recording, run `make demo` once so the downloads are cached. During the recording, show the terminal and a
browser side by side.

---

### 0:00 – 0:20 · The problem (show the README top)
**Say:** "Data that looks clean often isn't. This bank file looks zero percent missing to pandas, but more than
fifty thousand cells are really missing, written as 'unknown' or 999. Data Gatekeeper checks a file before anyone uses
it, and tells a person in plain words whether it can be used."

### 0:20 – 1:10 · One command (terminal)
```bash
make demo
```
**Show** the output scrolling: the downloads with their SHA-256, then the two pipelines.
**Say:** "One command downloads two real public datasets, checks their checksums, and runs the whole pipeline on both:
profile, contract, cleaning, personal-data check, quality, unusual rows, drift. Here there's no Docker on this laptop,
so it says so and runs the same commands locally; CI runs them inside the Docker image."
**Point at:** `profile PASSED`, `validate PASSED`, … and the two `Report ->` lines at the end
(`done locally in 287 s`).

### 1:10 – 2:10 · The report a person reads (browser: the Bank `report.html`)
**Show, top to bottom:**
1. the green banner: "This file passed every check that was run. It can be used."
2. "What was done": eight steps in plain words, with seconds
3. "Contract": 97 of 97 rules passed
4. "What cleaning changed": 12 duplicates removed, 4,121 hidden missing values found, 4,120 filled in, and who decided
   each uncertain value
5. "Data quality score": 98.48 → 98.60, **and the warning under it**

**Say:** "Every change is logged, and no personal data ever appears on this page. And notice this warning: a higher
quality score does not mean a better model. That is one of the honest findings of this project."

### 2:10 – 3:00 · The honest finding (browser: the Adult `report.html`, then `docs/sprint-10.md` / `sprint-11.md`)
**Show** "Decisions for a person" in the Adult report: capital-gain 99999 (244 rows, 100% earn >50K), workclass and
occupation "?" (about 9.4% vs 24.81%).
**Say:** "On the second dataset, census income, cleaning first made the models worse. The reason: 99999 is not a
missing value, it is a capped income, and every one of those 244 people earns over 50K. A '?' for job also means
something. So the tool now lists values like these with the evidence, a person confirms them in a small YAML file, and
the cleaner obeys. With those decisions and the transform chosen by evidence, the gatekeeper no longer hurts either
model on this data."

### 3:00 – 3:40 · It is measured (terminal or `docs/sprint-12.md`)
```bash
python gatekeeper.py decisions data/raw/adult.csv --target income
```
**Say:** "The key rules of the last sprints were written down before measuring. On dirty data the gatekeeper is far more robust: with 20% mixed
corruption, logistic regression scores 0.8168 ROC-AUC with it and 0.5081 without. The decision detector was also
checked on a dataset I had never used, mushrooms, and its finding held up on rows it had never seen."

### 3:40 – 4:00 · Close (README key results table)
**Say:** "Twelve sprints, 449 tests, one command. Data Gatekeeper tells you what is in your data, what it
changed, what only you can decide, and whether it actually helped."
