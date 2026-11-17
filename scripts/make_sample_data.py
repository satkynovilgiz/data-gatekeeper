"""Generate a small, deliberately messy HR dataset for testing the pipeline.

Run:  python scripts/make_sample_data.py
Out:  data/raw/hr_employees_messy.csv

Replace this with your team's real dataset once you pick a domain.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

first = ["Aibek", "Maria", "John", "Aisha", "Chen", "Liam", "Sofia", "Omar", "Elena", "Raj"]
last = ["Smith", "Garcia", "Lee", "Khan", "Ivanov", "Brown", "Kim", "Patel", "Nguyen", "Lopez"]
depts = ["Engineering", "Sales", "HR", "Finance", "Marketing"]

names = [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(N)]
df = pd.DataFrame({
    "employee_id": [f"E{1000 + i}" for i in range(N)],
    "full_name": names,
    "email": [n.lower().replace(" ", ".") + "@company.com" for n in names],
    "phone": [f"+1-408-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}" for _ in range(N)],
    "department": rng.choice(depts, N),
    "age": rng.integers(21, 64, N).astype(float),
    "salary": rng.normal(85000, 20000, N).round(0),
    "years_experience": rng.integers(0, 35, N).astype(float),
    "hire_date": pd.to_datetime("2015-01-01")
    + pd.to_timedelta(rng.integers(0, 3500, N), unit="D"),
    "performance_score": rng.integers(1, 6, N).astype(float),
})
df["hire_date"] = df["hire_date"].dt.strftime("%Y-%m-%d")

# ---- Inject realistic mess ----
def pick(frac):
    return rng.choice(N, int(N * frac), replace=False)

df.loc[pick(0.08), "salary"] = np.nan                       # missing values
df.loc[pick(0.05), "age"] = np.nan
df.loc[pick(0.04), "email"] = None
df.loc[pick(0.02), "age"] = 230                              # impossible values
df.loc[pick(0.01), "salary"] = 9_999_999                     # outliers
df.loc[pick(0.06), "department"] = rng.choice(               # inconsistent categories
    ["engineering", "ENGINEERING ", "Sales ", "hr", "Fin."], int(N * 0.06))
idx = pick(0.10)                                             # mixed date formats
df.loc[idx, "hire_date"] = pd.to_datetime(df.loc[idx, "hire_date"]).dt.strftime("%d/%m/%Y")
df.loc[pick(0.03), "email"] = "not-an-email"                 # invalid emails
df["years_experience"] = df["years_experience"].astype(object)
df.loc[pick(0.03), "years_experience"] = "five"             # mixed types
df = pd.concat([df, df.sample(15, random_state=1)])          # exact duplicates

out = Path(__file__).resolve().parents[1] / "data" / "raw" / "hr_employees_messy.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows -> {out}")
