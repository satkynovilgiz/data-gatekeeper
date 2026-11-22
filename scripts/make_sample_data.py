"""Generate a small, deliberately messy HR dataset for testing the pipeline.

Run:  python scripts/make_sample_data.py
Out:  data/raw/hr_employees_messy.csv   (HR, messy on purpose)
      data/raw/customers_pii.csv        (Sprint 6: fake customers WITH fake PII)

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
    """Return random row positions covering `frac` of the N rows (e.g. 0.08 = 8%)."""
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

# Sprint 5: near-duplicates ("fuzzy" duplicates) — the same employee typed in again with a
# new ID and a small change in the name; the email follows the typed name. Added at the end
# with their own seed, so the rows above stay exactly the same.
def retype(name):
    """One realistic re-typing of 'First Last', picked at random."""
    first, last = name.split(" ")
    kind = rng2.integers(0, 4)
    if kind == 0:
        return f"{first[:-1]} {last}"                          # a letter missing: "Aibe Patel"
    if kind == 1:
        return f"{first} {last[:2]}{last[1]}{last[2:]}"        # a letter doubled: "Paatel"
    if kind == 2:
        return f"{last}, {first}"                              # surname first: "Patel, Aibek"
    return f"{first[0]}{first[2]}{first[1]}{first[3:]} {last}" # two letters swapped: "Abiek"

rng2 = np.random.default_rng(7)
near = df.iloc[rng2.choice(500, 8, replace=False)].copy()
near["full_name"] = [retype(n) for n in near["full_name"]]
near["email"] = [n.lower().replace(", ", ".").replace(" ", ".") + "@company.com"
                 for n in near["full_name"]]
near["employee_id"] = [f"E{2000 + i}" for i in range(len(near))]
df = pd.concat([df, near])

out = Path(__file__).resolve().parents[1] / "data" / "raw" / "hr_employees_messy.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows -> {out}")


# ---- Sprint 6: fake customers with fake personal data, for the PII module -------------
# Bank Marketing has no PII, so we need a file that does. Every value is FAKE and chosen
# from ranges that are reserved for examples, so no real person can be hit:
#   emails  @example.com / .org / .net        (reserved by RFC 2606)
#   phones  555-0100 .. 555-0199               (reserved for fiction in North America)
#   SSNs    area 000 or 666                    (never issued by the US SSA)
#   cards   public payment test numbers        (pass the Luhn check, cannot be charged)
#   IPs     192.0.2.x, 198.51.100.x, 203.0.113.x  (TEST-NET documentation ranges)
# Own seed, after the HR file is written, so the HR rows above stay byte-identical.
rng3 = np.random.default_rng(2026)
M = 300
TEST_CARDS = ["4242424242424242", "4000056655665556", "5555555555554444",
              "2223003122003222", "5200828282828210", "378282246310005",
              "6011111111111117", "3056930009020004", "3566002020360505"]
streets = ["Maple", "Oak", "Cedar", "Pine", "Elm", "Birch", "Willow", "Lakeview", "Hillside"]
kinds = ["Street", "Avenue", "Road", "Lane", "Drive", "Blvd"]
nets = ["192.0.2", "198.51.100", "203.0.113"]


def card(number):
    """Write a card number the way people type it: with spaces, dashes or nothing."""
    style = rng3.integers(0, 3)
    if style == 0:
        return number
    groups = [number[i:i + 4] for i in range(0, len(number), 4)]
    return (" " if style == 1 else "-").join(groups)


cust_names = [f"{rng3.choice(first)} {rng3.choice(last)}" for _ in range(M)]
domains = rng3.choice(["example.com", "example.org", "example.net"], M)
customers = pd.DataFrame({
    "customer_id": [f"C{5000 + i}" for i in range(M)],
    "full_name": cust_names,
    "email": [f"{n.lower().replace(' ', '.')}{i}@{d}"
              for i, (n, d) in enumerate(zip(cust_names, domains))],
    "phone": [f"+1-{rng3.integers(200, 999)}-555-01{rng3.integers(0, 100):02d}" for _ in range(M)],
    "ssn": [f"{rng3.choice(['000', '666'])}-{rng3.integers(1, 100):02d}-"
            f"{rng3.integers(1, 10000):04d}" for _ in range(M)],
    "credit_card": [card(rng3.choice(TEST_CARDS)) for _ in range(M)],
    "ip_address": [f"{rng3.choice(nets)}.{rng3.integers(1, 255)}" for _ in range(M)],
    "street_address": [f"{rng3.integers(1, 9999)} {rng3.choice(streets)} {rng3.choice(kinds)}"
                       for _ in range(M)],
    "zip_code": [f"{rng3.integers(10000, 99999)}" for _ in range(M)],
    "date_of_birth": (pd.to_datetime("1950-01-01")
                      + pd.to_timedelta(rng3.integers(0, 20000, M), unit="D")).strftime("%Y-%m-%d"),
    "signup_date": (pd.to_datetime("2021-01-01")
                    + pd.to_timedelta(rng3.integers(0, 1200, M), unit="D")).strftime("%Y-%m-%d"),
    "plan": rng3.choice(["basic", "plus", "pro"], M, p=[0.5, 0.35, 0.15]),
    "monthly_spend": rng3.gamma(2.0, 20.0, M).round(2),
    "churned": rng3.choice(["no", "yes"], M, p=[0.8, 0.2]),
})
customers.loc[rng3.choice(M, 9, replace=False), "email"] = None      # a few gaps
out2 = out.parent / "customers_pii.csv"
customers.to_csv(out2, index=False)
print(f"Wrote {len(customers)} rows -> {out2}  (all personal data is fake)")
