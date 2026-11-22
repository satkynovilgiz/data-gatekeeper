"""Find columns that hold personal data (PII = personally identifiable information).

Two signals per column, as in Module 1's semantic types, but stricter and with more types:
  name     a word in the column name ("email", "dob", "zip", "ssn" ...)        -> +0.3
  content  share of sampled values that look like the type (regex, Luhn)   -> +0.7 x share
  confidence = 0.3 x name + 0.7 x content share;  PII when confidence >= MIN_CONFIDENCE

Some types need the name too: a 5-digit number is only a zip code, and a date is only a
date of birth, when the column name says so (salary 82114 is not a zip code; hire_date
is not a birthday).

Privacy rule: the result holds column names, types and rates — NEVER a value.
"""
from __future__ import annotations

import re
from collections.abc import Callable

import pandas as pd

from src.profiling.semantic import PATTERNS, date_parse_rate, luhn_ok, name_words

MIN_CONFIDENCE = 0.5
SAMPLE = 1000            # values checked per column (enough for a stable share, and fast)

SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
SSN_NO_DASH = re.compile(r"^\d{9}$")                       # only with an "ssn" column name
PHONE = re.compile(r"^\+?\d{0,3}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$")
ZIP = re.compile(r"^(\d{5}(-\d{4})?|[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2})$", re.I)
CARD = re.compile(r"^\d(?:[ -]?\d){12,18}$")
STREET = re.compile(r"^\d+[A-Za-z]?\s+[\w .'-]+\s(street|st|avenue|ave|road|rd|lane|ln|drive|dr|"
                    r"blvd|boulevard|way|court|ct|place|pl|square|sq)\.?$", re.I)
PERSON = re.compile(r"^[A-Z][a-z'-]+,?(\s[A-Z][a-z'-]+){1,2}$")


def _rate(values: pd.Series, test: Callable[[str], bool]) -> float:
    """Share of the (string) values for which test(value) is True."""
    return float(values.map(test).mean()) if len(values) else 0.0


def _card(v: str) -> bool:
    """13-19 digits (spaces/dashes allowed) that pass the Luhn checksum."""
    return bool(CARD.match(v)) and luhn_ok(re.sub(r"[ -]", "", v))


def _person(values: pd.Series) -> float:
    """Share of values that look like 'First Last' or 'Last, First'."""
    return _rate(values, lambda v: bool(PERSON.match(v)))


# type -> (words in the column name, content check, name required?)
# Order = priority when two types score the same (cards before phones: both are digits).
RULES: dict[str, tuple[set[str], Callable[[pd.Series], float], bool]] = {
    "email": ({"email", "mail"}, lambda s: _rate(s, lambda v: bool(PATTERNS["email"].match(v))),
              False),
    "credit_card": ({"card", "cc", "pan"}, lambda s: _rate(s, _card), False),
    "ssn": ({"ssn", "social", "sin"}, lambda s: _rate(s, lambda v: bool(SSN.match(v))), False),
    "ip_address": ({"ip"}, lambda s: _rate(s, lambda v: bool(PATTERNS["ip_address"].match(v))),
                   False),
    "phone": ({"phone", "mobile", "tel", "telephone", "cell"},
              lambda s: _rate(s, lambda v: bool(PHONE.match(v))), False),
    "street_address": ({"address", "street", "addr"},
                       lambda s: _rate(s, lambda v: bool(STREET.match(v))), False),
    "person_name": ({"name", "fullname", "firstname", "lastname", "surname"}, _person, False),
    "date_of_birth": ({"dob", "birth", "birthday", "birthdate"}, date_parse_rate, True),
    "zip_code": ({"zip", "zipcode", "postcode", "postal"},
                 lambda s: _rate(s, lambda v: bool(ZIP.match(v))), True),
}


def detect_column(name: str, series: pd.Series) -> dict | None:
    """Best PII match for one column: {column, pii_type, confidence, evidence} or None."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""].head(SAMPLE)
    words = set(name_words(name))
    best = None
    for pii_type, (hints, content, needs_name) in RULES.items():
        named = bool(words & hints)
        if needs_name and not named:
            continue
        if pii_type == "person_name" and not named and values.nunique() < 0.3 * len(values):
            continue        # no "name" in the column name and values repeat: a category
        share = content(values) if len(values) else 0.0
        if pii_type == "ssn" and named:                    # "123456789" in a column called ssn
            share = max(share, _rate(values, lambda v: bool(SSN_NO_DASH.match(v))))
        confidence = round(0.3 * named + 0.7 * share, 2)
        if confidence >= MIN_CONFIDENCE and (best is None or confidence > best["confidence"]):
            evidence = [f"column name contains '{sorted(words & hints)[0]}'"] if named else []
            evidence.append(f"{share:.0%} of {len(values)} sampled values look like {pii_type}")
            best = {"column": name, "pii_type": pii_type, "confidence": confidence,
                    "evidence": "; ".join(evidence)}
    return best


def detect_pii(df: pd.DataFrame) -> list[dict]:
    """All PII columns of a DataFrame, in column order (no values, only rates)."""
    return [hit for c in df.columns if (hit := detect_column(str(c), df[c])) is not None]
