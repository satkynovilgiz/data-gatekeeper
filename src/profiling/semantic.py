"""Detect what a column *means* (email, phone, date, ID, name, ...) and its basic type.

Two signals are combined:
  1. the column name (e.g. "email", "dob", "customer_id")
  2. the values themselves (regex match rate on a sample)
"""
from __future__ import annotations

import re

import pandas as pd

# Order matters: the first pattern that matches >= 80% of values wins.
# "ip_address" must come before "phone", because "192.168.0.1" also fits the
# loose phone pattern (digits and dots).
PATTERNS = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
    "ip_address": re.compile(
        r"^((25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(25[0-5]|2[0-4]\d|1?\d?\d)$"),  # IPv4, each part 0-255
    "phone": re.compile(r"^\+?[\d\s().-]{7,20}$"),
    "postcode": re.compile(r"^([A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}|\d{5}(-\d{4})?)$", re.I),
    "url": re.compile(r"^https?://", re.I),
}

# Credit card: 13-19 digits, optionally split by spaces or dashes.
CARD_PATTERN = re.compile(r"^\d(?:[ -]?\d){12,18}$")

# Column names that mean "date of birth" (a date alone is not personal data,
# but a birth date is).
DOB_HINTS = ["dob", "birth", "birthday"]

# ID hints are matched as whole words, so "cons.price.idx" or "paid" are NOT ids.
ID_WORDS = {"id", "uuid", "guid", "code", "key"}

NAME_HINTS = {
    "email": ["email", "e-mail", "mail"],
    "phone": ["phone", "mobile", "tel"],
    "postcode": ["postcode", "zip", "postal"],
    "date": ["date", "dob", "time", "created", "updated"],
    "person_name": ["name", "first", "last", "surname"],
    "address": ["address", "street", "city"],
}

PII_TYPES = {"email", "phone", "postcode", "person_name", "address",
             "credit_card", "ip_address", "date_of_birth"}


def name_words(name: str) -> list[str]:
    """Split a column name into lowercase words.

    'customerID' and 'customer_id' both give ['customer', 'id'].
    """
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)      # camelCase -> camel Case
    return [w for w in re.split(r"[^a-z0-9]+", name.lower()) if w]


def is_id_name(name: str) -> bool:
    """True if the column name looks like an identifier (customer_id, ID, uuid, ...)."""
    return any(w in ID_WORDS for w in name_words(name))


def luhn_ok(digits: str) -> bool:
    """Luhn checksum: the check every real credit card number passes.

    From the right, double every second digit (subtract 9 if > 9) and add
    everything up; valid numbers give a total that ends in 0.
    """
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def credit_card_rate(series: pd.Series) -> float:
    """Share of non-null values that look like a card number AND pass the Luhn check."""
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return 0.0
    ok = s.str.match(CARD_PATTERN) & s.str.replace(r"[ -]", "", regex=True).map(
        lambda v: v.isdigit() and luhn_ok(v))
    return float(ok.mean())


def match_rate(series: pd.Series, pattern: re.Pattern) -> float:
    """Share of non-null values that match a regex (0..1)."""
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return 0.0
    return float(s.str.match(pattern).mean())


def date_parse_rate(series: pd.Series) -> float:
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
    return float(parsed.notna().mean())


def numeric_parse_rate(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    return float(pd.to_numeric(s, errors="coerce").notna().mean())


def infer_basic_type(series: pd.Series) -> str:
    """numeric | datetime | boolean | categorical | text | empty."""
    s = series.dropna()
    if s.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if numeric_parse_rate(series) >= 0.95:
        return "numeric"
    is_texty = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    if is_texty and date_parse_rate(series) >= 0.9 and numeric_parse_rate(series) < 0.5:
        return "datetime"
    nunique = s.nunique()
    if nunique <= max(20, 0.05 * len(s)):
        return "categorical"
    return "text"


def detect_semantic(name: str, series: pd.Series, basic_type: str) -> dict:
    """Return {'semantic_type': str, 'confidence': float, 'evidence': str}."""
    lname = name.lower()

    # 0) date of birth: decided by the column name (a birth date looks like any other date)
    if any(h in lname for h in DOB_HINTS):
        return {"semantic_type": "date_of_birth", "confidence": 0.9,
                "evidence": "column name says date of birth"}

    # 1) dates first — date strings like 2024-01-31 would otherwise look like phone numbers
    if basic_type == "datetime":
        return {"semantic_type": "date", "confidence": round(date_parse_rate(series), 3),
                "evidence": "values parse as dates"}

    # 2) value-based checks (strongest evidence)
    card = credit_card_rate(series)
    if card >= 0.8:  # checked first: card numbers would also match the phone pattern
        return {"semantic_type": "credit_card", "confidence": round(card, 3),
                "evidence": f"{card:.0%} of values are 13-19 digits and pass the Luhn check"}
    for sem, pat in PATTERNS.items():
        rate = match_rate(series, pat)
        if rate >= 0.8 and (sem != "phone" or basic_type != "numeric"):
            return {"semantic_type": sem, "confidence": round(rate, 3),
                    "evidence": f"{rate:.0%} of values match the {sem} pattern"}

    # 3) identifier: almost all values unique + id-like name
    s = series.dropna()
    unique_ratio = s.nunique() / len(s) if len(s) else 0
    if unique_ratio > 0.95 and is_id_name(name):
        return {"semantic_type": "id", "confidence": round(unique_ratio, 3),
                "evidence": "unique values and id-like column name"}

    # 4) name-based hints (weaker evidence)
    if is_id_name(name):
        return {"semantic_type": "id", "confidence": 0.6,
                "evidence": "column name contains an id keyword"}
    for sem, hints in NAME_HINTS.items():
        if any(h in lname for h in hints):
            return {"semantic_type": sem, "confidence": 0.6,
                    "evidence": f"column name contains a '{sem}' keyword"}

    return {"semantic_type": basic_type, "confidence": 0.5, "evidence": "basic type only"}
