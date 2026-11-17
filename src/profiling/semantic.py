"""Detect what a column *means* (email, phone, date, ID, name, ...) and its basic type.

Two signals are combined:
  1. the column name (e.g. "email", "dob", "customer_id")
  2. the values themselves (regex match rate on a sample)
"""
from __future__ import annotations

import re

import pandas as pd

PATTERNS = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
    "phone": re.compile(r"^\+?[\d\s().-]{7,20}$"),
    "postcode": re.compile(r"^([A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}|\d{5}(-\d{4})?)$", re.I),
    "url": re.compile(r"^https?://", re.I),
}

NAME_HINTS = {
    "email": ["email", "e-mail", "mail"],
    "phone": ["phone", "mobile", "tel"],
    "postcode": ["postcode", "zip", "postal"],
    "date": ["date", "dob", "time", "created", "updated"],
    "id": ["_id", "id", "uuid", "code"],
    "person_name": ["name", "first", "last", "surname"],
    "address": ["address", "street", "city"],
}

PII_TYPES = {"email", "phone", "postcode", "person_name", "address"}


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

    # 1) dates first — date strings like 2024-01-31 would otherwise look like phone numbers
    if basic_type == "datetime":
        return {"semantic_type": "date", "confidence": round(date_parse_rate(series), 3),
                "evidence": "values parse as dates"}

    # 2) value-based checks (strongest evidence)
    for sem, pat in PATTERNS.items():
        rate = match_rate(series, pat)
        if rate >= 0.8 and (sem != "phone" or basic_type != "numeric"):
            return {"semantic_type": sem, "confidence": round(rate, 3),
                    "evidence": f"{rate:.0%} of values match the {sem} pattern"}

    # 3) identifier: almost all values unique + id-like name
    s = series.dropna()
    unique_ratio = s.nunique() / len(s) if len(s) else 0
    if unique_ratio > 0.95 and any(h in lname for h in NAME_HINTS["id"]):
        return {"semantic_type": "id", "confidence": round(unique_ratio, 3),
                "evidence": "unique values and id-like column name"}

    # 4) name-based hints (weaker evidence)
    for sem, hints in NAME_HINTS.items():
        if any(h in lname for h in hints):
            return {"semantic_type": sem, "confidence": 0.6,
                    "evidence": f"column name contains a '{sem}' keyword"}

    return {"semantic_type": basic_type, "confidence": 0.5, "evidence": "basic type only"}
