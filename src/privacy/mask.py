"""Protect PII columns: mask, hash, drop or generalise them — and report what was done.

Strategies (choose per column; DEFAULT_ACTIONS when nothing is chosen):
  mask        hide most characters, keep the shape:  john.smith@gmail.com -> j***@g***.com,
              +1-408-555-0101 -> +*-***-***-0101 (last 4 digits stay, like on a receipt)
  hash        salted SHA-256 (HMAC): the same input always gives the same code, so two
              tables can still be JOINED on it, but the code cannot be turned back.
              The salt (a secret) comes from the environment variable GATEKEEPER_PII_SALT
              and is never written in the code: without it, nobody can rebuild the codes
              by hashing a list of known emails.
  drop        remove the column (when nobody needs it: SSNs, card numbers)
  generalize  keep a coarse version: date of birth -> age band "30-39",
              zip code -> first 3 digits "660**", IP -> network "192.0.2.0/24"
  keep        leave the column as it is (a human decided it is fine)

Privacy rule: values never appear in the report, in logs or in error messages.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from datetime import date

import pandas as pd

from .detect import detect_pii

SALT_ENV = "GATEKEEPER_PII_SALT"
ACTIONS = {"mask", "hash", "drop", "generalize", "keep"}
GENERALIZABLE = {"date_of_birth", "zip_code", "ip_address"}
DIGIT_TYPES = {"phone", "ssn", "credit_card"}           # compared on digits only
AGE_BAND = 10

# Chosen so the data stays useful where it can, and disappears where it cannot:
DEFAULT_ACTIONS = {
    "email": "hash",            # a join key in many systems
    "person_name": "hash",
    "phone": "mask",
    "ssn": "drop",              # no analysis needs it
    "credit_card": "drop",
    "street_address": "drop",
    "ip_address": "generalize",
    "date_of_birth": "generalize",
    "zip_code": "generalize",
}


class PrivacyError(ValueError):
    """A problem with the PII settings (the message never contains a data value)."""


def get_salt() -> bytes:
    """The hashing secret from the environment; a clear error when it is not set."""
    salt = os.environ.get(SALT_ENV, "")
    if not salt:
        raise PrivacyError(f"hashing needs a secret salt: set the environment variable "
                           f"{SALT_ENV} (e.g. export {SALT_ENV}=$(openssl rand -hex 32))")
    return salt.encode()


def normalise(value: str, pii_type: str) -> str:
    """Same person -> same text before hashing: trimmed, lowercase, digits only for numbers."""
    text = str(value).strip().lower()
    return re.sub(r"\D", "", text) if pii_type in DIGIT_TYPES else text


def hash_value(value: str, pii_type: str, salt: bytes) -> str:
    """HMAC-SHA256 of the normalised value, as 64 hex characters."""
    return hmac.new(salt, normalise(value, pii_type).encode(), hashlib.sha256).hexdigest()


def _stars(word: str) -> str:
    """'Smith' -> 'S***'; numbers become stars of the same length."""
    return "*" * len(word) if word.isdigit() else word[:1] + "***"


def mask_value(value: str, pii_type: str) -> str:
    """Hide most of a value but keep its shape (see the module docstring)."""
    text = str(value).strip()
    if pii_type == "email" and "@" in text:
        local, _, domain = text.partition("@")
        name, dot, tld = domain.rpartition(".")
        return f"{local[:1]}***@{name[:1]}***{dot}{tld}" if dot else f"{local[:1]}***@***"
    if pii_type in DIGIT_TYPES:
        keep_from = len(re.sub(r"\D", "", text)) - 4      # keep the last 4 digits
        out, seen = [], 0
        for ch in text:
            if ch.isdigit():
                out.append(ch if seen >= keep_from else "*")
                seen += 1
            else:
                out.append(ch)
        return "".join(out)
    if pii_type in ("person_name", "street_address"):
        return " ".join(_stars(w) for w in re.split(r"[\s,]+", text) if w)
    return text[:1] + "***"


def generalize_value(value: str, pii_type: str, today: date) -> str | None:
    """Coarse version of a value; None when it cannot be read (e.g. a broken date)."""
    text = str(value).strip()
    if pii_type == "date_of_birth":
        born = pd.to_datetime(text, errors="coerce")
        if pd.isna(born) or born.date() > today:
            return None
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        low = age // AGE_BAND * AGE_BAND
        return f"{low}-{low + AGE_BAND - 1}"
    if pii_type == "zip_code":
        digits = re.sub(r"\D", "", text)
        return digits[:3] + "**" if len(digits) >= 3 else None
    if pii_type == "ip_address":
        parts = text.split(".")
        return ".".join(parts[:3]) + ".0/24" if len(parts) == 4 else None
    raise PrivacyError(f"generalize is not available for {pii_type}; "
                       f"use it for {sorted(GENERALIZABLE)}")


def protect_column(series: pd.Series, pii_type: str, action: str,
                   salt: bytes | None = None, today: date | None = None) -> pd.Series:
    """Apply one action to one column (not for 'drop'); empty cells stay empty."""
    known = series.notna()
    if action == "keep":
        return series
    if action == "mask":
        func = lambda v: mask_value(v, pii_type)              # noqa: E731
    elif action == "hash":
        func = lambda v: hash_value(v, pii_type, salt or get_salt())   # noqa: E731
    elif action == "generalize":
        func = lambda v: generalize_value(v, pii_type, today or date.today())  # noqa: E731
    else:
        raise PrivacyError(f"unknown PII action {action!r}; use one of {sorted(ACTIONS)}")
    return series.where(~known, series[known].map(func))


def _check_actions(actions: dict[str, str], found: dict[str, dict]) -> None:
    """Fail early (before any data is changed) on a bad setting."""
    for column, action in actions.items():
        if action not in ACTIONS:
            raise PrivacyError(f"{column}: unknown PII action {action!r}; "
                               f"use one of {sorted(ACTIONS)}")
        pii_type = found.get(column, {}).get("pii_type", "manual")
        if action == "generalize" and pii_type not in GENERALIZABLE:
            raise PrivacyError(f"{column}: generalize is not available for {pii_type}; "
                               f"use it for {sorted(GENERALIZABLE)}")


def mask_dataframe(df: pd.DataFrame, actions: dict[str, str] | None = None,
                   today: date | None = None) -> tuple[pd.DataFrame, dict]:
    """Detect PII, protect every PII column, return (protected copy, PII report).

    actions: {column: action} overrides DEFAULT_ACTIONS; it may also name a column the
    detector missed (it is then treated as type "manual" and can be masked/hashed/dropped).
    """
    found = {hit["column"]: hit for hit in detect_pii(df)}
    chosen = {c: DEFAULT_ACTIONS.get(h["pii_type"], "mask") for c, h in found.items()}
    for column in actions or {}:
        if column not in df.columns:
            raise PrivacyError(f"column {column!r} not found")
    chosen.update(actions or {})
    _check_actions(chosen, found)
    salt = get_salt() if "hash" in chosen.values() else None      # fail before changing data

    out = df.copy()
    rows = []
    for column, action in chosen.items():
        hit = found.get(column, {"column": column, "pii_type": "manual", "confidence": 1.0,
                                 "evidence": "chosen by the user"})
        filled = int(df[column].notna().sum())
        if action == "drop":
            out = out.drop(columns=column)
        else:
            out[column] = protect_column(df[column], hit["pii_type"], action, salt, today)
        unreadable = int((out[column].isna() & df[column].notna()).sum()) \
            if action == "generalize" else 0
        rows.append({**hit, "action": action,
                     "cells_changed": 0 if action == "keep" else filled - unreadable,
                     "cells_unreadable": unreadable})
    report = {
        "pii_columns_found": len(found),
        "columns": rows,
        "columns_dropped": [r["column"] for r in rows if r["action"] == "drop"],
        "hash": {"method": "HMAC-SHA256", "salt_from": f"environment variable {SALT_ENV}"}
        if salt else None,
        "note": "this report never contains data values",
    }
    return out, report
