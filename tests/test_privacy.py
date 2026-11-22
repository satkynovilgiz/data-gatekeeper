"""Sprint 6 — PII detection and masking (src/privacy/)."""
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.privacy.detect import detect_column, detect_pii
from src.privacy.mask import (
    SALT_ENV,
    PrivacyError,
    generalize_value,
    hash_value,
    mask_dataframe,
    mask_value,
)

ROOT = Path(__file__).resolve().parents[1]
CUSTOMERS = ROOT / "data" / "raw" / "customers_pii.csv"
TODAY = date(2026, 11, 21)
PII_COLUMNS = ["full_name", "email", "phone", "ssn", "credit_card", "ip_address",
               "street_address", "zip_code", "date_of_birth"]


@pytest.fixture
def salt(monkeypatch):
    monkeypatch.setenv(SALT_ENV, "test-only-salt")


def customers() -> pd.DataFrame:
    return pd.read_csv(CUSTOMERS, dtype=str)


# ---- detection --------------------------------------------------------------------
def test_finds_every_fake_pii_column_and_nothing_else():
    found = {h["column"]: h for h in detect_pii(customers())}
    assert list(found) == PII_COLUMNS
    assert {h["pii_type"] for h in found.values()} == {
        "person_name", "email", "phone", "ssn", "credit_card", "ip_address",
        "street_address", "zip_code", "date_of_birth"}
    assert all(h["confidence"] >= 0.9 for h in found.values())


def test_content_alone_is_enough_for_strong_patterns():
    s = pd.Series(["a.b@example.com", "c@example.org", "d@example.net"])
    hit = detect_column("contact", s)                      # no hint in the name
    assert hit["pii_type"] == "email" and hit["confidence"] == 0.7


def test_luhn_separates_cards_from_random_digits():
    cards = pd.Series(["4242 4242 4242 4242", "5555-5555-5555-4444", "6011111111111117"])
    fakes = pd.Series(["4242 4242 4242 4241", "1234-5678-9012-3456", "6011111111111118"])
    assert detect_column("value", cards)["pii_type"] == "credit_card"
    assert detect_column("value", fakes) is None


def test_zip_and_birthday_need_the_column_name():
    numbers = pd.Series(["82114", "65000", "91234"])
    dates = pd.Series(["1990-01-02", "1985-05-06", "2000-07-08"])
    assert detect_column("salary", numbers) is None        # a number, not a zip code
    assert detect_column("zip", numbers)["pii_type"] == "zip_code"
    assert detect_column("hire_date", dates) is None       # a date, not a birthday
    assert detect_column("dob", dates)["pii_type"] == "date_of_birth"


def test_categories_and_dates_are_not_phones_or_names():
    df = pd.DataFrame({"department": ["Human Resources", "Sales Team"] * 20,
                       "hire_date": ["2020-01-15"] * 40,
                       "job": ["blue-collar", "admin."] * 20})
    assert detect_pii(df) == []


def test_bank_marketing_has_no_pii():
    bank = pd.read_csv(ROOT / "data" / "raw" / "bank_marketing_sample.csv", dtype=str)
    assert detect_pii(bank) == []


# ---- strategies ---------------------------------------------------------------------
def test_mask_keeps_the_shape():
    assert mask_value("john.smith@gmail.com", "email") == "j***@g***.com"
    assert mask_value("+1-408-555-0101", "phone") == "+*-***-***-0101"
    assert mask_value("4242 4242 4242 4242", "credit_card") == "**** **** **** 4242"
    assert mask_value("Garcia, Maria", "person_name") == "G*** M***"


def test_hash_is_consistent_so_joins_still_work():
    salt = b"s1"
    assert hash_value(" John@Example.com", "email", salt) == hash_value("john@example.com",
                                                                        "email", salt)
    assert hash_value("4242-4242-4242-4242", "credit_card", salt) == \
        hash_value("4242 4242 4242 4242", "credit_card", salt)
    assert hash_value("john@example.com", "email", b"s2") != hash_value(
        "john@example.com", "email", salt)                  # another salt, another code
    assert len(hash_value("x", "email", salt)) == 64


def test_hash_join_across_two_tables(salt):
    a = pd.DataFrame({"email": ["ann@example.com", "bo@example.com"], "spend": ["1", "2"]})
    b = pd.DataFrame({"email": ["BO@example.com ", "ann@example.com"], "plan": ["pro", "basic"]})
    ma, _ = mask_dataframe(a)
    mb, _ = mask_dataframe(b)
    joined = ma.merge(mb, on="email")
    assert len(joined) == 2 and set(joined["plan"]) == {"pro", "basic"}


def test_hash_without_salt_fails_before_changing_anything(monkeypatch):
    monkeypatch.delenv(SALT_ENV, raising=False)
    with pytest.raises(PrivacyError, match=SALT_ENV):
        mask_dataframe(customers())


def test_generalize():
    assert generalize_value("1980-02-20", "date_of_birth", TODAY) == "40-49"
    assert generalize_value("1980-12-31", "date_of_birth", TODAY) == "40-49"   # 45, birthday later
    assert generalize_value("66067", "zip_code", TODAY) == "660**"
    assert generalize_value("192.0.2.59", "ip_address", TODAY) == "192.0.2.0/24"
    assert generalize_value("not a date", "date_of_birth", TODAY) is None
    with pytest.raises(PrivacyError):
        generalize_value("x", "email", TODAY)


def test_default_actions_and_report(salt):
    out, report = mask_dataframe(customers(), today=TODAY)
    actions = {r["column"]: r["action"] for r in report["columns"]}
    assert actions["ssn"] == actions["credit_card"] == actions["street_address"] == "drop"
    assert {"ssn", "credit_card", "street_address"}.isdisjoint(out.columns)
    assert out["email"].isna().sum() == customers()["email"].isna().sum()     # gaps stay gaps
    assert set(out["date_of_birth"]) <= {f"{a}-{a + 9}" for a in range(0, 100, 10)}
    assert report["pii_columns_found"] == 9
    assert report["hash"]["salt_from"] == f"environment variable {SALT_ENV}"
    assert set(report["columns"][0]) == {"column", "pii_type", "confidence", "evidence",
                                         "action", "cells_changed", "cells_unreadable"}


def test_no_raw_pii_anywhere_in_output_or_report(salt):
    # The most important privacy test: not one original value survives.
    raw = customers()
    out, report = mask_dataframe(raw, today=TODAY)
    report_text = json.dumps(report)
    for c in PII_COLUMNS:
        for i, value in raw[c].dropna().items():
            assert value not in report_text, f"raw {c} value in the report"
            if c in out.columns:      # each cell vs. its own original (a 5-digit zip can
                assert value not in str(out.at[i, c]), f"raw {c} leaked"   # hide in a hash)


def test_user_actions_override_and_bad_ones_are_rejected(salt):
    df = customers()
    out, report = mask_dataframe(df, {"email": "mask", "plan": "drop"})
    assert out["email"].dropna().str.contains(r"\*\*\*@").all()
    assert "plan" not in out.columns                           # a column the detector missed
    with pytest.raises(PrivacyError, match="unknown PII action"):
        mask_dataframe(df, {"email": "encrypt"})
    with pytest.raises(PrivacyError, match="generalize is not available"):
        mask_dataframe(df, {"email": "generalize"})
    with pytest.raises(PrivacyError, match="not found"):
        mask_dataframe(df, {"nope": "drop"})


def test_errors_never_contain_values(salt):
    df = pd.DataFrame({"email": ["secret.person@example.com"] * 3})
    with pytest.raises(PrivacyError) as err:
        mask_dataframe(df, {"email": "generalize"})
    assert "secret.person" not in str(err.value)


def test_masking_cli(tmp_path, monkeypatch):
    import subprocess
    import sys
    monkeypatch.setenv(SALT_ENV, "cli-salt")
    r = subprocess.run([sys.executable, str(ROOT / "run_masking.py"), "--input", str(CUSTOMERS),
                        "--output", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "PII columns found: 9" in r.stdout
    assert (tmp_path / "masked_data.csv").exists() and (tmp_path / "pii_report.json").exists()
