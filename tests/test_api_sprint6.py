"""Sprint 6 — /clean, /transform, /mask-pii and the upload rules (FastAPI TestClient)."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.api.app import app
from src.privacy.mask import SALT_ENV

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]
HR = ROOT / "data" / "raw" / "hr_employees_messy.csv"
CUSTOMERS = ROOT / "data" / "raw" / "customers_pii.csv"


def post(path: str, file: Path | bytes, name: str = "data.csv", **params):
    data = file.read_bytes() if isinstance(file, Path) else file
    return client.post(path, files={"file": (name, data, "text/csv")}, params=params)


def small_clean_csv() -> bytes:
    rows = ["age,plan,spend,y"] + [f"{20 + i % 40},{['a', 'b', 'c'][i % 3]},{i * 1.5},"
                                   f"{'yes' if i % 4 == 0 else 'no'}" for i in range(80)]
    return "\n".join(rows).encode()


# ---- /clean ---------------------------------------------------------------------------
def test_clean_returns_log_and_quality():
    r = post("/clean", HR, "hr.csv", imputer="median")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["rows_after"] < body["summary"]["rows_before"]   # duplicates gone
    q = body["quality"]
    assert q["after"]["overall"] > q["before"]["overall"]
    assert "salary_outlier" in q["after"]["flag_columns"]["columns"]       # Sprint 6 fix
    assert len(body["preview"]) == 5


def test_clean_csv_download():
    r = post("/clean", HR, "hr.csv", imputer="median", output="csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert 'filename="cleaned_hr.csv"' in r.headers["content-disposition"]
    df = pd.read_csv(io.StringIO(r.text))
    assert df["employee_id"].is_unique


def test_clean_rejects_unknown_imputer():
    assert post("/clean", HR, imputer="magic").status_code == 422


# ---- /transform -----------------------------------------------------------------------
def test_transform_report():
    r = post("/transform", small_clean_csv(), target="y")
    assert r.status_code == 200
    report = r.json()["report"]
    assert report["rows"] == {"input": 80, "train": 64, "test": 16}
    assert report["fitted_on"].startswith("train rows only")
    assert {c["column"] for c in report["columns"]} == {"age", "plan", "spend"}


def test_transform_csv_has_both_splits():
    r = post("/transform", small_clean_csv(), target="y", output="csv")
    df = pd.read_csv(io.StringIO(r.text))
    assert df["split"].value_counts().to_dict() == {"train": 64, "test": 16}


def test_transform_bad_target_is_400():
    r = post("/transform", small_clean_csv(), target="label")
    assert r.status_code == 400 and "target column" in r.json()["detail"]


def test_transform_bad_test_size_is_422():
    assert post("/transform", small_clean_csv(), test_size=1.5).status_code == 422


# ---- /mask-pii ------------------------------------------------------------------------
def test_mask_pii_report_without_raw_values(monkeypatch):
    monkeypatch.setenv(SALT_ENV, "api-test-salt")
    raw = pd.read_csv(CUSTOMERS, dtype=str)
    r = post("/mask-pii", CUSTOMERS, "customers.csv")
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["pii_columns_found"] == 9
    text = r.text
    for c in ["email", "phone", "ssn", "credit_card", "street_address"]:
        assert raw.at[0, c] not in text                       # not in report, not in preview


def test_mask_pii_csv_with_user_actions(monkeypatch):
    monkeypatch.delenv(SALT_ENV, raising=False)               # no hashing needed below
    r = post("/mask-pii", CUSTOMERS, "customers.csv", output="csv",
             action=["full_name=mask", "email=mask"])
    assert r.status_code == 200
    df = pd.read_csv(io.StringIO(r.text))
    assert "ssn" not in df.columns and df["email"].dropna().str.contains(r"\*\*\*@").all()


def test_mask_pii_without_salt_is_503(monkeypatch):
    monkeypatch.delenv(SALT_ENV, raising=False)
    r = post("/mask-pii", CUSTOMERS)                          # defaults hash email + name
    assert r.status_code == 503 and "action=email=mask" in r.json()["detail"]


def test_mask_pii_bad_action_is_400(monkeypatch):
    monkeypatch.setenv(SALT_ENV, "x")
    assert post("/mask-pii", CUSTOMERS, action=["email=encrypt"]).status_code == 400
    assert post("/mask-pii", CUSTOMERS, action=["email"]).status_code == 400


# ---- upload rules (every endpoint) ------------------------------------------------------
@pytest.mark.parametrize("path", ["/profile", "/clean", "/transform", "/mask-pii"])
def test_every_endpoint_rejects_non_csv_and_empty(path):
    assert post(path, b"a,b\n1,2\n", "notes.txt").status_code == 400
    assert post(path, b"   ", "empty.csv").status_code == 400
    assert client.post(path).status_code == 422                   # no file at all


def test_header_only_csv_is_400():
    r = post("/profile", b"a,b\n")
    assert r.status_code == 400 and "no data rows" in r.json()["detail"]


def test_too_many_rows_is_413(monkeypatch):
    monkeypatch.setattr(api_module, "MAX_ROWS", 10)
    r = post("/profile", b"a\n" + b"1\n" * 11)
    assert r.status_code == 413 and "limit is 10" in r.json()["detail"]


def test_content_length_checked_before_reading(monkeypatch):
    monkeypatch.setattr(api_module, "MAX_UPLOAD_MB", 0.01)            # ~10 KB
    monkeypatch.setattr(api_module, "FORM_OVERHEAD", 0)
    r = post("/clean", b"a,b\n" + b"1,2\n" * 5000)
    assert r.status_code == 413


def test_unexpected_error_is_a_plain_500(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret.person@example.com broke it")    # a value in the message
    monkeypatch.setattr(api_module, "profile_dataset", boom)
    r = TestClient(app, raise_server_exceptions=False).post(
        "/profile", files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal error while processing the file"}
    assert "secret.person" not in r.text and "Traceback" not in r.text
