"""Sprint 2 — FastAPI profiling endpoint (uses FastAPI's TestClient, no real server needed)."""
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_module
from src.api.app import app
from tests.test_schema import assert_valid

client = TestClient(app)
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "raw" / "bank_marketing_sample.csv"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_profile_upload_returns_report():
    with open(SAMPLE, "rb") as f:
        r = client.post("/profile", files={"file": ("bank.csv", f, "text/csv")})
    assert r.status_code == 200
    report = r.json()
    assert report["dataset"]["name"] == "bank.csv"
    assert report["dataset"]["rows"] == 1000
    assert "issues" in report and "summary" in report


def test_rejects_non_csv():
    r = client.post("/profile", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_empty_file():
    r = client.post("/profile", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_missing_file_is_422():
    assert client.post("/profile").status_code == 422   # FastAPI's "required field missing"


def test_profile_returns_v12_sections_and_matches_schema():
    # 'unknown' and 999 are disguised missing values; row 1 and 2 are duplicates
    csv = b"job,pdays\nadmin,999\nadmin,999\nunknown,3\ntech,999\ntech,5\n"
    r = client.post("/profile", files={"file": ("tiny.csv", csv, "text/csv")})
    assert r.status_code == 200
    report = r.json()
    assert_valid(report)                                     # same contract as the CLI
    cols = {c["name"]: c for c in report["columns"]}
    assert cols["job"]["disguised_missing"]["values"] == {"unknown": 1}
    assert cols["pdays"]["disguised_missing"]["values"] == {"999": 3}
    assert 0 <= report["quality_score"]["overall"] < 100
    actions = {s["action"] for s in report["cleaning_suggestions"]}
    assert "drop_duplicates" in actions


def test_rejects_broken_csv():
    broken = b'a,b\n"1,2\n3,4\n'                            # quote that never closes
    r = client.post("/profile", files={"file": ("broken.csv", broken, "text/csv")})
    assert r.status_code == 400
    assert "Could not read the CSV" in r.json()["detail"]


def test_rejects_too_large_file(monkeypatch):
    monkeypatch.setattr(api_module, "MAX_UPLOAD_MB", 0.0001)   # limit ~100 bytes
    big = b"a,b\n" + b"1,2\n" * 100
    r = client.post("/profile", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413
