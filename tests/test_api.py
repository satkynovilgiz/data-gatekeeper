"""Sprint 2 — FastAPI profiling endpoint (uses FastAPI's TestClient, no real server needed)."""
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.profiling_api import app

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
