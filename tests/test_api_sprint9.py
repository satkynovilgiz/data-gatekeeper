"""Sprint 9 — POST /validate, /drift, /anomalies with the Sprint 6 upload rules (TestClient)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from src.api import app as api_module
from src.api.app import app

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
EXAMPLE = (ROOT / "contracts" / "example.yaml").read_bytes()
SAMPLE = ROOT / "data" / "raw" / "bank_marketing_sample.csv"
CUSTOMERS = ROOT / "data" / "raw" / "customers_pii.csv"
SCHEMA = json.loads((ROOT / "docs" / "validation_report.schema.json").read_text())


def csv(path: Path, name: str | None = None) -> tuple:
    return (name or path.name, path.read_bytes(), "text/csv")


def yaml_file(data: bytes = EXAMPLE, name: str = "contract.yaml") -> tuple:
    return (name, data, "application/x-yaml")


# --------------------------------------------------------------- /validate
def test_validate_pass_and_fail_reports_match_schema():
    ok = client.post("/validate", files={"file": csv(FIX / "orders_ok.csv"),
                                         "contract": yaml_file()})
    assert ok.status_code == 200
    assert ok.json()["status"] == "pass"
    Draft202012Validator(SCHEMA).validate(ok.json())
    bad = client.post("/validate", files={"file": csv(FIX / "orders_bad.csv"),
                                          "contract": yaml_file()})
    assert bad.status_code == 200 and bad.json()["status"] == "fail"   # data result, not error
    failed = {(r["rule"], r["column"]) for r in bad.json()["contract"]["results"]
              if r["status"] == "fail"}
    assert ("allowed_values", "country") in failed
    assert "carlos.at.example.com" not in bad.text                    # PII stays masked


def test_validate_with_reference_and_anomalies():
    r = client.post("/validate",
                    files={"file": csv(FIX / "orders_ok.csv"), "contract": yaml_file(),
                           "reference": csv(FIX / "orders_ok.csv", "ref.csv")},
                    params={"anomalies": "true", "exclude": ["paid"]})
    assert r.status_code == 200
    body = r.json()
    Draft202012Validator(SCHEMA).validate(body)
    assert body["drift"]["status"] == "stable"
    assert body["anomalies"]["fitted_on"] == "reference"
    assert "order_id" not in body["anomalies"]["features"]             # an ID, never used
    assert "aida@example.com" not in r.text


def test_validate_bad_contract_and_upload_rules(monkeypatch):
    def send(contract=None, data=FIX / "orders_ok.csv", name=None, **params):
        return client.post("/validate", files={"file": csv(data, name),
                                               "contract": contract or yaml_file()},
                           params=params)
    r = send(yaml_file(b"columns:\n  - name: a\n    dtype: int\n"))
    assert r.status_code == 400 and "dtype must be one of" in r.json()["detail"]
    assert send(yaml_file(b"columns: [\n")).status_code == 400
    assert send(yaml_file(name="contract.txt")).status_code == 400
    assert send(name="orders.txt").status_code == 400                  # .csv only
    r = send(fail_on_share="0.1")                                      # needs anomalies=true
    assert r.status_code == 400 and "anomaly section" in r.json()["detail"]
    monkeypatch.setattr(api_module, "MAX_ROWS", 3)
    assert send().status_code == 413                                   # Sprint 6 row limit
    monkeypatch.setattr(api_module, "MAX_CONTRACT_KB", 0)
    assert send().status_code == 413


def test_validate_anomaly_gate():
    loose = "columns:\n" + "".join(f"  - name: {c}\n"
                                   for c in pd.read_csv(SAMPLE, sep=";", nrows=1).columns)
    r = client.post("/validate", files={"file": csv(SAMPLE), "contract": yaml_file(loose.encode())},
                    params={"anomalies": "true", "exclude": ["y"], "fail_on_share": "0.005"})
    assert r.status_code == 200
    assert r.json()["gates"]["anomalies"]["status"] == "fail" and r.json()["status"] == "fail"


# ------------------------------------------------------------------ /drift
def test_drift_endpoint():
    r = client.post("/drift", files={"reference": csv(FIX / "orders_ok.csv"),
                                     "current": csv(FIX / "orders_ok.csv", "new.csv")})
    assert r.status_code == 200
    assert r.json()["status"] == "stable" and r.json()["current"]["name"] == "new.csv"
    shifted = "x\n" + "\n".join(str(i + 500) for i in range(60))
    base = "x\n" + "\n".join(str(i) for i in range(60))
    r = client.post("/drift", files={"reference": ("ref.csv", base.encode(), "text/csv"),
                                     "current": ("cur.csv", shifted.encode(), "text/csv")},
                    )
    assert r.json()["major_columns"] == ["x"]
    r = client.post("/drift", files={"reference": csv(CUSTOMERS),
                                     "current": csv(FIX / "orders_ok.csv")})
    assert r.status_code == 200
    raw = pd.read_csv(CUSTOMERS, dtype=str)
    assert not any(v in r.text for v in raw["email"].dropna().head(50))
    assert client.post("/drift", files={"reference": csv(FIX / "orders_ok.csv")}).status_code == 422


# -------------------------------------------------------------- /anomalies
def test_anomalies_endpoint():
    r = client.post("/anomalies", files={"file": csv(SAMPLE)},
                    params={"exclude": ["y"], "top": 3})
    assert r.status_code == 200
    body = r.json()
    assert set(body["methods"]) == {"iforest", "zscore", "iqr"}
    assert "y" not in body["features"] and len(body["methods"]["iforest"]["top"]) == 3
    assert body["methods"]["iforest"]["flagged"] == 10                 # 1% of 1,000 rows
    assert client.post("/anomalies", files={"file": csv(SAMPLE)},
                       params={"method": "lof"}).status_code == 400
    assert client.post("/anomalies", files={"file": csv(SAMPLE)},
                       params={"contamination": "0.9"}).status_code == 422
    r = client.post("/anomalies", files={"file": csv(CUSTOMERS)})
    assert r.status_code == 200 and r.json()["features"] == ["monthly_spend"]
    raw = pd.read_csv(CUSTOMERS, dtype=str)
    for col in ("email", "phone", "customer_id"):
        assert not any(v in r.text for v in raw[col].dropna().head(50))
