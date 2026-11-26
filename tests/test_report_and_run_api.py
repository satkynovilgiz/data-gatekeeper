"""Sprint 11: the HTML report, run.log, `gatekeeper report` and POST /run."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.api.app import app
from src.pipeline.html import build_report
from src.pipeline.run import parse_config, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
CUSTOMERS = ROOT / "data" / "raw" / "customers_pii.csv"
PII_COLUMNS = ("full_name", "email", "phone", "ssn", "credit_card", "street_address")


def pii_values(n: int = 60) -> list[str]:
    raw = pd.read_csv(CUSTOMERS, dtype=str)
    return [v for c in PII_COLUMNS for v in raw[c].dropna().head(n)]


@pytest.fixture(scope="module")
def pii_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    cfg = parse_config({"input": str(CUSTOMERS), "reports_dir": str(out), "steps": [
        "profile", {"clean": {"imputer": "median"}},
        {"mask_pii": {"actions": {"email": "mask", "full_name": "mask"}}},
        "quality", {"anomalies": {"exclude": ["churned"]}}]})
    code, summary, run_dir = run_pipeline(cfg)
    return code, summary, run_dir


def test_report_is_written_self_contained_and_readable(pii_run):
    code, _, run_dir = pii_run
    assert code == 0
    page = (run_dir / "report.html").read_text()
    assert page.startswith("<!doctype html>") and "</html>" in page
    assert "This file passed every check that was run" in page
    for needle in ("What was done", "Decisions for a person", "Data quality score",
                   "does not mean a better model", "Most unusual rows", "What cleaning changed",
                   "@media print"):
        assert needle in page, needle
    lowered = page.lower()
    for external in ("http", "<script", " src=", "<link", "@import", "url("):
        assert external not in lowered, external                 # nothing is loaded


def test_report_never_shows_raw_pii(pii_run):
    _, _, run_dir = pii_run
    page = (run_dir / "report.html").read_text()
    leaked = [v for v in pii_values() if v in page]
    assert not leaked, leaked[:3]


def test_run_log_has_levels_and_timings_but_no_pii(pii_run):
    _, summary, run_dir = pii_run
    text = (run_dir / "run.log").read_text()
    assert "INFO" in text and "DEBUG" in text
    assert "step profile passed in" in text and "step anomalies passed in" in text
    assert f"run {summary['name']}: passed (exit 0)" in text
    assert not [v for v in pii_values() if v in text]


def test_stopped_run_says_so_in_plain_words(tmp_path):
    cfg = parse_config({"input": str(FIX / "orders_bad.csv"), "reports_dir": str(tmp_path),
                        "steps": ["profile", {"validate": {"contract":
                                                           str(ROOT / "contracts/example.yaml")}},
                                  "clean"]})
    code, _, run_dir = run_pipeline(cfg)
    page = build_report(run_dir)
    assert code == 1 and "STOPPED" in page and "Check the file against its contract" in page
    assert "allowed_values" in page                              # the broken rule is shown
    assert "carlos.at.example.com" not in page                   # masked PII stays masked


def test_report_cli(tmp_path, pii_run):
    _, _, run_dir = pii_run
    r = subprocess.run([sys.executable, str(ROOT / "gatekeeper.py"), "report", str(run_dir)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "report.html" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "gatekeeper.py"), "report", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "summary.json" in r.stderr


# --------------------------------------------------------------- POST /run
client = TestClient(app)
GOOD = {"steps": ["profile", {"validate": {"contract": "example"}},
                  {"clean": {"imputer": "median"}}, "quality"]}


def post_run(config, data=None, name="orders.csv", **params):
    body = config if isinstance(config, bytes) else json.dumps(config).encode()
    data = data if data is not None else (FIX / "orders_ok.csv").read_bytes()
    return client.post("/run", params=params, files={
        "file": (name, data, "text/csv"), "config": ("config.json", body, "application/json")})


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "API_RUNS_DIR", tmp_path / "api_runs")
    return tmp_path / "api_runs"


def test_post_run_success_and_report_link(runs_dir):
    r = post_run(GOOD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "passed" and body["exit_code"] == 0
    assert [s["status"] for s in body["summary"]["steps"]] == ["passed"] * 4
    page = client.get(body["report_url"])
    assert page.status_code == 200 and "Data Gatekeeper report" in page.text
    assert not list(runs_dir.rglob("*.csv"))                      # no data kept on the server
    assert not list(runs_dir.rglob("cleaning_log.json"))
    html = post_run(GOOD, output="html")
    assert html.status_code == 200 and html.text.startswith("<!doctype html>")
    assert client.get("/runs/../../etc/report.html").status_code == 404
    assert client.get("/runs/" + "0" * 32 + "/report.html").status_code == 404


def test_post_run_gate_is_a_result_not_an_error():
    r = post_run(GOOD, data=(FIX / "orders_bad.csv").read_bytes())
    assert r.status_code == 200
    assert r.json()["exit_code"] == 1 and r.json()["status"] == "stopped at a gate"


@pytest.mark.parametrize("config, message", [
    (b"{not json", "not valid JSON"),
    ([1, 2], "JSON object"),
    ({"steps": ["profile"], "input": "/etc/passwd"}, "set by the server"),
    ({"steps": [{"drift": {"reference": "x.csv"}}]}, "not available over HTTP"),
    ({"steps": ["impact"]}, "not available over HTTP"),
    ({"steps": [{"validate": {"contract": "../../secret"}}]}, "name of a contract"),
    ({"steps": [{"validate": {"contract": "missing_one"}}]}, "not found"),
    ({"steps": ["clean", "profile"]}, "must keep this order"),
])
def test_post_run_bad_config(config, message):
    r = post_run(config)
    assert r.status_code == 400 and message in r.json()["detail"], r.text


def test_post_run_limits(monkeypatch):
    monkeypatch.setattr(api_module, "MAX_ROWS", 3)
    assert post_run(GOOD).status_code == 413                       # Sprint 6 row limit
    monkeypatch.setattr(api_module, "MAX_ROWS", 200_000)
    monkeypatch.setattr(api_module, "MAX_CONFIG_KB", 0)
    assert post_run(GOOD).status_code == 413
    monkeypatch.setattr(api_module, "MAX_CONFIG_KB", 16)
    assert post_run(GOOD, name="orders.txt").status_code == 400
    monkeypatch.setattr(api_module, "RUN_TIMEOUT_S", 0.01)
    r = post_run(GOOD)
    assert r.status_code == 504 and "longer than" in r.json()["detail"]
