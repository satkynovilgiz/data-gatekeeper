"""Sprint 9: the combined validation report matches its JSON Schema; CLI exit codes."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.profiling.profiler import read_csv_raw
from src.validation.contract import load_contract, parse_contract
from src.validation.report import parse_methods, validation_report

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures"
SCHEMA = json.loads((ROOT / "docs/validation_report.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
EXAMPLE = load_contract(ROOT / "contracts/example.yaml")
SAMPLE = ROOT / "data/raw/bank_marketing_sample.csv"


def check(report: dict) -> None:
    errors = sorted(VALIDATOR.iter_errors(json.loads(json.dumps(report))), key=str)
    assert not errors, [f"{list(e.path)}: {e.message}" for e in errors[:5]]


def test_schema_itself_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


def test_contract_only_report_matches_schema():
    ok = read_csv_raw(FIX / "orders_ok.csv")
    report = validation_report(ok, EXAMPLE, "orders_ok")
    check(report)
    assert report["status"] == "pass" and report["drift"] is None
    assert report["anomalies"] is None and report["gates"]["anomalies"] is None


def test_full_report_on_real_data_matches_schema():
    df = read_csv_raw(SAMPLE)
    contract = parse_contract({"columns": [{"name": c} for c in df.columns]})
    report = validation_report(df.iloc[500:], contract, "bank_half", reference=df.iloc[:500],
                               anomalies=True, methods=parse_methods("all"), exclude=["y"])
    check(report)
    assert report["anomalies"]["fitted_on"] == "reference"
    assert set(report["anomalies"]["methods"]) == {"iforest", "zscore", "iqr", "zscore_mixture",
                                                   "iforest_with_categories"}
    # the file is in time order, so the two halves really drift (economic columns)
    assert "euribor3m" in report["drift"]["major_columns"]
    assert "y" not in report["anomalies"]["features"]


def test_failing_reports_match_schema_and_say_why():
    bad = read_csv_raw(FIX / "orders_bad.csv")
    report = validation_report(bad, EXAMPLE, "orders_bad")
    check(report)
    assert report["status"] == "fail" and report["gates"]["contract"] == "fail"
    df = read_csv_raw(SAMPLE)
    loose = parse_contract({"columns": [{"name": c} for c in df.columns]})
    gate = validation_report(df, loose, "bank", anomalies=True, exclude=["y"],
                             fail_on_share=0.005)
    check(gate)
    assert gate["gates"]["contract"] == "pass"
    assert gate["gates"]["anomalies"]["status"] == "fail" and gate["status"] == "fail"


def test_bad_settings():
    df = pd.DataFrame({"a": ["1", "2"]})
    c = parse_contract({"columns": [{"name": "a"}]})
    with pytest.raises(ValueError, match="needs the anomaly section"):
        validation_report(df, c, "d", fail_on_share=0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        validation_report(df, c, "d", anomalies=True, fail_on_share=2)
    with pytest.raises(ValueError, match="method must be"):
        parse_methods("lof")
    assert parse_methods("zscore,iqr") == ("zscore", "iqr")


# ------------------------------------------------------------------- CLI
def gk(*args):
    return subprocess.run([sys.executable, str(ROOT / "gatekeeper.py"), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli_validate_writes_one_report_with_all_sections(tmp_path):
    r = gk("validate", FIX / "orders_ok.csv", "--contract", ROOT / "contracts/example.yaml",
           "--reference", FIX / "orders_ok.csv", "--anomalies", "--reports-dir", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((tmp_path / "orders_ok_validation_report.json").read_text())
    check(report)
    assert report["drift"] and report["anomalies"]
    assert (tmp_path / "orders_ok_contract.json").exists()             # Sprint 7 file kept
    assert "drift" in r.stdout and "anomalies" in r.stdout
    assert "aida@example.com" not in r.stdout + json.dumps(report)


def test_cli_exit_codes(tmp_path):
    contract = tmp_path / "bank.yaml"
    assert gk("contract", "init", SAMPLE, "--out", contract).returncode == 0
    base = ["validate", SAMPLE, "--contract", contract, "--reports-dir", tmp_path]
    assert gk(*base).returncode == 0
    assert gk(*base, "--anomalies", "--exclude", "y").returncode == 0      # never fails alone
    r = gk(*base, "--anomalies", "--exclude", "y", "--fail-on-anomaly-share", "0.005")
    assert r.returncode == 1 and "FAIL" in r.stdout                      # 1% flagged > 0.5%
    assert gk(*base, "--anomalies", "--exclude", "y",
              "--fail-on-anomaly-share", "0.5").returncode == 0
    assert gk("validate", FIX / "orders_bad.csv", "--contract",
              ROOT / "contracts/example.yaml", "--reports-dir", tmp_path).returncode == 1
    r = gk(*base, "--fail-on-anomaly-share", "0.1")                      # without --anomalies
    assert r.returncode == 2 and "needs the anomaly section" in r.stderr
    assert gk(*base, "--anomalies", "--anomaly-method", "lof").returncode == 2


def test_cli_anomalies(tmp_path):
    r = gk("anomalies", ROOT / "data/raw/customers_pii.csv", "--exclude", "churned",
           "--top", "3", "--reports-dir", tmp_path)
    assert r.returncode == 0, r.stderr
    section = json.loads((tmp_path / "customers_pii_anomalies.json").read_text())
    assert section["features"] == ["monthly_spend"]                      # no PII, no ID
    raw = pd.read_csv(ROOT / "data/raw/customers_pii.csv", dtype=str)
    text = r.stdout + json.dumps(section)
    for col in ("email", "phone", "ssn", "full_name", "customer_id"):
        assert not any(v in text for v in raw[col].dropna().head(50)), col
    r = gk("anomalies", ROOT / "data/raw/customers_pii.csv", "--fail-on-anomaly-share", "0",
           "--reports-dir", tmp_path)
    assert r.returncode == 1
    assert gk("anomalies", tmp_path / "missing.csv").returncode == 2


def test_cli_anomalies_new_methods_and_encoding(tmp_path):
    r = gk("anomalies", SAMPLE, "--method", "zscore_mixture,iforest_with_categories",
           "--encoding", "onehot", "--exclude", "y", "--top", "2", "--reports-dir", tmp_path)
    assert r.returncode == 0, r.stderr
    section = json.loads((tmp_path / "bank_marketing_sample_anomalies.json").read_text())
    assert list(section["methods"]) == ["zscore_mixture", "iforest_with_categories"]
    assert section["settings"]["encoding"] == "onehot"
    assert "categorical" in r.stdout
    assert gk("anomalies", SAMPLE, "--encoding", "target").returncode == 2
