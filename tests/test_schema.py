"""Sprint 2 — every report must match docs/profiling_report.schema.json."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.profiling.profiler import profile_dataset, profile_file

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "docs" / "profiling_report.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def assert_valid(report):
    errors = [f"{list(e.path)}: {e.message}" for e in VALIDATOR.iter_errors(report)]
    assert not errors, "\n".join(errors[:10])


def test_schema_itself_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("csv", ["bank_marketing_sample.csv", "hr_employees_messy.csv"])
def test_real_files_match_schema(csv):
    assert_valid(profile_file(ROOT / "data" / "raw" / csv))


def test_edge_cases_match_schema():
    df = pd.DataFrame({
        "empty": [None, None, None],            # all missing
        "one_number": ["5", None, None],        # std/skew cannot be computed -> null
        "card": ["4111111111111111"] * 3,       # PII + constant
        "dob": ["1990-01-01", "1991-02-02", "1992-03-03"],
    })
    assert_valid(profile_dataset(df, "edge.csv"))


def test_cli_report_with_charts_matches_schema(tmp_path):
    subprocess.run([sys.executable, "run_profiling.py",
                    "--input", "data/raw/bank_marketing_sample.csv", "--output", str(tmp_path)],
                   cwd=ROOT, check=True, capture_output=True)
    report = json.loads((tmp_path / "profiling_report.json").read_text())
    assert "charts" in report
    assert "pdays" in report["charts"]["histograms"]
    assert "job" in report["charts"]["bar_charts"]
    assert_valid(report)


def test_schema_catches_broken_report():
    good = profile_dataset(pd.DataFrame({"a": ["1", "2"]}))
    bad = copy.deepcopy(good)
    bad["issues"].append({"column": "a", "rule": "made_up_rule", "severity": "urgent", "detail": "x"})
    assert list(VALIDATOR.iter_errors(bad))
    bad = copy.deepcopy(good)
    del bad["dataset"]["rows"]
    assert list(VALIDATOR.iter_errors(bad))
