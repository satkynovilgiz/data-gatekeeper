"""The whole cleaning pipeline: clean_file(), run_cleaning.py, cleaning_log.json, schema.json."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from run_cleaning import parse_impute
from src.cleaning.cleaner import STEPS, clean_file
from src.profiling.profiler import profile_file, save_report

ROOT = Path(__file__).resolve().parents[1]

# A tiny file with one of every problem (";" separated, like the bank data)
MESSY_CSV = """id;dept;age;years;hired;email;pdays
E1;HR;34;5;2020-01-15;a@x.com;999
E2;hr ;41;7;2019-03-02;b@x.com;999
E3;Sales;230;3;20/09/2023;c@x.com;3
E4;Fin.;29;five;2021-07-30;not-an-email;999
E5;Finance;unknown;2;2018-11-05;e@x.com;6
E6;Finance;38;?;2022-02-14;f@x.com;999
E6;Finance;38;?;2022-02-14;f@x.com;999
E7;Sales;45;10;2017-05-20;g@x.com;999
E8;HR;52;12;2016-09-09;h@x.com;999
"""


@pytest.fixture
def messy(tmp_path: Path) -> Path:
    path = tmp_path / "messy.csv"
    path.write_text(MESSY_CSV)
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run `python run_cleaning.py <args>` from the repo root and capture its output."""
    return subprocess.run([sys.executable, "run_cleaning.py", *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_raw_file_is_never_modified(messy, tmp_path):
    before = sha256(messy), messy.stat().st_mtime_ns
    clean_file(messy, tmp_path / "out")
    assert (sha256(messy), messy.stat().st_mtime_ns) == before


def test_output_cannot_overwrite_input(tmp_path):
    path = tmp_path / "cleaned_data.csv"
    path.write_text("a\n1\n2\n")
    with pytest.raises(ValueError, match="overwrite the input"):
        clean_file(path, tmp_path)
    assert path.read_text() == "a\n1\n2\n"


def test_end_to_end_cli_on_tiny_messy_csv(messy, tmp_path):
    out = tmp_path / "out"
    result = run_cli("--input", str(messy), "--output", str(out))
    assert result.returncode == 0, result.stderr
    assert "Quality score" in result.stdout

    # no --profile -> Module 1 ran first and saved its report
    assert (out / "profiling_report.json").exists()
    df = pd.read_csv(out / "cleaned_data.csv", dtype=str)
    log = json.loads((out / "cleaning_log.json").read_text())
    schema = json.loads((out / "schema.json").read_text())

    assert len(df) == 8                                     # one duplicate (E6) removed
    assert sorted(df["dept"].unique()) == ["Finance", "HR", "Sales"]
    assert df.loc[df["id"] == "E3", "hired"].item() == "2023-09-20"
    assert df.loc[df["id"] == "E4", "email"].isna().item()  # invalid email -> empty, not invented
    assert df["age"].notna().all() and "230" not in set(df["age"])
    assert df["years"].notna().all()                         # "five" and "?" -> NaN -> median
    assert df["previously_contacted"].tolist() == ["no", "no", "yes", "no", "yes",
                                                   "no", "no", "no"]
    assert df["pdays"].dropna().tolist() == ["3", "6"]       # 999 -> empty, not imputed

    s = log["summary"]
    assert (s["rows_before"], s["rows_after"]) == (9, 8)
    assert s["duplicates_removed"] == 1
    assert s["columns_added"] == ["previously_contacted"]
    assert s["impossible_values_fixed"] == 1
    assert s["not_a_number_to_nan"] == 1                     # "five" ("?" is a missing code)
    assert s["invalid_format_to_nan"] == 1
    assert log["quality"]["after"]["overall"] > log["quality"]["before"]["overall"]

    assert schema["columns"]["dept"]["allowed_values"] == ["Finance", "HR", "Sales"]
    assert schema["columns"]["age"]["valid_range"] == [0, 120]
    assert schema["columns"]["hired"]["format"] == "YYYY-MM-DD"


def test_cli_uses_a_given_profile(messy, tmp_path):
    report_dir = tmp_path / "profile"
    save_report(profile_file(messy), report_dir)
    out = tmp_path / "out"
    result = run_cli("--input", str(messy), "--profile", str(report_dir / "profiling_report.json"),
                     "--output", str(out), "--impute", "age=none")
    assert result.returncode == 0, result.stderr
    assert not (out / "profiling_report.json").exists()     # profiling did not run again
    df = pd.read_csv(out / "cleaned_data.csv", dtype=str)
    assert df["age"].isna().sum() == 2                       # 230 and "unknown" left empty


def test_cli_errors(messy, tmp_path):
    assert run_cli("--input", str(tmp_path / "nope.csv")).returncode != 0
    bad = run_cli("--input", str(messy), "--output", str(tmp_path), "--impute", "age=magic")
    assert bad.returncode != 0 and "STRATEGY" in bad.stderr
    assert not (tmp_path / "cleaned_data.csv").exists()


def test_parse_impute():
    assert parse_impute(["age=mean", "a=b=none"]) == {"age": "mean", "a=b": "none"}
    with pytest.raises(ValueError):
        parse_impute(["age"])


def test_cleaning_log_format(messy, tmp_path):
    log = clean_file(messy, tmp_path)
    assert log["step_order"] == [s.__name__ for s in STEPS]
    for entry in log["steps"]:
        assert set(entry) == {"step", "column", "rule", "cells_changed", "rows_changed",
                              "examples"}
        assert 1 <= len(entry["examples"]) <= 3
    # every Module 1 suggestion is accounted for
    statuses = {s["status"] for s in log["suggestions"]}
    assert statuses <= {"done", "skipped", "nothing_to_do"}
    assert all("reason" in s for s in log["suggestions"] if s["status"] != "done")
    # quality before / after / delta for all four dimensions
    q = log["quality"]
    for dim in ("completeness", "uniqueness", "validity", "consistency"):
        assert q["delta"][dim] == pytest.approx(
            q["after"]["dimensions"][dim] - q["before"]["dimensions"][dim], abs=0.01)
    # the file on disk is strict JSON (no bare NaN values) and equals what was returned
    def no_nan(token: str) -> None:
        raise AssertionError(f"{token} in cleaning_log.json")
    on_disk = json.loads((tmp_path / "cleaning_log.json").read_text(), parse_constant=no_nan)
    assert on_disk["summary"] == log["summary"]
