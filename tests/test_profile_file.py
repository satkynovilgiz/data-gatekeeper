"""Sprint 2 — profile_file(path): read a CSV from disk and profile it."""
import json
from pathlib import Path

import pytest

from src.profiling.profiler import detect_separator, profile_file

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "raw" / "bank_marketing_sample.csv"


@pytest.mark.parametrize("sep", [",", ";", "\t"])
def test_detects_separator(tmp_path, sep):
    f = tmp_path / "data.csv"
    f.write_text(sep.join(["id", "city", "amount"]) + "\n" + sep.join(["1", "Paris", "9.5"]) + "\n")
    assert detect_separator(f) == sep
    report = profile_file(f)
    assert report["dataset"]["columns"] == 3        # columns split correctly


def test_profile_file_uses_file_name(tmp_path):
    f = tmp_path / "people.csv"
    f.write_text("name,age\nAna,30\nBob,41\n")
    r = profile_file(f)
    assert r["dataset"]["name"] == "people.csv"
    assert r["dataset"]["rows"] == 2


def test_report_is_plain_json(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,x\n2,y\n3,x\n")
    r = profile_file(f)
    assert json.loads(json.dumps(r)) == r           # no numpy types left inside


def test_real_bank_sample():
    r = profile_file(SAMPLE)
    assert r["dataset"]["rows"] == 1000 and r["dataset"]["columns"] == 21
    rules = {(i["column"], i["rule"]) for i in r["issues"]}
    assert ("pdays", "disguised_missing") in rules        # pdays = 999 = "never contacted"
    assert ("default", "disguised_missing") in rules      # 'unknown' = hidden missing
    assert r["summary"]["pii_columns"] == []              # dataset is anonymised
    ids = [c["name"] for c in r["columns"] if c["semantic"]["semantic_type"] == "id"]
    assert ids == []                                      # cons.price.idx is not an ID
