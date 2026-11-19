"""Sprint 3 — sampling for big files (--sample N) and timing in report["run"]."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.profiling.profiler import profile_dataset, profile_file, sample_rows

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "raw" / "bank_marketing_sample.csv"


def test_sample_rows_is_random_but_repeatable():
    df = pd.DataFrame({"a": [str(i) for i in range(1000)]})
    s1, s2 = sample_rows(df, 100), sample_rows(df, 100)
    assert len(s1) == 100
    assert s1.index.tolist() == s2.index.tolist()          # fixed seed -> same rows
    assert s1.index.tolist() != list(range(100))           # not just the first 100 rows


def test_sample_rows_keeps_small_files():
    df = pd.DataFrame({"a": ["1", "2", "3"]})
    assert sample_rows(df, None) is df
    assert sample_rows(df, 10) is df
    with pytest.raises(ValueError):
        sample_rows(df, 0)


def test_run_section_without_sample():
    run = profile_dataset(pd.DataFrame({"a": ["1", "2"]}))["run"]
    assert run["rows_in_file"] == run["rows_profiled"] == 2
    assert run["sampled"] is False
    assert run["seconds"] >= 0


def test_profile_file_with_sample():
    r = profile_file(SAMPLE, sample=200)
    assert r["dataset"]["rows"] == 200
    assert r["run"] == {**r["run"], "rows_in_file": 1000, "rows_profiled": 200, "sampled": True}


def test_cli_sample_flag(tmp_path):
    out = subprocess.run([sys.executable, "run_profiling.py", "--input", str(SAMPLE),
                          "--output", str(tmp_path), "--sample", "300"],
                         cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "profiling_report.json").read_text())
    assert report["run"]["rows_profiled"] == 300 and report["run"]["sampled"]
    assert "Profiling took" in out.stderr                  # timing is logged


def test_cli_rejects_bad_sample(tmp_path):
    out = subprocess.run([sys.executable, "run_profiling.py", "--input", str(SAMPLE),
                          "--output", str(tmp_path), "--sample", "0"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode != 0
