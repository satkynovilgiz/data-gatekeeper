"""Unit tests for Module 1. Run with:  pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from src.profiling.profiler import profile_dataset, save_report
from src.profiling.semantic import detect_semantic, infer_basic_type


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(20)],
        "email": ["a@x.com"] * 18 + ["bad-email", None],
        "signup_date": ["2024-01-05"] * 15 + ["05/01/2024"] * 5,
        "age": ["25"] * 17 + ["thirty", "300", None],
        "city": ["London"] * 10 + ["london "] * 5 + ["Paris"] * 5,
    })


def test_basic_types():
    assert infer_basic_type(pd.Series(["1", "2", "3.5"])) == "numeric"
    assert infer_basic_type(pd.Series(["2024-01-01", "2024-02-01"])) == "datetime"
    assert infer_basic_type(pd.Series([None, None])) == "empty"


def test_semantic_email_and_date():
    s = pd.Series(["a@x.com", "b@y.org", "c@z.net"])
    assert detect_semantic("contact", s, "text")["semantic_type"] == "email"
    d = pd.Series(["2024-01-05", "2023-12-31"])
    assert detect_semantic("signup_date", d, "datetime")["semantic_type"] == "date"


def test_report_structure(messy_df):
    r = profile_dataset(messy_df, "test.csv")
    for key in ("dataset", "columns", "missing_matrix", "correlation", "issues", "summary"):
        assert key in r
    assert r["dataset"]["rows"] == 20
    assert len(r["columns"]) == 5


def test_detects_problems(messy_df):
    r = profile_dataset(messy_df, "test.csv")
    rules = {(i["column"], i["rule"]) for i in r["issues"]}
    assert ("age", "mixed_types") in rules
    assert ("city", "inconsistent_category_format") in rules
    assert ("signup_date", "inconsistent_date_format") in rules
    assert "email" in r["summary"]["pii_columns"]


def test_duplicates_detected():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    r = profile_dataset(df)
    assert r["dataset"]["duplicate_rows"] == 1


def test_report_is_valid_json(tmp_path, messy_df):
    r = profile_dataset(messy_df, "test.csv")
    path = save_report(r, tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["dataset"]["name"] == "test.csv"

