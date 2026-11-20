"""Schema inference (src/cleaning/schema.py)."""
import json

import pandas as pd
import pytest

from src.cleaning.schema import (
    business_rule,
    canonical_forms,
    infer_schema,
    save_schema,
)
from src.profiling.profiler import profile_dataset


def make(data: dict) -> tuple[pd.DataFrame, dict]:
    """A text DataFrame (like read_csv_raw gives) and its profiling report."""
    df = pd.DataFrame(data, dtype=str)
    return df, profile_dataset(df, "t.csv")


def test_expected_types():
    df, rep = make({
        "customer_id": [f"C{i}" for i in range(30)],
        "age": [str(20 + i) for i in range(30)],
        "city": ["London", "Paris", "Rome"] * 10,
        "joined": [f"2024-01-{i + 1:02d}" for i in range(30)],
        "email": [f"u{i}@x.com" for i in range(30)],
    })
    cols = infer_schema(df, rep)["columns"]
    assert cols["customer_id"]["expected_type"] == "id"
    assert cols["age"]["expected_type"] == "numeric"
    assert cols["city"]["expected_type"] == "categorical"
    assert cols["joined"]["expected_type"] == "date"
    assert cols["joined"]["format"] == "YYYY-MM-DD"
    assert cols["email"]["expected_type"] == "text"
    assert cols["email"]["format"] == "email"


def test_numeric_ranges_ignore_impossible_values_and_codes():
    ages = [str(20 + i % 40) for i in range(200)] + ["230", "999"]
    df, rep = make({"age": ages})
    age = infer_schema(df, rep)["columns"]["age"]
    assert age["valid_range"] == [0, 120]                  # business rule
    assert age["range_source"] == "business_rule:age"
    assert age["observed_range"] == [20.0, 59.0]            # 230 and 999 not included
    lo, hi = age["typical_range"]                           # p1-p99
    assert 20 <= lo <= hi <= 59
    assert age["is_integer"] is True


def test_numeric_with_missing_codes_is_still_numeric():
    # 8 of 9 values are numbers (89%, below Module 1's 95%), but "unknown" is a missing
    # code, not text: without it, 100% are numbers -> numeric, so age 230 gets checked
    df, rep = make({"age": ["34", "41", "230", "29", "unknown", "38", "45", "52", "60"]})
    assert infer_schema(df, rep)["columns"]["age"]["expected_type"] == "numeric"


def test_business_rule_matches_whole_words_only():
    assert business_rule("customer_age") == ("age", [0, 120])
    assert business_rule("page_views") is None              # "page" is not "age"
    assert business_rule("emp.var.rate") is None


def test_allowed_values_merge_case_space_and_abbreviation_variants():
    s = pd.Series(["HR"] * 5 + ["hr "] + ["Finance"] * 4 + ["Fin."] + ["Sales"] * 3)
    forms = canonical_forms(s)
    assert forms["hr "] == "HR"
    assert forms["Fin."] == "Finance"
    assert set(forms.values()) == {"HR", "Finance", "Sales"}


def test_abbreviation_not_merged_when_it_is_the_real_value():
    # Bank Marketing has "admin." as a real job and no other value starting with "admin"
    forms = canonical_forms(pd.Series(["admin."] * 5 + ["technician"] * 5))
    assert forms["admin."] == "admin."


def test_missing_codes_are_not_allowed_values():
    df, rep = make({"job": ["admin.", "technician", "unknown"] * 10})
    job = infer_schema(df, rep)["columns"]["job"]
    assert job["missing_codes"] == ["unknown"]
    assert "unknown" not in job["allowed_values"]
    assert job["nullable"] is True


def test_report_from_other_file_is_rejected():
    df, rep = make({"a": ["1", "2"]})
    with pytest.raises(ValueError, match="different file"):
        infer_schema(df.rename(columns={"a": "b"}), rep)


def test_save_schema_writes_json(tmp_path):
    df, rep = make({"a": ["1", "2", "3"]})
    path = save_schema(infer_schema(df, rep), tmp_path)
    assert path.name == "schema.json"
    assert json.loads(path.read_text())["columns"]["a"]["expected_type"] == "numeric"
