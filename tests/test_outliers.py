"""Outlier detection and actions (src/cleaning/outliers.py + handle_outliers step)."""
import numpy as np
import pandas as pd
import pytest

from src.cleaning.cleaner import clean_dataset
from src.cleaning.outliers import iqr_fences, iqr_outliers, isolation_forest_rows
from src.profiling.profiler import profile_dataset


def staff(n: int = 100) -> pd.DataFrame:
    """Salaries around 80k with one 9,999,999 (row 7), like the HR sample."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"age": rng.integers(25, 60, n).astype(str),
                       "salary": rng.normal(80_000, 8_000, n).round().astype(int).astype(str)})
    df.loc[7, "salary"] = "9999999"
    return df


def clean(df: pd.DataFrame, **config) -> tuple[pd.DataFrame, dict]:
    return clean_dataset(df, profile_dataset(df, "t.csv"), config)


def test_iqr_fences():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100], dtype=float)
    lo, hi = iqr_fences(s)                                     # Q1 = 3, Q3 = 7, IQR = 4
    assert (lo, hi) == (-3.0, 13.0)
    mask, _ = iqr_outliers(s)
    assert mask.tolist() == [False] * 8 + [True]


def test_iqr_not_used_when_half_the_values_are_equal():
    assert iqr_fences(pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 3], dtype=float)) is None


def test_isolation_forest_finds_the_unusual_row():
    df = staff().astype(float)
    rows = isolation_forest_rows(df, ["age", "salary"], share=0.01)
    assert rows.sum() == 1 and rows[7]


def test_default_action_flags_and_keeps_the_value():
    cleaned, log = clean(staff())
    assert len(cleaned) == 100                                 # no row deleted
    assert cleaned.loc[7, "salary"] == 9_999_999               # value kept
    assert cleaned.loc[7, "salary_outlier"] == 1
    cols = list(cleaned.columns)
    assert cols[cols.index("salary") + 1] == "salary_outlier"
    assert cleaned.loc[7, "is_anomaly"] == 1
    entry = next(s for s in log["steps"] if s["step"] == "isolation_forest")
    assert entry["examples"][0]["before"].startswith("salary=9999999")
    stats = log["summary"]["outliers"]
    assert stats["isolation_forest_rows"] == 1 and stats["rows_found_by_both"] == 1
    assert stats["flagged"] == stats["iqr_cells"] >= 1


def test_cap_clips_to_the_fences():
    cleaned, log = clean(staff(), outlier_action="cap")
    entry = next(s for s in log["steps"] if s["step"] == "outlier_cap")
    hi = float(entry["rule"].split("capped to [")[1].split(", ")[1].rstrip("]"))
    assert cleaned.loc[7, "salary"] == hi < 150_000
    assert "salary_outlier" not in cleaned.columns
    assert entry["examples"][0]["before"] == 9_999_999


def test_nan_then_imputed():
    cleaned, log = clean(staff(), outlier_action="nan", anomaly_share=0)
    steps = [(s["step"], s["column"]) for s in log["steps"]]
    assert ("outlier_to_nan", "salary") in steps and ("impute", "salary") in steps
    assert 60_000 < cleaned.loc[7, "salary"] < 100_000          # the median, not 9,999,999
    assert "is_anomaly" not in cleaned.columns                  # anomaly_share 0 = off


def test_action_per_column_overrides_global():
    df = staff()
    df.loc[3, "age"] = "95"                                     # an age outlier too
    cleaned, _ = clean(df, outlier_action="flag", outliers={"salary": "cap"})
    assert "age_outlier" in cleaned.columns and "salary_outlier" not in cleaned.columns


def test_special_code_column_is_not_touched():
    pdays = ["999"] * 40 + ["1", "2", "3", "2", "1", "27"]
    df = pd.DataFrame({"client": [f"C{i}" for i in range(46)], "pdays": pdays}, dtype=str)
    cleaned, log = clean(df, outlier_action="nan")
    assert cleaned["pdays"].dropna().tolist() == [1, 2, 3, 2, 1, 27]
    status = {s["action"]: s["status"] for s in log["suggestions"] if s["column"] == "pdays"}
    assert status.get("cap_outliers", "skipped") == "skipped"


def test_unknown_outlier_action_rejected():
    with pytest.raises(ValueError, match="unknown outlier action"):
        clean(staff(), outlier_action="delete")
