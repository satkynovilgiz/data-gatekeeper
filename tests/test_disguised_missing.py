"""Sprint 3 — disguised missing values (placeholder text + numeric sentinels like 999)."""
import pandas as pd

from src.profiling.profiler import profile_dataset


def _col(report, name):
    return next(c for c in report["columns"] if c["name"] == name)


def _issue(report, column, rule):
    return next((i for i in report["issues"] if i["column"] == column and i["rule"] == rule), None)


def test_text_placeholders_count_as_missing():
    # 2 real NaN + 3 placeholders ("n/a", "-", "  ") out of 10 rows
    df = pd.DataFrame({"job": ["admin", "n/a", None, "-", "admin", "  ",
                               "tech", None, "tech", "admin"]})
    c = _col(profile_dataset(df), "job")
    assert c["missing_pct"] == 20.0
    assert c["disguised_missing"]["values"] == {"n/a": 1, "-": 1, "  ": 1}
    assert c["disguised_missing"]["count"] == 3
    assert c["true_missing_pct"] == 50.0          # 20% empty + 30% disguised


def test_pdays_999_is_a_sentinel():
    # like the bank dataset: 999 = "never contacted", real values are small
    df = pd.DataFrame({"pdays": ["999"] * 8 + ["3", "6"]})
    r = profile_dataset(df)
    c = _col(r, "pdays")
    assert c["disguised_missing"]["values"] == {"999": 8}
    assert c["true_missing_pct"] == 80.0
    # stats are computed WITHOUT the 999s
    assert c["numeric_stats"]["max"] == 6
    # >= 50% true missing -> high severity; dominant_value is not repeated
    assert _issue(r, "pdays", "disguised_missing")["severity"] == "high"
    assert _issue(r, "pdays", "dominant_value") is None


def test_minus_one_is_a_sentinel_only_when_other_values_are_positive():
    df = pd.DataFrame({
        "days_since": ["-1", "5", "10", "-1", "7"],      # -1 = "no value"
        "change": ["-1", "-2.5", "0.5", "1", "-0.3"],     # -1 is a real change here
    })
    r = profile_dataset(df)
    assert _col(r, "days_since")["disguised_missing"]["values"] == {"-1": 2}
    assert _col(r, "change")["disguised_missing"]["count"] == 0


def test_real_999_in_a_wide_range_is_not_flagged():
    # salaries: 999 is a normal value between 500 and 5000
    df = pd.DataFrame({"salary": ["500", "999", "1500", "3000", "5000"]})
    assert _col(profile_dataset(df), "salary")["disguised_missing"]["count"] == 0


def test_dataset_true_missing_pct():
    df = pd.DataFrame({"a": ["x", "unknown", None, "y"], "b": ["1", "2", "3", "4"]})
    d = profile_dataset(df)["dataset"]
    assert d["missing_cells_pct"] == 12.5             # 1 of 8 cells
    assert d["disguised_missing_cells"] == 1
    assert d["true_missing_cells_pct"] == 25.0        # 2 of 8 cells


def test_outliers_not_flagged_when_iqr_is_zero():
    # 'previous' in the bank data: mostly 0, so Q1 = Q3 = 0 and IQR = 0.
    # Before Sprint 3 every non-zero value was called an outlier.
    df = pd.DataFrame({"previous": ["0"] * 80 + ["1"] * 15 + ["2"] * 5})
    r = profile_dataset(df)
    out = _col(r, "previous")["numeric_stats"]["outliers_iqr"]
    assert out == {"count": 0, "pct": 0.0, "lower_fence": None, "upper_fence": None}
    assert _issue(r, "previous", "outliers") is None
