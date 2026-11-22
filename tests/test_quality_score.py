"""Sprint 3 — data quality score (src/quality/score.py)."""
import pandas as pd
import pytest

from src.profiling.profiler import profile_dataset
from src.quality import quality_delta, quality_score


def test_perfect_data_scores_100():
    df = pd.DataFrame({"a": ["x", "y", "z", "w"], "b": ["1", "2", "3", "4"]})
    q = profile_dataset(df)["quality_score"]
    assert q["overall"] == 100.0
    assert set(q["dimensions"].values()) == {100.0}
    assert sum(q["weights"].values()) == pytest.approx(1.0)


def test_completeness_counts_disguised_missing():
    # 8 cells: 1 empty + 1 "unknown" -> 25% true missing -> completeness 75
    df = pd.DataFrame({"a": ["x", "unknown", None, "y"], "b": ["1", "2", "3", "4"]})
    assert profile_dataset(df)["quality_score"]["dimensions"]["completeness"] == 75.0


def test_uniqueness_counts_duplicates():
    df = pd.DataFrame({"a": ["x", "x", "y", "z"]})       # 1 duplicate of 4 rows
    assert profile_dataset(df)["quality_score"]["dimensions"]["uniqueness"] == 75.0


def test_validity_and_consistency_are_per_column():
    df = pd.DataFrame({
        "age": ["25"] * 18 + ["thirty", "40"],                 # mixed types -> invalid
        "city": ["London"] * 10 + ["london "] * 10,            # case variants -> inconsistent
        "ok": [str(i) for i in range(20)],
        "ok2": [f"v{i}" for i in range(20)],
    })
    dims = profile_dataset(df)["quality_score"]["dimensions"]
    assert dims["validity"] == 75.0        # 3 of 4 columns are fine
    assert dims["consistency"] == 75.0


def test_overall_is_weighted_average():
    report = {"dataset": {"rows": 10, "columns": 2, "duplicate_rows": 2,
                          "true_missing_cells_pct": 10.0},
              "issues": [{"column": "a", "rule": "mixed_types"}]}
    q = quality_score(report)
    assert q["dimensions"] == {"completeness": 90.0, "uniqueness": 80.0,
                               "validity": 50.0, "consistency": 100.0}
    assert q["overall"] == 80.0            # (90 + 80 + 50 + 100) / 4


def test_quality_delta_for_module2():
    before = {"overall": 80.0, "dimensions": {"completeness": 90.0, "uniqueness": 80.0}}
    after = {"overall": 95.5, "dimensions": {"completeness": 100.0, "uniqueness": 100.0}}
    assert quality_delta(before, after) == {"overall": 15.5, "completeness": 10.0,
                                            "uniqueness": 20.0}


def test_flag_columns_do_not_inflate_completeness():
    # Sprint 6 fix: a 0/1 flag column is never empty. Counting it raised completeness
    # completeness from 75% (1 of 4 cells in a) to 87.5% (1 of 8) without fixing anything.
    df = pd.DataFrame({"a": ["x", None, "y", "z"], "a_outlier": ["0", "0", "1", "0"]})
    report = profile_dataset(df)
    assert quality_score(report)["dimensions"]["completeness"] == 87.5      # old way
    q = quality_score(report, flag_columns=["a_outlier"])
    assert q["dimensions"]["completeness"] == 75.0                          # real data only
    assert q["flag_columns"] == {"columns": ["a_outlier"], "completeness": 100.0,
                                 "note": "added by cleaning; not part of overall"}


def test_flag_columns_left_out_of_validity():
    df = pd.DataFrame({"a": ["1", "2", "3", "4"] * 5,
                       "bad_flag": ["1"] * 18 + ["x", "y"]})                # mixed types
    report = profile_dataset(df)
    assert quality_score(report)["dimensions"]["validity"] == 50.0
    assert quality_score(report, flag_columns=["bad_flag"])["dimensions"]["validity"] == 100.0


def test_unknown_flag_names_are_ignored():
    df = pd.DataFrame({"a": ["x", "y"]})
    q = quality_score(profile_dataset(df), flag_columns=["not_there"])
    assert q["overall"] == 100.0 and "flag_columns" not in q
