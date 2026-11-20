"""One test per cleaning step (src/cleaning/cleaner.py), on tiny hand-made tables."""
import numpy as np
import pandas as pd
import pytest

from src.cleaning import cleaner
from src.cleaning.cleaner import DEFAULT_CONFIG, CleaningRun, clean_dataset
from src.cleaning.schema import infer_schema
from src.profiling.profiler import profile_dataset


def make_run(data: dict, **config) -> CleaningRun:
    """A CleaningRun for a small text table, like read_csv_raw would give it."""
    df = pd.DataFrame(data, dtype=str)
    rep = profile_dataset(df, "t.csv")
    return CleaningRun(df=df.copy(), schema=infer_schema(df, rep),
                       config={**DEFAULT_CONFIG, **config},
                       suggestions=rep["cleaning_suggestions"])


def steps_named(run: CleaningRun, step: str) -> list[dict]:
    return [s for s in run.steps if s["step"] == step]


def test_trim_whitespace():
    run = make_run({"city": [" London", "Paris ", "Rome", None]})
    cleaner.trim_whitespace(run)
    assert run.df["city"].tolist()[:3] == ["London", "Paris", "Rome"]
    assert pd.isna(run.df["city"].iloc[3])
    (entry,) = steps_named(run, "trim_whitespace")
    assert entry["cells_changed"] == 2
    assert entry["examples"][0] == {"row": 0, "before": " London", "after": "London"}


def test_disguised_missing_to_nan():
    run = make_run({"job": ["admin.", "unknown", "n/a", "?", "technician", "Unknown"]})
    cleaner.disguised_missing_to_nan(run)
    assert run.df["job"].isna().tolist() == [False, True, True, True, False, True]
    (entry,) = steps_named(run, "disguised_missing_to_nan")
    assert entry["cells_changed"] == 4


def test_pdays_999_becomes_previously_contacted_not_imputed():
    # pdays = days since last contact; 999 = "never contacted before" (UCI Bank Marketing)
    pdays = ["999"] * 20 + ["3", "6", "4"]
    df = pd.DataFrame({"age": [str(30 + i) for i in range(23)], "pdays": pdays}, dtype=str)
    cleaned, log = clean_dataset(df, profile_dataset(df, "bank.csv"))

    # new column right after pdays: no = never contacted, yes = contacted before
    cols = list(cleaned.columns)
    assert cols[cols.index("pdays") + 1] == "previously_contacted"
    assert cleaned["previously_contacted"].tolist() == ["no"] * 20 + ["yes"] * 3
    # pdays is NaN ONLY for the never-contacted rows, the real values are kept
    assert cleaned["pdays"].isna().sum() == 20
    assert cleaned["pdays"].dropna().tolist() == [3, 6, 4]
    # ... and it is NOT imputed (no median filled into the 999 rows)
    assert not any(s["step"] == "impute" and s["column"] in ("pdays", "previously_contacted")
                   for s in log["steps"])
    assert log["summary"]["columns_added"] == ["previously_contacted"]
    assert log["summary"]["special_codes_to_flag"] == 20


def test_mostly_missing_column_gets_flag_instead_of_imputation():
    run = make_run({"bonus": ["unknown"] * 8 + ["100", "200"]})
    run.suggestions.append({"column": "bonus", "action": "replace_with_nan_and_add_flag"})
    cleaner.disguised_missing_to_nan(run)
    assert run.df["bonus_missing"].tolist() == [1] * 8 + [0, 0]
    assert "bonus" in run.no_impute


def test_normalise_categories():
    run = make_run({"dept": ["HR"] * 4 + ["hr", "Hr"] + ["Finance"] * 3 + ["Fin.", "Sales"]})
    cleaner.normalise_categories(run)
    assert set(run.df["dept"]) == {"HR", "Finance", "Sales"}
    (entry,) = steps_named(run, "normalise_categories")
    assert entry["cells_changed"] == 3                     # hr, Hr, Fin.


def test_convert_to_number_flags_text():
    years = [str(i) for i in range(30)] + ["five"]
    run = make_run({"years": years})
    cleaner.convert_to_number(run)
    assert pd.api.types.is_numeric_dtype(run.df["years"])
    assert np.isnan(run.df["years"].iloc[30])
    (entry,) = steps_named(run, "convert_to_number")
    assert entry["cells_changed"] == 1
    assert entry["examples"] == [{"row": 30, "before": "five", "after": None}]


def test_parse_dates_to_iso_and_flag_unreadable():
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)] + ["20/09/2023", "05/06/2020", "soon"]
    run = make_run({"hired": dates})
    cleaner.parse_dates(run)
    out = run.df["hired"].tolist()
    assert out[0] == "2024-01-01"                           # ISO stays ISO
    assert out[20] == "2023-09-20"                          # 20 > 12, so day first
    assert out[21] == "2020-06-05"                          # same order for the whole column
    assert pd.isna(out[22])                                 # "soon" cannot be read
    reformat, bad = steps_named(run, "parse_dates")
    assert reformat["cells_changed"] == 2 and bad["cells_changed"] == 1


def test_dayfirst_detected_from_values():
    assert cleaner._dayfirst(pd.Series(["09/20/2023", "01/02/2023"]), default=True) is False
    assert cleaner._dayfirst(pd.Series(["20/09/2023"]), default=False) is True
    assert cleaner._dayfirst(pd.Series(["01/02/2023"]), default=True) is True   # no clue


def test_invalid_format_to_nan():
    emails = [f"user{i}@company.com" for i in range(20)] + ["not-an-email"]
    run = make_run({"email": emails})
    cleaner.invalid_format_to_nan(run)
    assert run.df["email"].isna().sum() == 1
    assert run.df["email"].iloc[0] == "user0@company.com"


def test_impossible_values_set_to_nan_and_rows_kept():
    ages = [str(20 + i) for i in range(30)] + ["230", "-5"]
    run = make_run({"age": ages})
    cleaner.convert_to_number(run)
    cleaner.impossible_value_to_nan(run)
    assert len(run.df) == 32                                # no row deleted
    assert run.df["age"].isna().sum() == 2
    (entry,) = steps_named(run, "impossible_value_to_nan")
    assert "[0, 120]" in entry["rule"] and "business_rule:age" in entry["rule"]


def test_valid_range_can_be_set_in_config():
    run = make_run({"age": [str(10 + i) for i in range(30)]}, valid_ranges={"age": [18, 100]})
    cleaner.convert_to_number(run)
    cleaner.impossible_value_to_nan(run)
    assert run.df["age"].isna().sum() == 8                  # 10..17


def test_drop_duplicates_keeps_first_copy():
    run = make_run({"a": ["1", "2", "1", "3", "2"], "b": ["x", "y", "x", "z", "y"]})
    cleaner.drop_duplicates(run)
    assert run.df.index.tolist() == [0, 1, 3]
    (entry,) = steps_named(run, "drop_duplicates")
    assert entry["rows_changed"] == 2
    assert entry["examples"][0] == {"row": 2, "duplicate_of_row": 0}


def test_impute_median_for_numbers_and_mode_for_categories():
    df = pd.DataFrame({"age": ["20", "30", None, "40", "100"],
                       "city": ["Rome", "Rome", "Paris", None, "Rome"]}, dtype=str)
    cleaned, log = clean_dataset(df, profile_dataset(df, "t.csv"))
    assert cleaned["age"].tolist() == [20, 30, 35, 40, 100]           # median of 20,30,40,100
    assert cleaned["city"].tolist()[3] == "Rome"                       # mode
    rules = {s["column"]: s["rule"] for s in log["steps"] if s["step"] == "impute"}
    assert rules == {"age": "median = 35", "city": "mode = Rome"}


def test_impute_strategy_configurable_per_column():
    df = pd.DataFrame({"age": ["20", "30", None, "40", "100"],
                       "score": ["1.0", "2.0", None, "3.0", "10.0"]}, dtype=str)
    cleaned, _ = clean_dataset(df, profile_dataset(df, "t.csv"),
                               {"impute": {"age": "none", "score": "mean"}})
    assert pd.isna(cleaned["age"].iloc[2])                             # left empty
    assert cleaned["score"].iloc[2] == pytest.approx(4.0)              # mean of 1, 2, 3, 10


def test_unknown_strategy_is_rejected():
    df = pd.DataFrame({"a": ["1", None]}, dtype=str)
    with pytest.raises(ValueError, match="unknown imputation strategy"):
        clean_dataset(df, profile_dataset(df, "t.csv"), {"impute": {"a": "magic"}})


def test_dates_and_text_are_never_imputed():
    # small file: Module 1 calls email "categorical" (<= 20 values), it must still not be filled
    df = pd.DataFrame({"hired": [f"2024-01-{i + 1:02d}" if i % 3 else None for i in range(15)],
                       "email": [f"u{i}@b.com" if i % 3 != 1 else None for i in range(15)]},
                      dtype=str)
    cleaned, _ = clean_dataset(df, profile_dataset(df, "t.csv"))
    assert cleaned["hired"].isna().sum() == 5
    assert cleaned["email"].isna().sum() == 5


def test_input_dataframe_is_not_changed():
    df = pd.DataFrame({"city": [" Rome", "unknown", "Rome"]}, dtype=str)
    before = df.copy()
    clean_dataset(df, profile_dataset(df, "t.csv"))
    pd.testing.assert_frame_equal(df, before)
