"""Fuzzy duplicate detection (src/cleaning/fuzzy.py + fuzzy_duplicates step)."""
import pandas as pd

from src.cleaning.cleaner import clean_dataset
from src.cleaning.fuzzy import find_fuzzy_duplicates, name_key
from src.profiling.profiler import profile_dataset

COLUMNS = ["employee_id", "full_name", "department", "age", "phone", "hire_date"]
ROWS = [
    ["E1", "John Smith", "Sales", "34", "+1-408-555-0101", "2020-01-15"],
    ["E2", "Maria Garcia", "HR", "41", "+1-408-555-0102", "2019-03-02"],
    ["E3", "Jon Smith", "Sales", "34", "+1-408-555-0101", "2020-01-15"],     # typo of E1
    ["E4", "Maria Garcia", "IT", "29", "+1-408-555-0199", "2022-07-30"],     # another Maria
    ["E5", "Garcia, Maria", "HR", "41", "+1-408-555-0102", "2019-03-02"],    # E2, reordered
    ["E6", "Chen Lee", "IT", "52", "+1-408-555-0106", "2016-09-09"],
    ["E7", "Aisha Khan", "Sales", "45", "+1-408-555-0107", "2017-05-20"],
    ["E8", "Omar Patel", "HR", "38", "+1-408-555-0108", "2018-11-05"],
]


def staff() -> pd.DataFrame:
    return pd.DataFrame(ROWS, columns=COLUMNS, dtype=str)


def test_name_key_normalises():
    assert name_key("  Garcia,  MARIA ") == "garcia maria"


def test_finds_typos_and_reordered_names_but_not_namesakes():
    df = staff()
    compare = ["department", "age", "phone", "hire_date"]
    pairs = find_fuzzy_duplicates(df, ["full_name"], compare, threshold=90)
    found = {(p["row_a"], p["row_b"]) for p in pairs}
    assert (0, 2) in found                                     # John / Jon Smith
    assert (1, 4) in found                                     # Maria Garcia / Garcia, Maria
    assert (1, 3) not in found and (3, 4) not in found         # same name, different person
    jon = next(p for p in pairs if p["row_b"] == 2)
    assert 90 <= jon["name_similarity"] < 100
    assert jon["agreement"] == 1.0 and jon["differs"] == []


def test_higher_threshold_finds_fewer_pairs():
    compare = ["department", "age", "phone", "hire_date"]
    pairs = find_fuzzy_duplicates(staff(), ["full_name"], compare, threshold=99)
    assert {(p["row_a"], p["row_b"]) for p in pairs} == {(1, 4)}   # only the reordered name


def test_pipeline_flags_both_rows_and_keeps_them():
    cleaned, log = clean_dataset(staff(), profile_dataset(staff(), "hr.csv"))
    assert len(cleaned) == 8                                   # nothing removed
    assert cleaned["possible_duplicate"].tolist() == [1, 1, 1, 0, 1, 0, 0, 0]
    s = log["summary"]["fuzzy_duplicates"]
    assert s == {"pairs_found": 2, "rows_flagged": 4, "rows_merged": 0}
    assert log["fuzzy_duplicates"]["key_columns"] == ["full_name"]
    assert "employee_id" not in log["fuzzy_duplicates"]["compared_columns"]  # IDs differ anyway


def test_pipeline_merges_above_the_merge_threshold():
    cleaned, log = clean_dataset(staff(), profile_dataset(staff(), "hr.csv"),
                                 {"fuzzy_merge_threshold": 99})
    assert "E5" not in set(cleaned["employee_id"])            # reordered name: merged
    assert cleaned.loc[[0, 2], "possible_duplicate"].tolist() == [1, 1]   # Jon: only flagged
    merge = next(s for s in log["steps"] if s["step"] == "fuzzy_merge")
    assert merge["rows_changed"] == 1
    assert merge["examples"][0]["similar_to_row"] == 1


def test_skipped_without_a_name_column():
    df = pd.DataFrame({"age": ["30", "31", "32"], "job": ["a", "b", "c"]}, dtype=str)
    cleaned, log = clean_dataset(df, profile_dataset(df, "bank.csv"))
    assert "possible_duplicate" not in cleaned.columns
    assert log["summary"]["fuzzy_duplicates"]["skipped"].startswith("no name column")
