"""Sprint 3 — cleaning_suggestions: the hand-off from Module 1 to Module 2."""
import json
from pathlib import Path

import pandas as pd

from src.profiling.profiler import profile_dataset
from src.profiling.suggestions import ACTIONS

ROOT = Path(__file__).resolve().parents[1]


def _actions(report):
    return {(s["column"], s["action"]) for s in report["cleaning_suggestions"]}


def test_one_suggestion_per_issue_most_severe_first():
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": ["1", "1", "unknown"]})
    r = profile_dataset(df)
    s = r["cleaning_suggestions"]
    assert len(s) == len(r["issues"])
    assert s[0]["severity"] == "high"


def test_typical_actions():
    df = pd.DataFrame({
        "age": ["25", "31", None, "40", "28", "33", "29", "35", "41", "38"] * 2,
        "city": ["London", "london ", "Paris", "Paris", "Rome"] * 4,
        "joined": ["2024-01-05", "05/01/2024"] * 10,
        "email": ["a@x.com"] * 20,
        "job": ["admin", "unknown", "tech", "tech", "admin"] * 4,
    })
    acts = _actions(profile_dataset(df))
    assert ("age", "impute_median") in acts                        # numeric -> median
    assert ("city", "standardise_case") in acts
    assert ("joined", "parse_dates_iso") in acts
    assert ("email", "mask_pii") in acts
    assert ("job", "replace_with_nan_then_impute_mode") in acts    # 'unknown' in a category
    assert ("<dataset>", "drop_duplicates") in acts


def test_mostly_disguised_column_gets_a_flag_not_imputation():
    df = pd.DataFrame({"pdays": ["999"] * 9 + ["3"], "x": [str(i) for i in range(10)]})
    acts = _actions(profile_dataset(df))
    assert ("pdays", "replace_with_nan_and_add_flag") in acts


def test_schema_lists_the_same_actions_as_the_code():
    schema = json.loads((ROOT / "docs" / "profiling_report.schema.json").read_text())
    enum = schema["properties"]["cleaning_suggestions"]["items"]["properties"]["action"]["enum"]
    assert enum == ACTIONS
