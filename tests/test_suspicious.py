"""Sprint 2 — suspicious-column rules."""
import pandas as pd

from src.profiling.profiler import profile_dataset


def _rules(report):
    return {(i["column"], i["rule"]) for i in report["issues"]}


def test_id_stored_as_number():
    df = pd.DataFrame({"customer_id": [str(1000 + i) for i in range(50)],
                       "amount": [str(i * 2.5) for i in range(50)]})
    r = profile_dataset(df)
    assert ("customer_id", "id_stored_as_number") in _rules(r)
    assert ("amount", "id_stored_as_number") not in _rules(r)


def test_text_ids_are_fine():
    df = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(50)]})
    assert ("customer_id", "id_stored_as_number") not in _rules(profile_dataset(df))


def test_dominant_value():
    # 95 of 100 rows have the same value (and it is NOT a missing code like 999,
    # which is reported by disguised_missing instead — see test_disguised_missing.py)
    df = pd.DataFrame({"campaign": ["1"] * 95 + ["3", "4", "5", "6", "7"],
                       "city": ["A", "B"] * 50})
    r = profile_dataset(df)
    col = next(c for c in r["columns"] if c["name"] == "campaign")
    assert col["top_value"] == {"value": "1", "pct": 95.0}
    assert ("campaign", "dominant_value") in _rules(r)
    assert ("city", "dominant_value") not in _rules(r)


def test_constant_column_is_not_also_dominant():
    # a column with ONE value is reported as constant_column only (no duplicate issue)
    df = pd.DataFrame({"country": ["PT"] * 10})
    rules = _rules(profile_dataset(df))
    assert ("country", "constant_column") in rules
    assert ("country", "dominant_value") not in rules


def test_placeholder_values():
    df = pd.DataFrame({"education": ["uni", "school", "unknown", "Unknown ", "?", "uni"]})
    r = profile_dataset(df)
    col = r["columns"][0]
    assert col["disguised_missing"]["values"] == {"unknown": 1, "Unknown ": 1, "?": 1}
    assert ("education", "disguised_missing") in _rules(r)


def test_suspicious_columns_summary():
    df = pd.DataFrame({"user_id": [str(i) for i in range(20)],
                       "flag": ["no"] * 19 + ["yes"],
                       "ok": ["a", "b", "c", "d"] * 5})
    assert profile_dataset(df)["summary"]["suspicious_columns"] == ["flag", "user_id"]
