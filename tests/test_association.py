"""Sprint 2 — Cramér's V between categorical columns."""
import pandas as pd

from src.profiling.profiler import cramers_v, profile_dataset


def test_identical_columns_have_v_of_1():
    s = pd.Series(["a", "b", "c", "a", "b", "c"])
    assert cramers_v(s, s) == 1.0


def test_independent_columns_have_v_of_0():
    # every colour appears exactly once with every size -> no link at all
    colour = pd.Series(["red", "red", "blue", "blue"])
    size = pd.Series(["S", "L", "S", "L"])
    assert cramers_v(colour, size) == 0.0


def test_single_category_returns_none():
    assert cramers_v(pd.Series(["x", "x"]), pd.Series(["a", "b"])) is None


def test_report_lists_strong_pairs():
    df = pd.DataFrame({
        "country": ["UK", "UK", "FR", "FR"] * 5,
        "currency": ["GBP", "GBP", "EUR", "EUR"] * 5,   # fully decided by country
        "plan": ["basic", "pro", "basic", "pro"] * 5,    # unrelated to country
    })
    assoc = profile_dataset(df)["categorical_association"]
    assert assoc["method"] == "cramers_v"
    assert assoc["matrix"]["country"]["currency"] == 1.0
    assert assoc["matrix"]["country"]["plan"] == 0.0
    assert {(p["a"], p["b"]) for p in assoc["strong_pairs"]} == {("country", "currency")}
