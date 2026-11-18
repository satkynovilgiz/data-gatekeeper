"""Sprint 2 — distribution summaries (histograms + top categories)."""
import pandas as pd

from src.profiling.profiler import profile_dataset


def _col(report, name):
    return next(c for c in report["columns"] if c["name"] == name)


def test_numeric_histogram():
    df = pd.DataFrame({"amount": [str(v) for v in range(100)]})   # 0..99
    hist = _col(profile_dataset(df), "amount")["numeric_stats"]["histogram"]
    assert len(hist["counts"]) == 10
    assert len(hist["bin_edges"]) == 11               # one more edge than bins
    assert sum(hist["counts"]) == 100                 # every value is in a bin
    assert hist["counts"] == [10] * 10                # evenly spread data -> equal bins
    assert hist["bin_edges"][0] == 0 and hist["bin_edges"][-1] == 99


def test_histogram_ignores_text_values():
    # 9 numbers + 1 word: still a numeric column (90%), the word is skipped
    df = pd.DataFrame({"age": [str(a) for a in range(20, 65, 5)] + ["forty", None]})
    hist = _col(profile_dataset(df), "age")["numeric_stats"]["histogram"]
    assert sum(hist["counts"]) == 9


def test_top_categories_with_percentages():
    df = pd.DataFrame({"job": ["admin"] * 6 + ["tech"] * 3 + ["retired"] * 1})
    stats = _col(profile_dataset(df), "job")["categorical_stats"]
    assert stats["top_values"] == {"admin": 6, "tech": 3, "retired": 1}
    assert stats["top_values_pct"] == {"admin": 60.0, "tech": 30.0, "retired": 10.0}
    assert stats["other_values_count"] == 0


def test_other_values_count_for_long_tail():
    df = pd.DataFrame({"city": [f"c{i}" for i in range(15)] * 2})   # 15 categories
    stats = _col(profile_dataset(df), "city")["categorical_stats"]
    assert len(stats["top_values"]) == 10
    assert stats["other_values_count"] == 10   # 5 remaining categories x 2 rows
