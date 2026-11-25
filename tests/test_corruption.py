"""Corruption injectors (Sprint 8): exact fractions, fixed seeds, input never changed."""
import numpy as np
import pandas as pd
import pytest

from src.impact.corruption import (
    NUMERIC_SENTINEL,
    TEXT_SENTINEL,
    corrupt,
    decimal_comma,
    inject_duplicates,
    inject_missing,
    inject_numbers_as_text,
    inject_sentinels,
)

ROWS = 200
NUMERIC = ["age", "rate"]
COLUMNS = ["age", "rate", "job"]


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    return pd.DataFrame({"age": [str(v) for v in rng.integers(18, 90, ROWS)],
                         "rate": [f"{v:.3f}" for v in rng.uniform(0, 5, ROWS)],
                         "job": rng.choice(["admin.", "services", "student"], ROWS),
                         "label": rng.choice(["no", "yes"], ROWS)})


def _changed(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> int:
    return int((a[cols].fillna("<NA>") != b[cols].fillna("<NA>")).to_numpy().sum())


@pytest.mark.parametrize("frac", [0.0, 0.05, 0.10, 0.20])
def test_missing_changes_exactly_the_fraction(df, frac):
    out, n = inject_missing(df, frac, seed=42, columns=COLUMNS)
    expected = round(frac * ROWS * len(COLUMNS))
    assert n == expected == int(out[COLUMNS].isna().sum().sum())
    assert _changed(df, out, COLUMNS) == expected
    assert out["label"].equals(df["label"])                       # the label is never touched


@pytest.mark.parametrize("frac", [0.05, 0.10, 0.20])
def test_sentinels_change_exactly_the_fraction(df, frac):
    out, n = inject_sentinels(df, frac, seed=42, numeric_columns=NUMERIC, columns=COLUMNS)
    assert n == round(frac * ROWS * len(COLUMNS)) == _changed(df, out, COLUMNS)
    nums = int((out[NUMERIC] == NUMERIC_SENTINEL).to_numpy().sum())
    text = int((out["job"] == TEXT_SENTINEL).sum())
    assert nums + text == n and text > 0 and nums > 0            # 999 in numbers, ? in text


@pytest.mark.parametrize("frac", [0.05, 0.10, 0.20])
def test_numbers_as_text_change_exactly_the_fraction(df, frac):
    out, n = inject_numbers_as_text(df, frac, seed=42, numeric_columns=NUMERIC)
    assert n == round(frac * ROWS * len(NUMERIC)) == _changed(df, out, NUMERIC)
    unreadable = pd.to_numeric(out[NUMERIC].stack(), errors="coerce").isna().sum()
    assert unreadable == n                                        # every changed cell broke
    assert out["job"].equals(df["job"])


@pytest.mark.parametrize("frac", [0.0, 0.05, 0.20])
def test_duplicates_add_exactly_the_fraction(df, frac):
    out, n = inject_duplicates(df, frac, seed=42)
    assert n == round(frac * ROWS) and len(out) == ROWS + n
    assert out.iloc[:ROWS].equals(df)
    assert out.duplicated().sum() >= n                            # each added row is a copy


def test_decimal_comma():
    assert decimal_comma("4.857") == "4,857"
    assert decimal_comma("42") == "42,0"
    assert decimal_comma("-1.5") == "-1,5"


def test_same_seed_same_result_other_seed_differs(df):
    a, _ = inject_missing(df, 0.1, seed=7, columns=COLUMNS)
    b, _ = inject_missing(df, 0.1, seed=7, columns=COLUMNS)
    c, _ = inject_missing(df, 0.1, seed=8, columns=COLUMNS)
    assert a.equals(b) and not a.equals(c)


def test_input_is_never_changed(df):
    before = df.copy()
    corrupt(df, 0.2, 42, NUMERIC, COLUMNS)
    assert df.equals(before)


def test_corrupt_chains_all_kinds_with_exact_counts(df):
    out, changed = corrupt(df, 0.10, 42, NUMERIC, COLUMNS)
    assert changed == {"missing": 60, "sentinels": 60, "numbers_as_text": 40,
                       "duplicates": 20}
    assert len(out) == ROWS + 20
    assert _changed(df, out.iloc[:ROWS], COLUMNS) == 60 + 60 + 40   # no cell hit twice


def test_bad_fractions(df):
    with pytest.raises(ValueError):
        inject_missing(df, 1.5, 0, COLUMNS)
    with pytest.raises(ValueError):
        inject_duplicates(df, -0.1, 0)
    with pytest.raises(ValueError, match="unknown corruption kind"):
        corrupt(df, 0.1, 0, NUMERIC, COLUMNS, kinds=("typos",))
    empty = df.assign(age=np.nan, rate=np.nan, job=np.nan)
    with pytest.raises(ValueError, match="only 0 can be changed"):
        inject_missing(empty, 0.1, 0, COLUMNS)
