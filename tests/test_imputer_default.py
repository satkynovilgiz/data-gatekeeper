"""Sprint 11: the imputer settings; skip_complete is an option (deterministic, with a caveat)."""
import numpy as np
import pandas as pd
import pytest

from src.cleaning.imputers import ITERATIVE, FoldImputer, impute_columns, make_iterative


@pytest.fixture
def table():
    rng = np.random.default_rng(3)
    a = rng.normal(50, 10, 400)
    df = pd.DataFrame({"a": a, "b": a * 0.5 + rng.normal(0, 1, 400),
                       "c": rng.choice(["x", "y", "z"], 400)})
    df.loc[rng.choice(400, 40, replace=False), "b"] = np.nan
    df.loc[rng.choice(400, 30, replace=False), "c"] = np.nan
    return df


@pytest.fixture
def skip_complete(monkeypatch):
    monkeypatch.setitem(ITERATIVE, "skip_complete", True)


def test_default_settings_and_seed():
    # skip_complete stays off by default: the robustness study showed its cost on new rows
    assert ITERATIVE == {"max_iter": 10, "skip_complete": False, "n_nearest_features": None}
    imp = make_iterative()
    assert not imp.skip_complete and imp.random_state == 42 and imp.max_iter == 10
    assert make_iterative({"skip_complete": True}).skip_complete


def test_skip_complete_option_is_deterministic(table, skip_complete):
    one = impute_columns(table, ["b", "c"], ["a", "b", "c"], "iterative")
    two = impute_columns(table, ["b", "c"], ["a", "b", "c"], "iterative")
    assert one.equals(two)
    f1 = FoldImputer("iterative", ["a", "b", "c"], ["a", "b", "c"]).fit(table).transform(table)
    f2 = FoldImputer("iterative", ["a", "b", "c"], ["a", "b", "c"]).fit(table).transform(table)
    assert f1.equals(f2)


def test_caveat_column_complete_in_training_gets_the_mean_on_new_rows(table, skip_complete):
    # "a" has no gaps in training, so skip_complete builds no model for it: a gap in a new row
    # gets the training mean (the imputer's initial fill), not a prediction from b
    fi = FoldImputer("iterative", ["a", "b", "c"], ["a", "b", "c"]).fit(table)
    new = pd.DataFrame({"a": [np.nan], "b": [40.0], "c": ["x"]})    # b = 40 suggests a ~ 80
    filled = fi.transform(new)["a"].iloc[0]
    assert filled == pytest.approx(table["a"].mean(), abs=1e-6)


def test_default_predicts_that_column_from_the_others(table):
    # the default (skip_complete off) models "a" too, so the new row's gap follows b = 40
    fi = FoldImputer("iterative", ["a", "b", "c"], ["a", "b", "c"]).fit(table)
    filled = fi.transform(pd.DataFrame({"a": [np.nan], "b": [40.0], "c": ["x"]}))["a"].iloc[0]
    assert filled > table["a"].mean() + 15
