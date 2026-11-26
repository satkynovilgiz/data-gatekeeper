"""Sprint 11: the faster iterative imputer default (skip_complete): deterministic + caveat."""
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


def test_default_is_skip_complete_and_seeded():
    assert ITERATIVE == {"max_iter": 10, "skip_complete": True, "n_nearest_features": None}
    imp = make_iterative()
    assert imp.skip_complete and imp.random_state == 42 and imp.max_iter == 10
    assert make_iterative({"max_iter": 3}).max_iter == 3


def test_new_default_is_deterministic(table):
    one = impute_columns(table, ["b", "c"], ["a", "b", "c"], "iterative")
    two = impute_columns(table, ["b", "c"], ["a", "b", "c"], "iterative")
    assert one.equals(two)
    f1 = FoldImputer("iterative", ["a", "b", "c"], ["a", "b", "c"]).fit(table).transform(table)
    f2 = FoldImputer("iterative", ["a", "b", "c"], ["a", "b", "c"]).fit(table).transform(table)
    assert f1.equals(f2)


def test_caveat_column_complete_in_training_gets_the_mean_on_new_rows(table):
    # "a" has no gaps in training, so skip_complete builds no model for it: a gap in a new row
    # gets the training mean (the imputer's initial fill), not a prediction from b
    fi = FoldImputer("iterative", ["a", "b", "c"], ["a", "b", "c"]).fit(table)
    new = pd.DataFrame({"a": [np.nan], "b": [40.0], "c": ["x"]})    # b = 40 suggests a ~ 80
    filled = fi.transform(new)["a"].iloc[0]
    assert filled == pytest.approx(table["a"].mean(), abs=1e-6)
