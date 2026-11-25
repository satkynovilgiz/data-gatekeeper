"""Sprint 9 sentinel fix: step 8b on new rows and the imputer's median / mode fallback."""
import numpy as np
import pandas as pd
import pytest

from src.cleaning.cleaner import DEFAULT_CONFIG, clean_dataset, learn_from_train
from src.cleaning.imputers import FoldImputer
from src.impact.impact import RowCleaner
from src.profiling.profiler import profile_dataset
from src.validation.contract import contract_ranges, parse_contract


@pytest.fixture
def train():
    rng = np.random.default_rng(3)
    n = 120
    return pd.DataFrame({
        "age": [str(v) for v in rng.integers(20, 61, n)],               # 20..60
        "rate": [f"{v:.2f}" for v in rng.uniform(0.5, 5.0, n)],         # 0.5..5.0
        "delta": [str(v) for v in rng.integers(-5, 6, n)],              # -5..5, has -1?
        "job": rng.choice(["admin.", "services", "student"], n),
    })


def test_learn_from_train_ranges_and_seen_sentinels():
    df = pd.DataFrame({"x": [1.0, 5.0, -1.0], "y": [10.0, 999.0, 20.0]})
    learned = learn_from_train(df, ["x", "y"], [999, -1])
    assert learned["ranges"] == {"x": [-1.0, 5.0], "y": [10.0, 999.0]}
    assert learned["seen_sentinels"] == {"x": [-1], "y": [999]}
    over = learn_from_train(df, ["x"], [999], ranges={"x": [0, 3]})
    assert over["ranges"]["x"] == [0.0, 3.0]


def test_out_of_range_and_unseen_sentinels_become_missing(train):
    cleaner = RowCleaner().fit(train)
    seen_minus_one = (train["delta"] == "-1").any()
    new = pd.DataFrame({"age": ["30", "45", "50", "35"],
                        "rate": ["999", "7.5", "2.0", "1.0"],       # 999 + 7.5 outside 0.5..5
                        "delta": ["-1", "2", "3", "999"],
                        "job": ["admin.", "student", "services", "admin."]})
    out = cleaner.transform(new)
    assert out["rate"].isna().tolist() == [True, True, False, False]
    assert out["age"].notna().all()                               # inside the range: kept
    assert pd.isna(out["delta"].iloc[3])                          # 999 outside -5..5
    # -1 is inside -5..5: kept only when the train rows really had it
    assert pd.isna(out["delta"].iloc[0]) == (not seen_minus_one)
    assert cleaner.out_of_range_["rate"] == 2
    assert cleaner.out_of_range_["delta"] == (1 if seen_minus_one else 2)


def test_unseen_sentinel_inside_the_range_becomes_missing():
    train = pd.DataFrame({"duration": [str(v) for v in range(0, 2000, 7)]})   # no 999
    assert not (train["duration"] == "999").any()
    cleaner = RowCleaner().fit(train)
    out = cleaner.transform(pd.DataFrame({"duration": ["999", "1000", "998"]}))
    assert out["duration"].isna().tolist() == [True, False, False]   # only the sentinel
    seen = RowCleaner().fit(pd.DataFrame({"duration": ["999"] + train["duration"].tolist()}))
    assert seen.transform(pd.DataFrame({"duration": ["999"]}))["duration"].notna().all()


def test_business_rule_still_applies_and_contract_ranges_override(train):
    cleaner = RowCleaner().fit(train)
    out = cleaner.transform(train.head(2).assign(age=["150", "61"]))
    assert out["age"].isna().all()           # 150: age rule 0-120; 61: outside train 20..60
    contract = parse_contract({"columns": [{"name": "age", "dtype": "integer",
                                            "min": 0, "max": 100}]})
    assert contract_ranges(contract) == {"age": [0.0, 100.0]}
    wide = RowCleaner(ranges=contract_ranges(contract)).fit(train)
    out = wide.transform(train.head(2).assign(age=["150", "61"]))
    assert out["age"].isna().tolist() == [True, False]               # 61 is fine now


def test_single_file_cleaning_is_unchanged(train):
    dirty = train.assign(rate=["999"] + train["rate"].tolist()[1:])
    _, log = clean_dataset(dirty, profile_dataset(dirty), {"numeric_strategy": "median",
                                                         "categorical_strategy": "mode"})
    assert not [s for s in log["steps"] if s["step"] == "out_of_range_to_missing"]
    assert log["summary"]["out_of_range_to_missing"] == 0
    assert DEFAULT_CONFIG["learned"] is None


def test_imputer_falls_back_for_rows_with_out_of_range_inputs():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 300)
    train = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 0.1, 300)})
    fi = FoldImputer("iterative", targets=["y"], features=["x", "y"]).fit(train)
    median_y = float(train["y"].median())
    new = pd.DataFrame({"x": [5.0, 1000.0], "y": [np.nan, np.nan]})
    out = fi.transform(new)
    assert out["y"].iloc[0] == pytest.approx(15.0, abs=0.5)        # normal row: the model
    assert out["y"].iloc[1] == pytest.approx(median_y)              # odd row: the median
    assert fi.fallbacks_ == 1
    fi.transform(new.head(1))
    assert fi.fallbacks_ == 0                                       # counted per call
