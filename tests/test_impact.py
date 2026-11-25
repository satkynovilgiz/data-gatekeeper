"""Impact (Sprint 8): deterministic, no leakage across folds, honest paired statistics."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.cleaning.imputers import FoldImputer, impute_columns
from src.impact import impact as imp
from src.impact.impact import paired, positive_label, run_impact, score
from src.impact.report import to_markdown, write_impact_report
from src.profiling.profiler import read_csv_raw

SAMPLE = Path(__file__).resolve().parents[1] / "data/raw/bank_marketing_sample.csv"


@pytest.fixture(scope="module")
def bank():
    """400 rows of the committed bank sample, keeping both classes."""
    df = read_csv_raw(SAMPLE)
    yes, no = df[df["y"] == "yes"], df[df["y"] == "no"]
    return pd.concat([yes.head(60), no.head(340)]).sort_index().reset_index(drop=True)


# ------------------------------------------------------------ statistics
def test_paired_difference_by_hand():
    # diffs 1..5: mean 3, sd 1.5811, t(0.975, 4) = 2.7764 -> half width 1.9632
    out = paired([2, 4, 6, 8, 10], [1, 2, 3, 4, 5])
    assert out["mean_diff"] == pytest.approx(3.0)
    assert out["ci95"] == pytest.approx([3 - 1.9632, 3 + 1.9632], abs=1e-4)
    assert out["verdict"] == "better"
    assert out["wilcoxon_p"] == pytest.approx(0.0625)            # the smallest possible, n=5


def test_paired_verdicts_respect_direction_and_uncertainty():
    up = paired([0.81, 0.82, 0.83, 0.84, 0.85], [0.80, 0.81, 0.82, 0.83, 0.84])
    assert up["verdict"] == "better" and up["ci95"][0] > 0
    # Brier: lower is better, so a higher Brier for the first variant is worse
    assert paired([0.11] * 5, [0.10] * 5, lower_is_better=True)["verdict"] == "worse"
    noisy = paired([0.80, 0.85, 0.79, 0.86, 0.80], [0.82, 0.81, 0.83, 0.82, 0.81])
    assert noisy["verdict"] == "no clear difference"
    assert noisy["ci95"][0] < 0 < noisy["ci95"][1]
    assert paired([1, 2], [0, 1])["wilcoxon_p"] is None           # < 5 folds: no p-value


def test_score_and_positive_label():
    s = score(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert s["roc_auc"] == pytest.approx(0.75) and 0 <= s["brier"] <= 1
    assert positive_label(pd.Series(["no", "yes", "no"])) == "yes"
    assert positive_label(pd.Series(["a", "a", "b"])) == "b"      # the rarer class
    with pytest.raises(ValueError):
        positive_label(pd.Series(["a", "b"]), positive="c")


# ---------------------------------------------------------- determinism
def test_impact_is_deterministic(bank):
    kw = dict(folds=2, robustness=False, models=("logreg",), dataset="t")
    a = run_impact(bank, "y", **kw)
    b = run_impact(bank, "y", **kw)
    assert a["results"] == b["results"] and a["comparisons"] == b["comparisons"]
    assert a["positive_label"] == "yes" and a["rows"] == 400
    assert set(a["results"]) == {"A", "B", "C"}


# -------------------------------------------------------------- leakage
def test_every_fitted_part_sees_only_train_rows(bank, monkeypatch):
    """Spy on every fit(): the rows it learns from must be inside that fold's train rows."""
    seen = []
    for cls in (imp.RawPrep, imp.RowCleaner, imp.PlanTransform, FoldImputer):
        original = cls.fit

        def spy(self, X, y=None, _original=original, _name=cls.__name__):
            seen.append((_name, set(X.index)))
            return _original(self, X, y)
        monkeypatch.setattr(cls, "fit", spy)

    run_impact(bank, "y", folds=3, robustness=False, models=("logreg",))
    y = (bank["y"] == "yes").astype(int).to_numpy()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42)
                 .split(bank.drop(columns="y"), y))
    names = [n for n, _ in seen]
    assert {"RawPrep", "RowCleaner", "PlanTransform", "FoldImputer"} <= set(names)
    # fits happen fold by fold, in order; map each fit to its fold by its rows
    for name, rows in seen:
        hits = [k for k, (tr, te) in enumerate(folds) if rows <= set(tr)]
        assert hits, f"{name} learned from rows outside every train fold"
        k = hits[0]
        assert not rows & set(folds[k][1]), f"{name} saw test rows of fold {k}"


def test_fitted_statistics_come_from_train_rows_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "c": ["a", "a", "b", None]})
    test = pd.DataFrame({"x": [100.0, np.nan], "c": ["zzz", None]})
    fi = FoldImputer("iterative", targets=["x", "c"], features=["x", "c"]).fit(train)
    assert fi.encoding_.scale["x"][0] == pytest.approx(2.0)       # train mean, not with 100
    assert fi.encoding_.categories["c"] == ["a", "b"]             # "zzz" never learned
    out = fi.transform(test)
    assert out["x"].notna().all() and 1.0 <= out["x"].iloc[1] <= 3.0   # clipped to train
    assert out["c"].iloc[1] in ("a", "b")
    raw = imp.RawPrep().fit(pd.DataFrame({"n": ["1", "2", "3"], "k": ["u", "v", "u"]}))
    scaler = raw.ct_.named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(2.0) and scaler.n_samples_seen_ == 3


def test_fold_imputer_equals_impute_columns_on_one_table():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(50, 10, 200).round(),
                       "c": rng.choice(["x", "y", "z"], 200)})
    df["b"] = df["a"] * 0.1 + rng.normal(0, 0.3, 200)
    df.loc[rng.choice(200, 30, replace=False), "b"] = np.nan
    df.loc[rng.choice(200, 20, replace=False), "c"] = np.nan
    for method in ("median", "knn", "iterative"):
        ref = impute_columns(df, ["b", "c"], ["a", "b", "c"], method)
        got = FoldImputer(method, ["b", "c"], ["a", "b", "c"]).fit_transform(df)
        assert np.allclose(ref["b"].astype(float), got["b"].astype(float)), method
        assert (ref["c"] == got["c"]).all(), method


def test_duplicates_removed_from_train_only(bank):
    doubled = pd.concat([bank, bank.head(20)], ignore_index=True)
    report = run_impact(doubled, "y", folds=2, robustness=False, models=("logreg",))
    removed = sum(f["train_duplicates_removed"] for f in report["fold_info"])
    assert removed > 0
    assert sum(f["test_rows"] for f in report["fold_info"]) == len(doubled)   # test intact


# ---------------------------------------------------- robustness + output
def test_robustness_and_report_files(bank, tmp_path):
    report = run_impact(bank, "y", folds=2, levels=(0.0, 0.1), models=("logreg",),
                        dataset="mini")
    rows = report["robustness"]["rows"]
    kinds = {r["kind"] for r in rows}
    assert kinds == {"all", "missing", "sentinels", "numbers_as_text", "duplicates"}
    zero = [r for r in rows if r["level"] == 0 and r["kind"] == "all"]
    assert {r["variant"] for r in zero} == {"A", "B"}
    assert all(abs(r["roc_auc_drop"]) < 1e-12 for r in zero)
    files = write_impact_report(report, tmp_path, "mini")
    assert {Path(p).name for p in files.values()} == {
        "mini_impact.json", "mini_impact.md", "mini_robustness.png"}
    md = to_markdown(report)
    assert "± " in md and "95% CI" in md and "B-A" in md


def test_bad_target():
    with pytest.raises(ValueError, match="not found"):
        run_impact(pd.DataFrame({"a": ["1"]}), "y")
