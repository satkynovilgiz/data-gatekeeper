"""Turn the column plan (plan.py) into one scikit-learn pipeline, fitted on TRAIN rows only.

Why "fit on train only"? A scaler learns numbers from the data (mean, std, min, max,
category lists). If it learns them from the test rows too, information about the test
set leaks into training and the model's test score looks better than it will be on
new data. So:  split first -> fit on train -> only TRANSFORM test with what was learned.

The fitted pipeline is saved with joblib, so new data later gets exactly the same
transformation (same means, same category columns) — no refitting.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from .plan import MAX_ONEHOT, plan_columns, yeo_johnson

SEED = 42
TEST_SIZE = 0.2
SCALER_CLASSES = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


# --- small functions used inside the pipeline (module level, so joblib can save them) ---
def log1p_clipped(x: np.ndarray) -> np.ndarray:
    """log(1 + x); a negative value in new data (never seen in train) is treated as 0."""
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


def cut_into_bins(x: pd.DataFrame, edges: list[float], labels: list[str]) -> pd.DataFrame:
    """Numbers -> group labels ('18-29'); values outside the edges -> 'other'."""
    out = {c: pd.cut(x[c], bins=edges, labels=labels, right=False).astype(object)
           .where(lambda s: s.notna(), "other") for c in x.columns}
    return pd.DataFrame(out, index=x.index)


def group_names(_: FunctionTransformer, names: list[str]) -> list[str]:
    """Output name of a binned column: age -> age_group."""
    return [f"{n}_group" for n in names]


def _steps(item: dict) -> Pipeline:
    """The scikit-learn steps for one planned column."""
    kind = item["kind"]
    if kind == "binary":
        return Pipeline([("fill", SimpleImputer(strategy="most_frequent"))])
    if kind == "numeric":
        steps = [("fill", SimpleImputer(strategy="median"))]
        if item["power"] == "log":
            log = FunctionTransformer(log1p_clipped, feature_names_out="one-to-one")
            steps.append(("log", log))
        elif item["power"] == "yeo-johnson":
            steps.append(("yeo_johnson", yeo_johnson()))
        steps.append(("scale", SCALER_CLASSES[item["scaler"]]()))
        return Pipeline(steps)
    if kind == "ordinal":
        return Pipeline([
            # values outside the order -> NaN -> filled with the median code, and
            # add_indicator adds a 0/1 column that says "this was outside the order"
            ("codes", OrdinalEncoder(categories=[item["order"]], handle_unknown="use_encoded_value",
                                     unknown_value=np.nan, encoded_missing_value=np.nan)),
            ("fill", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", MinMaxScaler()),
        ])
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=MAX_ONEHOT,
                           sparse_output=False)
    fill = SimpleImputer(strategy="constant", fill_value="missing")
    if kind == "bins":
        groups = FunctionTransformer(cut_into_bins, feature_names_out=group_names,
                                     kw_args={"edges": item["edges"], "labels": item["labels"]})
        return Pipeline([("groups", groups), ("fill", fill), ("onehot", onehot)])
    return Pipeline([("fill", fill), ("onehot", onehot)])


def build_pipeline(plan: list[dict]) -> ColumnTransformer:
    """One ColumnTransformer with one small pipeline per planned column; the rest is dropped."""
    parts = [(f"{p['column']}_{p['kind']}", _steps(p), [p["column"]]) for p in plan]
    ct = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def split(df: pd.DataFrame, target: str | None = None, test_size: float = TEST_SIZE,
          seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split with a fixed seed; stratified on the target when it is a category.

    Stratified = the test set gets the same share of 'yes' as the train set (11% in Bank).
    """
    strat = None
    if target and target in df.columns:
        counts = df[target].value_counts()
        if len(counts) > 1 and counts.min() >= 2 and not pd.api.types.is_float_dtype(df[target]):
            strat = df[target]
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)


@dataclass
class TransformResult:
    """Everything one run produces."""

    pipeline: ColumnTransformer
    train: pd.DataFrame                 # transformed train features (+ target, if any)
    test: pd.DataFrame                  # transformed test features (+ target, if any)
    report: dict = field(default_factory=dict)


def _skew(x: pd.Series) -> float:
    """Skewness, 0.0 for a constant column."""
    return 0.0 if x.nunique() < 2 else round(float(x.skew()), 3)


def transform_dataset(df: pd.DataFrame, target: str | None = None,
                      test_size: float = TEST_SIZE, seed: int = SEED,
                      config: dict | None = None) -> TransformResult:
    """Split, plan + fit on train, transform train and test, and explain every column.

    config keys (all optional): ordinal {col: [order]}, scaling {col: scaler},
    bins {col: [edges]}, drop [cols].
    """
    start = time.perf_counter()
    cfg = config or {}
    if target and target not in df.columns:
        raise ValueError(f"target column {target!r} not found")
    train_raw, test_raw = split(df, target, test_size, seed)
    plan, dropped = plan_columns(train_raw, target=target, ordinal=cfg.get("ordinal"),
                                 scaling=cfg.get("scaling"), bins=cfg.get("bins"),
                                 drop=cfg.get("drop"))
    if not plan:
        raise ValueError("no usable feature columns (all were dropped)")
    pipe = build_pipeline(plan)
    train = pipe.fit_transform(train_raw)          # learns from the train rows ONLY
    test = pipe.transform(test_raw)                # uses what was learned, learns nothing

    for p in plan:
        name = f"{p['column']}_{p['kind']}"
        p["output_columns"] = list(pipe.named_transformers_[name].get_feature_names_out())
        if p["kind"] == "numeric":
            p["skew_after"] = _skew(train[p["output_columns"][0]])
    if target:
        train[target] = train_raw[target].to_numpy()
        test[target] = test_raw[target].to_numpy()

    report = {
        "rows": {"input": int(len(df)), "train": int(len(train)), "test": int(len(test))},
        "test_size": test_size, "seed": seed, "target": target,
        "fitted_on": f"train rows only ({len(train)}); test rows were only transformed",
        "features_in": len(plan), "features_out": int(train.shape[1] - (1 if target else 0)),
        "columns": plan, "dropped": dropped,
        "skewed_columns": [{"column": p["column"], "power": p["power"],
                            "skew_before": p["skew_before"], "skew_after": p["skew_after"]}
                           for p in plan if p["kind"] == "numeric" and p["power"]],
        "seconds": round(time.perf_counter() - start, 2),
    }
    return TransformResult(pipe, train, test, report)


def save_transform(result: TransformResult, out_dir: str | Path) -> dict[str, str]:
    """Write train/test CSVs, the fitted pipeline (joblib) and the JSON report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"train": out / "train_transformed.csv", "test": out / "test_transformed.csv",
             "pipeline": out / "transform_pipeline.joblib", "report": out / "transform_report.json"}
    result.train.to_csv(files["train"], index=False)
    result.test.to_csv(files["test"], index=False)
    joblib.dump(result.pipeline, files["pipeline"])
    result.report["files"] = {k: str(v) for k, v in files.items()}
    files["report"].write_text(json.dumps(result.report, indent=2))
    return result.report["files"]


def load_pipeline(path: str | Path) -> ColumnTransformer:
    """Load a fitted pipeline saved by save_transform (only load files you made yourself:
    joblib files can run code when loaded)."""
    return joblib.load(path)
