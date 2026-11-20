"""Imputer evaluation — which method fills missing values best, measured on known values.

The idea ("hide and seek"):
  1. take only rows with NO missing values, so the true value of every cell is known
     (at most MAX_ROWS of them, chosen with a fixed seed, to keep KNN fast)
  2. hide 10% of those known cells at random (fixed seed -> the same cells every run)
  3. let every method fill the hidden cells
  4. compare the filled values with the true ones:
       numbers     RMSE = sqrt(mean(error^2))  (big misses count extra)
                   MAE  = mean(|error|)        (the average miss, in the column's units)
                   NRMSE = RMSE / std of the column, so columns with different units can
                   be averaged: 1.0 = as bad as guessing the average, 0 = perfect
       categories  accuracy = share of hidden cells filled with the right category
     plus the run time of each method.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .imputers import impute_columns, is_numeric  # noqa: E402

EVALUATION_VERSION = "1.0"
EVAL_METHODS = ("median", "knn", "regression", "iterative")
HIDE_SHARE = 0.10
MAX_ROWS = 5000
SEED = 42

# Fixed colour per method (categorical palette, validated for colour-blind readers)
COLOURS = {"median": "#2a78d6", "knn": "#eb6834", "regression": "#1baf7a",
           "iterative": "#eda100"}


def hide_cells(df: pd.DataFrame, columns: list[str], share: float = HIDE_SHARE,
               seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hide `share` of the cells in `columns` at random.

    Returns (masked copy of df with NaN in the hidden cells, True/False mask of hidden cells).
    """
    rng = np.random.default_rng(seed)
    n_cells = len(df) * len(columns)
    picked = rng.choice(n_cells, size=int(round(n_cells * share)), replace=False)
    mask = np.zeros(n_cells, dtype=bool)
    mask[picked] = True
    hidden = pd.DataFrame(mask.reshape(len(df), len(columns)), index=df.index, columns=columns)
    masked = df.copy()
    for c in columns:
        masked[c] = masked[c].mask(hidden[c])
    return masked, hidden


def _score(true: pd.Series, guess: pd.Series, column_std: float) -> dict:
    """Metrics for one column's hidden cells: RMSE/MAE/NRMSE for numbers, accuracy otherwise."""
    if is_numeric(true):
        err = guess.astype(float) - true.astype(float)
        rmse = float(np.sqrt((err ** 2).mean()))
        return {"type": "numeric", "hidden": int(len(true)), "rmse": round(rmse, 4),
                "mae": round(float(err.abs().mean()), 4),
                "nrmse": round(rmse / column_std, 4) if column_std else None}
    return {"type": "categorical", "hidden": int(len(true)),
            "accuracy": round(float((guess.astype(str) == true.astype(str)).mean()), 4)}


def evaluate_imputers(df: pd.DataFrame, targets: list[str], features: list[str],
                      methods: tuple[str, ...] = EVAL_METHODS, share: float = HIDE_SHARE,
                      seed: int = SEED, max_rows: int = MAX_ROWS) -> dict:
    """Hide `share` of the known values in `targets`, fill them with each method, score them.

    df:        the data after cleaning steps 1-9 (before imputation, so no invented values)
    targets:   columns whose cells are hidden and scored
    features:  columns the ML methods may look at
    Returns a dict (see docs/module2_cleaning.md) with per-column and average scores and
    the best method for numeric and for categorical columns.
    """
    columns = list(dict.fromkeys([*targets, *features]))
    complete = df[columns].dropna()
    rows_complete = len(complete)
    if rows_complete > max_rows:
        complete = complete.sample(n=max_rows, random_state=seed)
    if len(complete) < 10 or not targets:
        return {"evaluation_version": EVALUATION_VERSION, "skipped":
                "fewer than 10 complete rows or nothing to evaluate"}
    masked, hidden = hide_cells(complete, targets, share, seed)

    results = {}
    for method in methods:
        start = time.perf_counter()
        filled = impute_columns(masked, targets, features, method)
        seconds = time.perf_counter() - start
        per_col = {c: _score(complete.loc[hidden[c], c], filled.loc[hidden[c], c],
                             float(complete[c].std()) if is_numeric(complete[c]) else 0.0)
                   for c in targets if hidden[c].any()}
        nrmse = [m["nrmse"] for m in per_col.values() if m.get("nrmse") is not None]
        acc = [m["accuracy"] for m in per_col.values() if "accuracy" in m]
        results[method] = {
            "seconds": round(seconds, 3),
            "mean_nrmse": round(float(np.mean(nrmse)), 4) if nrmse else None,
            "mean_accuracy": round(float(np.mean(acc)), 4) if acc else None,
            "columns": per_col,
        }

    def best(key: str, lower_is_better: bool) -> str | None:
        scored = {m: r[key] for m, r in results.items() if r[key] is not None}
        if not scored:
            return None
        return (min if lower_is_better else max)(scored, key=scored.get)

    return {
        "evaluation_version": EVALUATION_VERSION,
        "rows_complete": int(rows_complete),
        "rows_used": int(len(complete)),
        "hidden_share": share,
        "hidden_cells": int(hidden.to_numpy().sum()),
        "seed": seed,
        "targets": targets,
        "features": features,
        "methods": results,
        "best": {"numeric": best("mean_nrmse", True),
                 "categorical": best("mean_accuracy", False)},
    }


def _bar_panel(ax: plt.Axes, methods: list[str], values: list[float | None], title: str,
               fmt: str) -> None:
    """One small bar chart: one bar per method, value written on top, recessive axes."""
    xs = np.arange(len(methods))
    heights = [v if v is not None else 0 for v in values]
    ax.bar(xs, heights, width=0.6, color=[COLOURS.get(m, "#888888") for m in methods])
    for x, v in zip(xs, values):
        if fmt == "seconds":                       # small times need more decimals
            text = "n/a" if v is None else (f"{v:.2f}" if v >= 0.1 else f"{v:.3f}")
        else:
            text = "n/a" if v is None else format(v, fmt)
        ax.annotate(text, (x, v or 0),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(xs, methods, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", color="#222222")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=8, colors="#666666", length=0)
    ax.margins(y=0.15)


def save_evaluation(result: dict, out_dir: str | Path, title: str = "") -> dict[str, str]:
    """Write imputer_evaluation.json and imputer_evaluation.png to out_dir; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, png_path = out_dir / "imputer_evaluation.json", out_dir / "imputer_evaluation.png"
    json_path.write_text(json.dumps(result, indent=2))
    if "methods" not in result:
        return {"json": str(json_path)}

    methods = list(result["methods"])
    r = result["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    _bar_panel(axes[0], methods, [r[m]["mean_nrmse"] for m in methods],
               "Numeric error: mean NRMSE (lower is better)", ".3f")
    _bar_panel(axes[1], methods, [r[m]["mean_accuracy"] for m in methods],
               "Categorical accuracy (higher is better)", ".1%")
    _bar_panel(axes[2], methods, [r[m]["seconds"] for m in methods],
               "Run time in seconds (lower is better)", "seconds")
    fig.suptitle(f"{title} — {result['hidden_cells']:,} hidden cells "
                 f"({result['hidden_share']:.0%} of {result['rows_used']:,} complete rows)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)
    return {"json": str(json_path), "chart": str(png_path)}
