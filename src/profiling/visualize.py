"""Backend-generated profiling charts (saved as PNG files, no notebook needed).

Dataset charts:  missing heatmap, correlation heatmap, outlier boxplots
Column charts:   one histogram per numeric column, one top-10 bar chart per
                 categorical column (Sprint 3). These are drawn from the numbers
                 already in the report, so the picture always matches the JSON.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed — works on servers and in Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .profiler import numeric_frame


def missing_heatmap(df: pd.DataFrame, path: Path) -> Path:
    """Draw a rows x columns grid where dark cells are empty; save it to path (PNG).

    Note: it only shows real empty cells, not disguised ones like "unknown".
    """
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.6), 5))
    ax.imshow(df.isna().values, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Row")
    ax.set_title("Missing values (dark = missing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def correlation_heatmap(num_df: pd.DataFrame, path: Path) -> Path | None:
    """Draw the Pearson correlation matrix of the numeric columns and save it to path.

    Returns None (no file) when there are fewer than 2 numeric columns.
    """
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap (Pearson)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def outlier_boxplots(num_df: pd.DataFrame, path: Path) -> Path | None:
    """Draw one boxplot per numeric column (as z-scores, so scales are comparable).

    Saves to path; returns None (no file) when there are no numeric columns.
    """
    if num_df.empty:
        return None
    # standardise so columns with different scales fit one chart
    z = (num_df - num_df.mean()) / num_df.std(ddof=0)
    fig, ax = plt.subplots(figsize=(max(6, num_df.shape[1] * 1.2), 4.5))
    ax.boxplot([z[c].dropna() for c in z.columns], showfliers=True)
    ax.set_xticks(range(1, len(z.columns) + 1))
    ax.set_xticklabels(z.columns, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("z-score")
    ax.set_title("Outlier distribution (standardised)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def safe_file_name(column: str) -> str:
    """Make a column name safe for a file name: 'emp.var.rate' stays, 'my col/x' -> 'my_col_x'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", column).strip("_") or "column"


def histogram_chart(col: dict, path: Path) -> Path | None:
    """Bar plot of the histogram stored in col['numeric_stats']['histogram']."""
    hist = col.get("numeric_stats", {}).get("histogram")
    if not hist:
        return None
    edges, counts = hist["bin_edges"], hist["counts"]
    fig, ax = plt.subplots(figsize=(6, 4))
    widths = np.diff(edges)
    widths[widths == 0] = 1                 # single-value column: give the bar a visible width
    ax.bar(edges[:-1], counts, width=widths, align="edge",
           color="#4C72B0", edgecolor="white")
    ax.set_xlabel(col["name"])
    ax.set_ylabel("Rows")
    title = f"Distribution of {col['name']}"
    if col["disguised_missing"]["count"]:   # e.g. pdays: the 999s are not in the picture
        title += f"\n(without {col['disguised_missing']['count']} disguised missing values)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def bar_chart(col: dict, path: Path, top_n: int = 10) -> Path | None:
    """Horizontal bar chart of the top N categories (most common at the top)."""
    top = col.get("categorical_stats", {}).get("top_values")
    if not top:
        return None
    items = list(top.items())[:top_n][::-1]          # reverse: biggest bar on top
    labels, counts = [k for k, _ in items], [v for _, v in items]
    # placeholders like 'unknown' are coloured differently, so they stand out
    hidden = col["disguised_missing"]["values"]
    colors = ["#DD8452" if k in hidden else "#4C72B0" for k in labels]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(items) + 1)))
    # bars by position + our own labels (passing strings like "1" makes matplotlib guess types)
    bars = ax.barh(range(len(items)), counts, color=colors)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)   # show the count (tiny bars stay readable)
    ax.margins(x=0.12)                                     # room for the labels
    ax.set_xlabel("Rows")
    ax.set_title(f"Top {len(items)} values of {col['name']}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def make_all(df: pd.DataFrame, report: dict, out_dir: str | Path) -> dict:
    """Draw every chart into out_dir and return the file names for report['charts']."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_df = numeric_frame(df, report["columns"])   # 999 etc. -> NaN, same as the report
    charts = {
        "missing_heatmap": missing_heatmap(df, out_dir / "missing_heatmap.png"),
        "correlation_heatmap": correlation_heatmap(num_df, out_dir / "correlation_heatmap.png"),
        "outlier_distribution": outlier_boxplots(num_df, out_dir / "outlier_distribution.png"),
    }
    result = {k: v.name for k, v in charts.items() if v is not None}

    # one chart per column: histogram (numeric) or top-10 bar chart (categorical)
    result["histograms"], result["bar_charts"] = {}, {}
    for col in report["columns"]:
        name = safe_file_name(col["name"])
        if col.get("numeric_stats"):
            path = histogram_chart(col, out_dir / f"hist_{name}.png")
            if path:
                result["histograms"][col["name"]] = path.name
        elif col["inferred_type"] == "categorical":
            path = bar_chart(col, out_dir / f"bar_{name}.png")
            if path:
                result["bar_charts"][col["name"]] = path.name
    return result
