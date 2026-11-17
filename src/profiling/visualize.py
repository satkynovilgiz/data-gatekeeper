"""Backend-generated profiling charts (saved as PNG files, no notebook needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed — works on servers and in Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def missing_heatmap(df: pd.DataFrame, path: Path) -> Path:
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


def make_all(df: pd.DataFrame, numeric_cols: list[str], out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    charts = {
        "missing_heatmap": missing_heatmap(df, out_dir / "missing_heatmap.png"),
        "correlation_heatmap": correlation_heatmap(num_df, out_dir / "correlation_heatmap.png"),
        "outlier_distribution": outlier_boxplots(num_df, out_dir / "outlier_distribution.png"),
    }
    return {k: str(v.name) for k, v in charts.items() if v is not None}
