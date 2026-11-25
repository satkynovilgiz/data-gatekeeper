"""Plant known anomalies in real data, to measure anomaly detection honestly (Sprint 9).

inject(df, share=0.01, seed=42) changes exactly round(share x rows) rows (in place of the
original values; the row count stays the same) and returns the ground truth. Three kinds,
split as evenly as possible (the last kind takes the remainder):

  extreme      ONE random column gets a value far outside what the column ever has:
               max + 0.5 x (max - min). One value is obviously wrong.
  combination  the two most correlated columns (Bank: emp.var.rate / euribor3m, r = 0.97):
               the second one is set to its 5th or 95th percentile, on the side that goes
               AGAINST the first one (a high first value gets a low second value when they
               normally rise together). Each value alone is normal; the pair is not.
  shift        EVERY usable column moves by 1.5 robust standard deviations (1.4826 x MAD),
               up or down at random. No single value is extreme; the row as a whole is odd.

Usable columns: numeric, at most 50% empty, MAD > 0 (so a "robust standard deviation"
exists), and not a column with a special code (pdays). Values are written back as text
with the column's own number of decimals. Everything is decided by the seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning.outliers import robust_scale

KINDS = ("extreme", "combination", "shift")
SHARE = 0.01
SEED = 42
SHIFT_SDS = 1.5
LOW_Q, HIGH_Q = 0.05, 0.95


def decimals(s: pd.Series) -> int:
    """How many decimals a text column of numbers uses (0 for whole numbers)."""
    text = s.dropna().astype(str)
    parts = text[text.str.contains(".", regex=False)].str.split(".").str[1]
    return int(parts.str.len().max()) if len(parts) else 0


def usable_columns(clean: pd.DataFrame, skip: list[str] | None = None) -> list[str]:
    """Numeric, mostly filled, MAD > 0, not skipped (label, special-code columns)."""
    out = []
    for c in clean.columns:
        s = clean[c]
        if c in (skip or []) or not pd.api.types.is_numeric_dtype(s) or s.isna().mean() > 0.5:
            continue
        if robust_scale(s) is not None:
            out.append(c)
    return out


def _write(value: float, places: int) -> str:
    return str(int(round(value))) if places == 0 else f"{value:.{places}f}"


def inject(df: pd.DataFrame, clean: pd.DataFrame, columns: list[str], share: float = SHARE,
           seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plant anomalies -> (changed copy of df, labels: row, kind, columns, before, after).

    df: the raw file (text); clean: the same rows as numbers (used for statistics only);
    columns: usable_columns(clean). Rows are positions 0..n-1.
    """
    if not 0 < share < 1:
        raise ValueError("share must be between 0 and 1")
    if len(columns) < 2:
        raise ValueError("need at least two usable numeric columns")
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    num = clean.reset_index(drop=True)[columns].astype(float)
    n = round(share * len(df))
    sizes = [n // len(KINDS)] * (len(KINDS) - 1)
    sizes.append(n - sum(sizes))
    places = {c: decimals(df[c]) for c in columns}
    labels = []

    # combination: the most correlated pair, rows where the planted value really changes
    corr = num.corr().abs().where(~np.eye(len(columns), dtype=bool))
    a, b = corr.stack().idxmax()
    sign = np.sign(num[a].corr(num[b]))
    low, high = num[b].quantile(LOW_Q), num[b].quantile(HIGH_Q)
    a_high = num[a] >= num[a].median()
    target = pd.Series(np.where(a_high == (sign > 0), low, high), index=num.index)
    target_text = target.map(lambda v: _write(v, places[b]))
    eligible = num.index[(out[b].astype(str) != target_text) & num[a].notna()]
    combo_rows = rng.choice(eligible, size=sizes[1], replace=False)
    rest = np.setdiff1d(num.index, combo_rows)
    others = rng.choice(rest, size=sizes[0] + sizes[2], replace=False)
    extreme_rows, shift_rows = others[:sizes[0]], others[sizes[0]:]

    for r in extreme_rows:
        c = columns[rng.integers(len(columns))]
        lo, hi = num[c].min(), num[c].max()
        before = out.at[r, c]
        out.at[r, c] = _write(hi + 0.5 * (hi - lo), places[c])
        labels.append({"row": int(r), "kind": "extreme", "columns": c,
                       "before": before, "after": out.at[r, c]})
    for r in combo_rows:
        before = out.at[r, b]
        out.at[r, b] = target_text[r]
        labels.append({"row": int(r), "kind": "combination", "columns": f"{a}+{b}",
                       "before": before, "after": out.at[r, b]})
    scales = {c: robust_scale(num[c]) for c in columns}
    for r in shift_rows:
        changed = []
        for c in columns:
            if pd.isna(num.at[r, c]):
                continue
            step = SHIFT_SDS * scales[c] * (1 if rng.random() < 0.5 else -1)
            out.at[r, c] = _write(num.at[r, c] + step, places[c])
            changed.append(c)
        labels.append({"row": int(r), "kind": "shift", "columns": "+".join(changed),
                       "before": "", "after": ""})
    truth = pd.DataFrame(labels).sort_values("row").reset_index(drop=True)
    return out, truth
