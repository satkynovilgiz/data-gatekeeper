"""Controlled corruption of clean data, to test how models cope with dirty inputs (Sprint 8).

Every injector takes a DataFrame of raw text values, a fraction and a seed, never changes
its input, and returns (corrupted copy, number of cells or rows changed). The number is
always exactly round(fraction x N), where N is:

  inject_missing          N = cells in `columns`             -> the value becomes empty
  inject_sentinels        N = cells in `columns`             -> numeric: "999", other: "?"
                          (both are in the tool's own lists: NUMERIC_SENTINELS, PLACEHOLDERS)
  inject_numbers_as_text  N = cells in `numeric_columns`     -> decimal comma, as a pt/de
                          spreadsheet writes it: 4.857 -> "4,857", 42 -> "42,0"
                          (pandas reads " 42 " fine, so padding would not be corruption)
  inject_duplicates       N = rows                           -> that many copies of random
                          rows appended (the label comes along with its row)

Cells are only taken from cells that are still original, so the injectors can be chained
(corrupt()) without one undoing another, and each still changes exactly its share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_SENTINEL = "999"
TEXT_SENTINEL = "?"
KINDS = ("missing", "sentinels", "numbers_as_text", "duplicates")


def _pick(df: pd.DataFrame, columns: list[str], eligible: pd.DataFrame, frac: float,
          rng: np.random.Generator) -> list[tuple[int, str]]:
    """Exactly round(frac x len(df) x len(columns)) random (row position, column) cells.

    Only cells where `eligible` is True are chosen. ValueError when there are too few.
    """
    if not 0 <= frac <= 1:
        raise ValueError("fraction must be between 0 and 1")
    n = round(frac * len(df) * len(columns))
    cells = [(r, c) for c in columns for r in np.flatnonzero(eligible[c].to_numpy())]
    if n > len(cells):
        raise ValueError(f"asked for {n} cells but only {len(cells)} can be changed")
    chosen = rng.choice(len(cells), size=n, replace=False) if n else []
    return [cells[i] for i in chosen]


def _apply(df: pd.DataFrame, cells: list[tuple[int, str]], values: list[object]) -> pd.DataFrame:
    out = df.copy()
    for (r, c), v in zip(cells, values):
        out.iat[r, out.columns.get_loc(c)] = v
    return out


def _is_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").notna()


def inject_missing(df: pd.DataFrame, frac: float, seed: int,
                   columns: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Empty a share of the (non-empty) cells in columns."""
    columns = list(columns if columns is not None else df.columns)
    cells = _pick(df, columns, df[columns].notna(), frac, np.random.default_rng(seed))
    return _apply(df, cells, [np.nan] * len(cells)), len(cells)


def inject_sentinels(df: pd.DataFrame, frac: float, seed: int, numeric_columns: list[str],
                     columns: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Replace a share of cells with "no value" codes: 999 in numbers, "?" in text."""
    columns = list(columns if columns is not None else df.columns)
    eligible = df[columns].notna() & ~df[columns].isin([NUMERIC_SENTINEL, TEXT_SENTINEL])
    cells = _pick(df, columns, eligible, frac, np.random.default_rng(seed))
    values = [NUMERIC_SENTINEL if c in numeric_columns else TEXT_SENTINEL for _, c in cells]
    return _apply(df, cells, values), len(cells)


def decimal_comma(value: str) -> str:
    """'4.857' -> '4,857'; '42' -> '42,0' (a number pandas can no longer read)."""
    text = str(value).strip()
    return text.replace(".", ",") if "." in text else text + ",0"


def inject_numbers_as_text(df: pd.DataFrame, frac: float, seed: int,
                           numeric_columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Write a share of the numbers with a decimal comma."""
    cols = list(numeric_columns)
    eligible = pd.DataFrame({c: _is_number(df[c]) & (df[c] != NUMERIC_SENTINEL) for c in cols})
    cells = _pick(df, cols, eligible, frac, np.random.default_rng(seed))
    values = [decimal_comma(df.iat[r, df.columns.get_loc(c)]) for r, c in cells]
    return _apply(df, cells, values), len(cells)


def inject_duplicates(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, int]:
    """Append round(frac x rows) copies of randomly chosen rows (each row at most once)."""
    if not 0 <= frac <= 1:
        raise ValueError("fraction must be between 0 and 1")
    n = round(frac * len(df))
    rows = np.random.default_rng(seed).choice(len(df), size=n, replace=False)
    return pd.concat([df, df.iloc[np.sort(rows)]], ignore_index=True), n


def corrupt(df: pd.DataFrame, frac: float, seed: int, numeric_columns: list[str],
            columns: list[str], kinds: tuple[str, ...] = KINDS) -> tuple[pd.DataFrame, dict]:
    """Apply several injectors in a fixed order, each at the same fraction.

    columns: the feature columns that may be changed (never the label);
    each injector gets its own seed (seed, seed + 1, ...), so a run is repeatable.
    Returns (corrupted copy, {kind: cells or rows changed}).
    """
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown corruption kind(s) {sorted(unknown)}; use {KINDS}")
    out, changed = df, {}
    numeric = [c for c in numeric_columns if c in columns]
    for i, kind in enumerate(KINDS):
        if kind not in kinds:
            continue
        if kind == "missing":
            out, changed[kind] = inject_missing(out, frac, seed + i, columns)
        elif kind == "sentinels":
            out, changed[kind] = inject_sentinels(out, frac, seed + i, numeric, columns)
        elif kind == "numbers_as_text":
            out, changed[kind] = inject_numbers_as_text(out, frac, seed + i, numeric)
        else:
            out, changed[kind] = inject_duplicates(out, frac, seed + i)
    return out, changed
