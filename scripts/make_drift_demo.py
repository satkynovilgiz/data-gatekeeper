"""Make the Sprint 7 drift demo files from the real Bank Marketing data (seed 42).

    python scripts/make_drift_demo.py            # needs data/raw/bank_marketing.csv

Writes data/demo/ (not committed; run this script to rebuild it):
  bank_reference.csv       70% of the rows, stratified on y      -> the "good" data
  bank_current.csv         the other 30%, same split              -> new data, no drift
  bank_current_shifted.csv bank_current.csv with 3 realistic changes:
    1. duration   seconds -> minutes (/60, 1 decimal): an upstream system changed unit
    2. contact    80% of "telephone" rows become "cellular": the call centre moved to mobiles
    3. age        5% of rows emptied: a form field became optional

Why a stratified random split and not a time split: the file is ordered by time
(May 2008 - Nov 2010) and the economic columns (euribor3m, nr.employed, ...) change
with time, so a time split would show large drift before we change anything. A random
split gives a "current" file from the same population, so only our changes should drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.profiling.profiler import read_csv_raw  # noqa: E402

SEED = 42
SOURCE = ROOT / "data/raw/bank_marketing.csv"
OUT = ROOT / "data/demo"
TELEPHONE_TO_CELLULAR = 0.80
AGE_MISSING = 0.05


def main() -> None:
    """Split the bank data, apply the three shifts, write data/demo/."""
    if not SOURCE.exists():
        sys.exit(f"{SOURCE} not found: run python scripts/download_bank_marketing.py first")
    df = read_csv_raw(SOURCE)                       # every value kept as text, unchanged
    reference, current = train_test_split(df, test_size=0.30, stratify=df["y"],
                                          random_state=SEED)
    reference, current = reference.sort_index(), current.sort_index()

    rng = np.random.default_rng(SEED)
    shifted = current.copy()
    # 1. duration in minutes instead of seconds
    shifted["duration"] = (shifted["duration"].astype(float) / 60).round(1).astype(str)
    # 2. most telephone contacts become cellular
    tel = shifted.index[shifted["contact"] == "telephone"]
    moved = rng.choice(tel, size=round(len(tel) * TELEPHONE_TO_CELLULAR), replace=False)
    shifted.loc[moved, "contact"] = "cellular"
    # 3. 5% of ages missing
    emptied = rng.choice(shifted.index, size=round(len(shifted) * AGE_MISSING), replace=False)
    shifted.loc[emptied, "age"] = np.nan

    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [("bank_reference", reference), ("bank_current", current),
                        ("bank_current_shifted", shifted)]:
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(frame)} rows, y = yes {(frame['y'] == 'yes').mean():.2%}")
    print(f"shifted: duration /60 on {len(shifted)} rows, {len(moved)} telephone -> cellular, "
          f"{len(emptied)} ages emptied")


if __name__ == "__main__":
    main()
