"""Why does B (iterative imputer) lose to A under ALL corruption kinds? (Sprint 8 finding)

    python scripts/diagnose_combined_corruption.py      # needs data/raw/bank_marketing.csv

Fold 0 of the impact CV, HistGradientBoosting, 20% corruption with the same seed scheme
as `gatekeeper impact`. Scores A / B / C with and without the sentinel injector, and shows
how many of B's filled values sit on the clip edge (train min / max).
Sprint 9: also prints what the sentinel fix did (step 8b out_of_range_to_missing per column,
and how many cells the imputer filled with its median / mode fallback).
Sprint 10: also the fake duration = 999 left after cleaning (sentinels only, 20%), and the
frequency rule's decision for every code it saw.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.impact import impact as imp  # noqa: E402
from src.impact.corruption import KINDS, corrupt  # noqa: E402
from src.profiling.profiler import read_csv_raw  # noqa: E402

LEVEL, LEVEL_INDEX, FOLD = 0.20, 3, 0


def main() -> None:
    """Print the three comparisons and the clip-edge shares."""
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    df = read_csv_raw(ROOT / "data/raw/bank_marketing.csv")
    y = (df["y"] == "yes").astype(int).to_numpy()
    X = df.drop(columns="y")
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=imp.SEED).split(X, y))[FOLD]
    fm, _ = imp._fit_fold(X.iloc[tr], y[tr], ("hgb",), ("A", "B", "C"))
    test = X.iloc[te].reset_index(drop=True).assign(__label__=y[te])
    numeric = fm.raw["hgb"].named_steps["prep"].numeric_
    seed = imp.SEED + 1000 * FOLD + 10 * LEVEL_INDEX
    for kinds in (KINDS, ("missing", "numbers_as_text", "duplicates"), ("missing", "sentinels")):
        dirty, _ = corrupt(test, LEVEL, seed, numeric, list(X.columns), kinds)
        yd = dirty.pop("__label__").to_numpy().astype(int)
        auc = {v: round(imp.score(yd, imp._predict(fm, v, "hgb", dirty))["roc_auc"], 4)
               for v in "ABC"}
        print(f"{' + '.join(kinds):45} ROC-AUC {auc}")
    dirty, _ = corrupt(test, LEVEL, seed, numeric, list(X.columns), KINDS)
    dirty.pop("__label__")
    cleaned = fm.cleaner.transform(dirty)
    filled = fm.imputers["B"].transform(cleaned)
    moved = getattr(fm.cleaner, "out_of_range_", {})
    print(f"out_of_range_to_missing: {sum(moved.values())} cells "
          f"({', '.join(f'{c} {n}' for c, n in sorted(moved.items(), key=lambda x: -x[1])[:6])})")
    print(f"imputer fallback to median / mode: {getattr(fm.imputers['B'], 'fallbacks_', 0)} cells")
    # the Sprint 9 gap: injected 999s in duration (which has two real 999s)
    dirty, _ = corrupt(test, LEVEL, seed, numeric, list(X.columns), ("sentinels",))
    dirty.pop("__label__")
    fake = (dirty["duration"] == "999") & (test["duration"] != "999")
    left = fm.cleaner.transform(dirty)["duration"] == 999
    print(f"sentinels only: duration cells turned into 999 {int(fake.sum())}, still 999 after "
          f"cleaning {int((left & fake.to_numpy()).sum())}")
    for d in getattr(fm.cleaner, "sentinel_log_", []):
        if d["column"] == "duration":
            print(f"  duration code {d['code']:g}: train share {d['train_share']:.4%}, new share "
                  f"{d['new_share']:.4%} -> {d['decision']}")
    for c in ("age", "euribor3m", "nr.employed", "emp.var.rate"):
        gaps = cleaned[c].isna()
        lo, hi, _ = fm.imputers["B"].ranges_[c]
        v = filled.loc[gaps, c].astype(float)
        edge = float(((v <= lo) | (v >= hi)).mean())
        print(f"{c:13} gaps {int(gaps.sum()):5}  filled on the clip edge {edge:.1%}  "
              f"999 left as a number {int((cleaned[c] == 999).sum())}")


if __name__ == "__main__":
    main()
