"""Which transform choice costs LogisticRegression on Adult? An ablation (Sprint 11, part 5).

    python scripts/ablate_adult_transform.py      # needs data/raw/adult.csv; ~10 minutes

Sprint 10 left a LogReg gap (C - A ROC-AUC -0.0079) even with "?" and 99999 kept. Same CV as
`gatekeeper impact` (5 folds, seed 42), A raw vs C (median imputer, fast) WITH the confirmed
decisions (contracts/adult_decisions.yaml), one Sprint 6 transform choice switched off at a time:
  as shipped            skewed columns -> log / Yeo-Johnson; > 15 categories -> rare ones grouped
  no power transform    skewed columns are only scaled
  no infrequent group   every category gets its own one-hot column (native-country: 41)
  both off
Only diagnoses: the shipped transform is not changed by it.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.impact import impact as imp  # noqa: E402
from src.profiling.decisions import load_decisions  # noqa: E402
from src.profiling.profiler import read_csv_raw  # noqa: E402
from src.transform import pipeline as tpipe  # noqa: E402
from src.transform import plan as tplan  # noqa: E402

ORIGINAL = {"SKEW_LIMIT": tplan.SKEW_LIMIT, "MAX_ONEHOT": tplan.MAX_ONEHOT}


def set_transform(power: bool, grouping: bool) -> None:
    """Switch the two choices on / off in the plan and the pipeline modules."""
    tplan.SKEW_LIMIT = ORIGINAL["SKEW_LIMIT"] if power else float("inf")
    big = ORIGINAL["MAX_ONEHOT"] if grouping else 10_000
    tplan.MAX_ONEHOT = big
    tpipe.MAX_ONEHOT = big


def main() -> None:
    """Run the four settings and print C - A for LogReg (and HGB for reference)."""
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    df = read_csv_raw(ROOT / "data/raw/adult.csv")
    config = {"decisions": load_decisions(ROOT / "contracts/adult_decisions.yaml")}
    settings = [("as shipped", True, True), ("no power transform", False, True),
                ("no infrequent group", True, False), ("both off", False, False)]
    print("| transform | model | ROC-AUC C - A [95% CI] | PR-AUC C - A [95% CI] |")
    print("|---|---|---|---|")
    try:
        for name, power, grouping in settings:
            set_transform(power, grouping)
            r = imp.run_impact(df, "income", variants=("A", "C"), robustness=False,
                               dataset="adult", cleaner_config=config)
            for m in ("logreg", "hgb"):
                c = r["comparisons"]["C-A"][m]
                cells = [f"{c[x]['mean_diff']:+.4f} [{c[x]['ci95'][0]:+.4f}, "
                         f"{c[x]['ci95'][1]:+.4f}]" for x in ("roc_auc", "pr_auc")]
                print(f"| {name} | {m} | {cells[0]} | {cells[1]} |", flush=True)
    finally:
        set_transform(True, True)


if __name__ == "__main__":
    main()
