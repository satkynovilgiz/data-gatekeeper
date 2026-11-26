"""Why does the gatekeeper make Adult models WORSE? An ablation (Sprint 10 finding).

    python scripts/diagnose_adult_impact.py      # needs data/raw/adult.csv; ~10 minutes

Same 5-fold CV as `gatekeeper impact` (seed 42), A raw vs C (gatekeeper with the median
imputer, so it runs fast), under four cleaning settings:
  default          as shipped: "?" -> missing -> imputed; capital-gain 99999 -> missing
  keep "?"         workclass / occupation / native-country keep "?" as their own category
                   (the existing keep_as_category option, like Bank's "default")
  keep 99999       capital-gain 99999 stays a number (removed from that column's codes)
  keep both
This only diagnoses: the shipped defaults are NOT changed by it.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.impact import impact as imp  # noqa: E402
from src.profiling.profiler import read_csv_raw  # noqa: E402

QUESTION_COLUMNS = ["workclass", "occupation", "native-country"]
ORIGINAL = imp.RowCleaner      # every setting subclasses the shipped cleaner, never another one


def cleaner_class(keep_question: bool, keep_99999: bool) -> type:
    """A RowCleaner with the chosen cleaning decisions switched off."""
    base = ORIGINAL

    class Ablated(base):
        def fit(self, X, y=None):  # noqa: ANN001, ANN201
            if keep_question:
                self.config = {"keep_as_category": ["default", *QUESTION_COLUMNS]}
            out = super().fit(X, y)
            if keep_99999:
                col = self.schema_["columns"]["capital-gain"]
                col["missing_codes"] = [m for m in col["missing_codes"] if str(m) != "99999"]
                # re-run the row steps WITHOUT step 8b, so the train range includes 99999
                learned, self.config_["learned"] = self.config_["learned"], None
                run = self._run(X)
                self.config_["learned"] = learned
                s = run.df["capital-gain"].astype(float)
                learned["ranges"]["capital-gain"] = [float(s.min()), float(s.max())]
                learned["sentinel_share"]["capital-gain"].pop(99999.0, None)
            return out
    return Ablated


def main() -> None:
    """Run the four settings and print C - A per model and metric."""
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    df = read_csv_raw(ROOT / "data/raw/adult.csv")
    settings = [("default", False, False), ('keep "?"', True, False),
                ("keep 99999", False, True), ("keep both", True, True)]
    print("| setting | model | ROC-AUC C - A [95% CI] | PR-AUC C - A [95% CI] |")
    print("|---|---|---|---|")
    try:
        for name, q, k in settings:
            imp.RowCleaner = cleaner_class(q, k)
            r = imp.run_impact(df, "income", variants=("A", "C"), robustness=False,
                               dataset="adult")
            for m in ("logreg", "hgb"):
                c = r["comparisons"]["C-A"][m]
                cells = [f"{c[x]['mean_diff']:+.4f} [{c[x]['ci95'][0]:+.4f}, "
                         f"{c[x]['ci95'][1]:+.4f}]" for x in ("roc_auc", "pr_auc")]
                print(f"| {name} | {m} | {cells[0]} | {cells[1]} |", flush=True)
    finally:
        imp.RowCleaner = ORIGINAL


if __name__ == "__main__":
    main()
