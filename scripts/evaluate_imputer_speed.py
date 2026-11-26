"""Choose a faster iterative imputer by measuring speed AND accuracy (Sprint 11).

    python scripts/evaluate_imputer_speed.py     # needs bank_marketing.csv and adult.csv

RULE, WRITTEN BEFORE ANY MEASUREMENT (committed before the first run; never changed after):
  - Accuracy = the Sprint 5 evaluation: hide 10% of the known values of every imputable column
    (at most 5,000 complete rows, seed 42), refill them, score mean NRMSE (numbers, lower is
    better) and mean accuracy (categories, higher is better).
  - A candidate is ACCEPTABLE when, on BOTH Bank Marketing and Adult, its mean NRMSE is at most
    0.02 higher and its mean accuracy at most 1.0 percentage point lower than the current
    setting (max_iter 10, all features, skip_complete False).
  - Among acceptable candidates the FASTEST wins, measured as FoldImputer.fit time on the train
    rows of CV fold 0 (5 folds, seed 42) of Adult, the slow dataset. No acceptable candidate ->
    the current setting stays.
Candidates: skip_complete=True; n_nearest_features=15; max_iter=5; skip_complete + n_nearest 15.
Not tried: frequency / ordinal encoding of high-cardinality categories (native-country is
itself an imputation target, and a frequency code cannot be turned back into a country).
Adult uses contracts/adult_decisions.yaml (the Sprint 11 setup). Writes reports/imputer_speed.json.
`--table` prints the table again from that JSON without re-measuring.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaning import imputers  # noqa: E402
from src.cleaning.evaluation import evaluate_imputers  # noqa: E402
from src.impact.impact import RowCleaner  # noqa: E402
from src.profiling.decisions import load_decisions  # noqa: E402
from src.profiling.profiler import read_csv_raw  # noqa: E402

CANDIDATES = {
    "current": {},
    "skip_complete": {"skip_complete": True},
    "nearest_15": {"n_nearest_features": 15},
    "max_iter_5": {"max_iter": 5},
    "skip_complete+nearest_15": {"skip_complete": True, "n_nearest_features": 15},
}
MAX_NRMSE_LOSS = 0.02
MAX_ACCURACY_LOSS = 0.01
DATASETS = {"bank": ("data/raw/bank_marketing.csv", "y", None),
            "adult": ("data/raw/adult.csv", "income", "contracts/adult_decisions.yaml")}


def main() -> None:
    """Measure every candidate on both datasets, apply the rule, print the table."""
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    base = dict(imputers.ITERATIVE)
    results = {}
    for name, (path, target, decisions) in DATASETS.items():
        df = read_csv_raw(ROOT / path)
        y = df.pop(target)
        tr, _ = next(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y))
        config = {"decisions": load_decisions(ROOT / decisions)} if decisions else None
        cleaner = RowCleaner(config=config).fit(df.iloc[tr])
        train = cleaner.transform(df.iloc[tr])
        features = cleaner.impute_features_
        targets = [c for c in features if c in cleaner.impute_targets_]
        for cand, settings in CANDIDATES.items():
            imputers.ITERATIVE.clear()
            imputers.ITERATIVE.update({**base, **settings})
            t0 = time.perf_counter()
            imputers.FoldImputer("iterative", cleaner.impute_targets_, features).fit(train)
            fit_seconds = time.perf_counter() - t0
            ev = evaluate_imputers(train, targets, features, methods=("iterative",))
            m = ev["methods"]["iterative"]
            results.setdefault(cand, {})[name] = {
                "fit_seconds": round(fit_seconds, 1), "mean_nrmse": m["mean_nrmse"],
                "mean_accuracy": m["mean_accuracy"], "eval_seconds": m["seconds"],
                "train_rows": len(train), "hidden_cells": ev["hidden_cells"]}
            print(name, cand, results[cand][name], flush=True)
    imputers.ITERATIVE.clear()
    imputers.ITERATIVE.update(base)

    def ok(cand: str) -> bool:
        for name in DATASETS:
            cur, new = results["current"][name], results[cand][name]
            if cur["mean_nrmse"] is not None and new["mean_nrmse"] - cur["mean_nrmse"] > \
                    MAX_NRMSE_LOSS:
                return False
            if cur["mean_accuracy"] is not None and cur["mean_accuracy"] - new["mean_accuracy"] > \
                    MAX_ACCURACY_LOSS:
                return False
        return True
    acceptable = [c for c in CANDIDATES if c != "current" and ok(c)]
    winner = min(acceptable, key=lambda c: results[c]["adult"]["fit_seconds"], default="current")
    if winner != "current" and results[winner]["adult"]["fit_seconds"] >= \
            results["current"]["adult"]["fit_seconds"]:
        winner = "current"
    out = {"rule": {"max_nrmse_loss": MAX_NRMSE_LOSS, "max_accuracy_loss": MAX_ACCURACY_LOSS},
           "candidates": CANDIDATES, "results": results, "acceptable": acceptable,
           "winner": winner}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/imputer_speed.json").write_text(json.dumps(out, indent=2) + "\n")
    print_table(out)


def print_table(out: dict) -> None:
    """The markdown table and the winner from a result dict."""
    results, acceptable, winner = out["results"], out["acceptable"], out["winner"]
    print("\n| setting | Bank fit s | Bank NRMSE | Bank accuracy | Adult fit s | Adult NRMSE | "
          "Adult accuracy | acceptable |")
    print("|---|---|---|---|---|---|---|---|")
    for c in CANDIDATES:
        b, a = results[c]["bank"], results[c]["adult"]

        def f(v: float | None, pct: bool = False) -> str:
            return "-" if v is None else (f"{v:.2%}" if pct else f"{v:.4f}")
        print(f"| {c} | {b['fit_seconds']} | {f(b['mean_nrmse'])} | "
              f"{f(b['mean_accuracy'], True)} | {a['fit_seconds']} | {f(a['mean_nrmse'])} | "
              f"{f(a['mean_accuracy'], True)} | "
              f"{'baseline' if c == 'current' else ('yes' if c in acceptable else 'no')} |")
    print(f"\nwinner: {winner}")


if __name__ == "__main__":
    if "--table" in sys.argv:
        print_table(json.loads((ROOT / "reports/imputer_speed.json").read_text()))
    else:
        main()
