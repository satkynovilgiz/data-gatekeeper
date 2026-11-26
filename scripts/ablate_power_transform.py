"""Should skewed columns get a power transform by default? Decided by evidence (Sprint 12).

    python scripts/ablate_power_transform.py      # needs bank_marketing.csv and adult.csv

RULE, WRITTEN BEFORE ANY MEASUREMENT (committed before the first run; never changed after):
  1. Compare "all off" (every skewed column only scaled) with "all on" (as shipped) by the
     paired fold difference of the C variant (gatekeeper cleaning + median imputer + Sprint 6
     transform; 5-fold stratified CV, seed 42), per model.
  2. The power transform STAYS ON (default unchanged) only if, on BOTH Bank and Adult,
     "all off - all on" for LogisticRegression ROC-AUC or PR-AUC is negative with a 95% CI
     that excludes 0, i.e. turning it off clearly hurts on both datasets.
  3. Otherwise the default for skewed columns becomes "scale only" (POWER["default"] = False),
     BUT only if trees are not hurt: for HistGradientBoosting on both datasets, "all off -
     all on" ROC-AUC and PR-AUC must not be negative with a CI excluding 0. If they are, the
     default stays ON.
  4. The one-column-at-a-time rows are reported to understand WHY; they decide nothing. No
     per-column rule was defined before measuring, so none is used.
Data: Bank as in Sprint 8 (no decisions); Adult with contracts/adult_decisions.yaml.
Step 0 logs the skewed columns (fold 0 train rows) BEFORE any model is trained.
Writes reports/power_ablation.json and prints the tables and the decision.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaning.imputers import FoldImputer  # noqa: E402
from src.impact import impact as imp  # noqa: E402
from src.profiling.decisions import load_decisions  # noqa: E402
from src.profiling.profiler import read_csv_raw  # noqa: E402
from src.transform import plan as tplan  # noqa: E402

DATASETS = {"bank": ("data/raw/bank_marketing.csv", "y", None),
            "adult": ("data/raw/adult.csv", "income", "contracts/adult_decisions.yaml")}
METRICS = ("roc_auc", "pr_auc")


def skewed_columns(df, target: str, config: dict | None) -> list[str]:  # noqa: ANN001
    """Columns that get a power transform in the C pipeline on fold 0's train rows."""
    y = df[target]
    X = df.drop(columns=[target])
    tr, _ = next(StratifiedKFold(5, shuffle=True, random_state=imp.SEED).split(X, y))
    cleaner = imp.RowCleaner(config=config).fit(X.iloc[tr])
    train = cleaner.transform(X.iloc[tr])
    filled = FoldImputer("median", cleaner.impute_targets_, cleaner.impute_features_) \
        .fit_transform(train)
    plan, _ = tplan.plan_columns(filled)
    return [p["column"] for p in plan if p["kind"] == "numeric" and p["power"]]


def set_power(default: bool, off: set[str] | None = None) -> None:
    """Switch power transforms (all, none, or all but some columns)."""
    tplan.POWER["default"] = default
    tplan.POWER["off"] = set(off or ())


def run_setting(df, target: str, config: dict | None, name: str) -> dict:  # noqa: ANN001
    """A vs C with the current POWER settings -> fold scores and C - A comparisons."""
    r = imp.run_impact(df, target, variants=("A", "C"), robustness=False,
                       dataset=name, cleaner_config=config)
    return {"C": {m: {k: r["results"]["C"][m][k]["folds"] for k in METRICS}
                  for m in imp.MODELS},
            "C-A": {m: {k: r["comparisons"]["C-A"][m][k] for k in METRICS}
                    for m in imp.MODELS}}


def decide(results: dict) -> tuple[bool, list[str]]:
    """Apply rule 2-3 to the 'all off - all on' comparisons -> (power default, reasons)."""
    why = []
    hurts_logreg = []
    for ds in DATASETS:
        off = results[ds]["settings"]["all off"]["vs_all_on"]["logreg"]
        bad = [k for k in METRICS if off[k]["ci95"][1] < 0]
        hurts_logreg.append(bool(bad))
        why.append(f"{ds}: turning it off hurts LogReg on {bad or 'no metric'} (CI below 0)")
    if all(hurts_logreg):
        return True, why + ["rule 2: off hurts LogReg on both -> power stays ON"]
    tree_hurt = []
    for ds in DATASETS:
        off = results[ds]["settings"]["all off"]["vs_all_on"]["hgb"]
        bad = [k for k in METRICS if off[k]["ci95"][1] < 0]
        if bad:
            tree_hurt.append(f"{ds} {bad}")
    if tree_hurt:
        return True, why + [f"rule 3: trees hurt by 'off' ({tree_hurt}) -> power stays ON"]
    return False, why + ["rule 3: not hurting LogReg on both, trees not hurt -> 'scale only'"]


def main() -> None:
    """Step 0 lists the skewed columns, then every setting runs; the rule decides."""
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    results: dict = {}
    frames = {}
    for ds, (path, target, decisions) in DATASETS.items():
        config = {"decisions": load_decisions(ROOT / decisions)} if decisions else None
        df = read_csv_raw(ROOT / path)
        set_power(True)
        cols = skewed_columns(df, target, config)
        frames[ds] = (df, target, config)
        results[ds] = {"skewed_columns": cols, "settings": {}}
        print(f"step 0 {ds}: skewed columns with a power transform (fold 0): {cols}", flush=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/power_ablation_columns.json").write_text(
        json.dumps({ds: r["skewed_columns"] for ds, r in results.items()}, indent=2) + "\n")

    try:
        for ds, (df, target, config) in frames.items():
            settings = [("all on", True, set()), ("all off", False, set())]
            settings += [(f"only {c} off", True, {c}) for c in results[ds]["skewed_columns"]]
            for name, default, off in settings:
                set_power(default, off)
                results[ds]["settings"][name] = run_setting(df, target, config, ds)
                print(ds, name, "done", flush=True)
            base = results[ds]["settings"]["all on"]["C"]
            for res in results[ds]["settings"].values():
                res["vs_all_on"] = {m: {k: imp.paired(res["C"][m][k], base[m][k])
                                        for k in METRICS} for m in imp.MODELS}
    finally:
        set_power(True)

    default, why = decide(results)
    out = {"rule": __doc__.split("RULE, ")[1].split("Data:")[0].strip(), "results": results,
           "decision": {"power_default": default, "why": why}}
    (ROOT / "reports/power_ablation.json").write_text(json.dumps(out, indent=2) + "\n")
    print_tables(out)


def _ci(c: dict) -> str:
    return f"{c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"


def print_tables(out: dict) -> None:
    """Markdown tables (C - A and setting - all on) + the decision."""
    for ds, r in out["results"].items():
        print(f"\n### {ds} (skewed: {', '.join(r['skewed_columns'])})")
        print("| setting | model | ROC-AUC C - A | PR-AUC C - A | ROC-AUC vs all on | "
              "PR-AUC vs all on |\n|---|---|---|---|---|---|")
        for name, res in r["settings"].items():
            for m in ("logreg", "hgb"):
                ca, vs = res["C-A"][m], res["vs_all_on"][m]
                vs_cells = ("-", "-") if name == "all on" else (_ci(vs["roc_auc"]),
                                                                 _ci(vs["pr_auc"]))
                print(f"| {name} | {m} | {_ci(ca['roc_auc'])} | {_ci(ca['pr_auc'])} | "
                      f"{vs_cells[0]} | {vs_cells[1]} |")
    print("\nDecision:", "power transform stays ON" if out["decision"]["power_default"]
          else "skewed columns are only SCALED by default")
    for w in out["decision"]["why"]:
        print(" -", w)


if __name__ == "__main__":
    if "--table" in sys.argv:
        print_tables(json.loads((ROOT / "reports/power_ablation.json").read_text()))
    else:
        main()
