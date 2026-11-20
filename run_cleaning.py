"""Module 2 CLI — clean a dataset.

Usage:
    python run_cleaning.py --input data/raw/bank_marketing.csv \
                           --profile results/bank/profiling_report.json --output results/bank
    python run_cleaning.py --input data/raw/hr_employees_messy.csv --output results/hr
        (no --profile: Module 1 profiling runs first and is saved as profiling_report.json)
    python run_cleaning.py --input ... --impute default=none --impute age=mean
    python run_cleaning.py --input ... --imputer knn --outliers cap --fuzzy-threshold 90
    python run_cleaning.py --input ... --evaluate-imputers     (+ imputer_evaluation.json/.png)

Outputs in --output: cleaned_data.csv, cleaning_log.json, schema.json
The input file is only read, never changed.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.cleaning.cleaner import IMPUTE_STRATEGIES, OUTLIER_ACTIONS, clean_file
from src.cleaning.evaluation import EVAL_METHODS

log = logging.getLogger("cleaning")


def parse_impute(items: list[str]) -> dict[str, str]:
    """Turn ["age=mean", "default=none"] into {"age": "mean", "default": "none"}.

    Raises ValueError for an item without "=" or with an unknown strategy.
    """
    out = {}
    for item in items:
        column, sep, strategy = item.rpartition("=")
        if not sep or not column or strategy not in IMPUTE_STRATEGIES:
            raise ValueError(f"--impute {item!r}: use COLUMN=STRATEGY with STRATEGY one of "
                             f"{sorted(IMPUTE_STRATEGIES)}")
        out[column] = strategy
    return out


def main() -> None:
    """Parse the command line, clean the CSV, write the three outputs, print a summary."""
    ap = argparse.ArgumentParser(description="Clean a CSV dataset (Module 2)")
    ap.add_argument("--input", required=True, help="path to the raw CSV file")
    ap.add_argument("--profile", default=None,
                    help="Module 1 profiling_report.json of the same file "
                         "(if missing, profiling runs first)")
    ap.add_argument("--output", default="results", help="folder for the outputs")
    ap.add_argument("--impute", action="append", default=[], metavar="COLUMN=STRATEGY",
                    help="imputation for one column: median, mean, mode, knn, regression, "
                         "iterative or none (repeatable; overrides --imputer)")
    ap.add_argument("--imputer", choices=EVAL_METHODS, default=None,
                    help="imputer for all numeric and categorical columns "
                         "(median = median for numbers, mode for categories)")
    ap.add_argument("--outliers", choices=sorted(OUTLIER_ACTIONS - {"none"}), default=None,
                    help="what to do with IQR outliers: flag (default), cap or nan")
    ap.add_argument("--fuzzy-threshold", type=float, default=None, metavar="N",
                    help="name similarity 0-100 for fuzzy duplicates (default 85)")
    ap.add_argument("--fuzzy-merge", type=float, default=None, metavar="N",
                    help="merge fuzzy duplicates with a name similarity >= N (default: never)")
    ap.add_argument("--evaluate-imputers", action="store_true",
                    help="hide 10%% of known values, score every imputer, save a JSON + chart")
    args = ap.parse_args()
    try:
        impute = parse_impute(args.impute)
    except ValueError as e:
        ap.error(str(e))
    if not Path(args.input).is_file():
        ap.error(f"input file not found: {args.input}")
    if args.profile and not Path(args.profile).is_file():
        ap.error(f"profiling report not found: {args.profile}")
    for name in ("fuzzy_threshold", "fuzzy_merge"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 100:
            ap.error(f"--{name.replace('_', '-')} must be between 0 and 100")

    # our own messages at INFO; other libraries only when something is wrong
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    report = None
    if args.profile:
        report = json.loads(Path(args.profile).read_text())
        log.info("Using profiling report %s", args.profile)
    else:
        log.info("No --profile given: running Module 1 profiling first")

    config = {"impute": impute, "evaluate_imputers": args.evaluate_imputers}
    if args.imputer:
        config["numeric_strategy"] = config["categorical_strategy"] = args.imputer
    if args.outliers:
        config["outlier_action"] = args.outliers
    if args.fuzzy_threshold is not None:
        config["fuzzy_threshold"] = args.fuzzy_threshold
    if args.fuzzy_merge is not None:
        config["fuzzy_merge_threshold"] = args.fuzzy_merge
    result = clean_file(args.input, args.output, report=report, config=config)

    s, q = result["summary"], result["quality"]
    log.info("Cleaning took %.2f s (total with profiling %.2f s)",
             s["seconds"], s["total_seconds"])
    print(f"Rows: {s['rows_before']} -> {s['rows_after']}   "
          f"Columns: {s['columns_before']} -> {s['columns_after']} (added {s['columns_added']})")
    print(f"Duplicates removed: {s['duplicates_removed']}   "
          f"Disguised missing -> NaN: {s['disguised_missing_converted']}   "
          f"Special codes -> flag: {s['special_codes_to_flag']}")
    print(f"Not a number: {s['not_a_number_to_nan']}   Dates reformatted: "
          f"{s['dates_reformatted']}   Invalid format: {s['invalid_format_to_nan']}   "
          f"Impossible values: {s['impossible_values_fixed']}   "
          f"Imputed: {s['values_imputed']}")
    o, f = s["outliers"], s["fuzzy_duplicates"]
    print(f"Outliers: IQR {o['iqr_cells']} cells in {o['iqr_rows']} rows (flagged "
          f"{o['flagged']}, capped {o['capped']}, to NaN {o['set_to_nan']})   "
          f"Isolation Forest {o['isolation_forest_rows']} rows, both {o['rows_found_by_both']}")
    print("Fuzzy duplicates: " + (f["skipped"] if "skipped" in f else
          f"{f['pairs_found']} pairs, {f['rows_flagged']} rows flagged, "
          f"{f['rows_merged']} merged"))
    if s["kept_as_category"]:
        print(f"'unknown' kept as a category: {s['kept_as_category']}")
    ev = result.get("imputer_evaluation")
    if ev and "methods" in ev:
        for m, r in ev["methods"].items():
            print(f"  imputer {m:10} NRMSE {r['mean_nrmse']}  accuracy {r['mean_accuracy']}  "
                  f"{r['seconds']} s")
        print(f"  best: {ev['best']}  -> {ev['files'].get('chart')}")
    print(f"Quality score: {q['before']['overall']} -> {q['after']['overall']} "
          f"({q['delta']['overall']:+})   delta per dimension: "
          f"{ {k: v for k, v in q['delta'].items() if k != 'overall'} }")
    for name, path in result["outputs"].items():
        print(f"{name:13} -> {path}")


if __name__ == "__main__":
    main()
