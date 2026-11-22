"""Module 2 CLI — turn a CLEANED dataset into model-ready numbers (Sprint 6).

Usage:
    python run_transform.py --input results/bank/cleaned_data.csv --target y \
                            --output results/bank/transform
    python run_transform.py --input results/hr/cleaned_data.csv --output results/hr/transform \
                            --bins age=18,30,45,60,100 --scale salary=robust

Outputs in --output: train_transformed.csv, test_transformed.csv,
transform_pipeline.joblib (the fitted pipeline), transform_report.json (why each column
was encoded/scaled the way it was, skew before/after).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.transform.pipeline import SEED, TEST_SIZE, save_transform, transform_dataset
from src.transform.plan import SCALERS


def parse_pairs(items: list[str], flag: str) -> dict[str, str]:
    """["age=18,30,45"] -> {"age": "18,30,45"}; raises ValueError without '='."""
    out = {}
    for item in items:
        column, sep, value = item.partition("=")
        if not sep or not column or not value:
            raise ValueError(f"{flag} {item!r}: use COLUMN=VALUE")
        out[column] = value
    return out


def parse_bins(items: list[str]) -> dict[str, list[float]]:
    """["age=18,30,45"] -> {"age": [18.0, 30.0, 45.0]} (at least 2 rising edges)."""
    bins = {}
    for column, value in parse_pairs(items, "--bins").items():
        try:
            edges = [float(v) for v in value.split(",")]
        except ValueError as e:
            raise ValueError(f"--bins {column}: edges must be numbers") from e
        if len(edges) < 2 or edges != sorted(set(edges)):
            raise ValueError(f"--bins {column}: give at least 2 rising edges, e.g. 18,30,45")
        bins[column] = edges
    return bins


def main() -> None:
    """Parse the command line, transform the CSV, write the outputs, print a summary."""
    ap = argparse.ArgumentParser(description="Encode + scale a cleaned CSV (Module 2)")
    ap.add_argument("--input", required=True, help="the cleaned CSV")
    ap.add_argument("--output", default="results/transform", help="folder for the outputs")
    ap.add_argument("--target", default=None, help="label column, kept apart (e.g. y)")
    ap.add_argument("--test-size", type=float, default=TEST_SIZE, help="share of rows for test")
    ap.add_argument("--seed", type=int, default=SEED, help="random seed for the split")
    ap.add_argument("--bins", action="append", default=[], metavar="COLUMN=E1,E2,...",
                    help="add groups for a numeric column, e.g. age=18,30,45,60,100")
    ap.add_argument("--scale", action="append", default=[], metavar="COLUMN=SCALER",
                    help=f"force a scaler for one column: {', '.join(sorted(SCALERS))}")
    ap.add_argument("--drop", action="append", default=[], metavar="COLUMN",
                    help="leave a column out")
    args = ap.parse_args()
    if not Path(args.input).is_file():
        ap.error(f"input file not found: {args.input}")
    if not 0 < args.test_size < 1:
        ap.error("--test-size must be between 0 and 1")
    try:
        config = {"bins": parse_bins(args.bins), "scaling": parse_pairs(args.scale, "--scale"),
                  "drop": args.drop}
        result = transform_dataset(pd.read_csv(args.input), target=args.target,
                                   test_size=args.test_size, seed=args.seed, config=config)
    except ValueError as e:
        ap.error(str(e))
    files = save_transform(result, args.output)

    r = result.report
    print(f"Rows: {r['rows']['train']} train / {r['rows']['test']} test   "
          f"({r['fitted_on']})")
    print(f"Features: {r['features_in']} columns in -> {r['features_out']} model columns out; "
          f"{len(r['dropped'])} dropped")
    for p in r["columns"]:
        print(f"  {p['column']:<22} {p['kind']:<8} {p['reason']}")
    for d in r["dropped"]:
        print(f"  {d['column']:<22} dropped  {d['reason']}")
    for k, v in files.items():
        print(f"{k:<9} -> {v}")


if __name__ == "__main__":
    main()
