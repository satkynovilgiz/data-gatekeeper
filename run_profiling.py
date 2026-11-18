"""Module 1 CLI — profile a dataset.

Usage:
    python run_profiling.py --input data/raw/hr_employees_messy.csv --output results/
"""
import argparse
from pathlib import Path

from src.profiling.profiler import profile_dataset, read_csv_raw, save_report
from src.profiling.visualize import make_all


def main():
    ap = argparse.ArgumentParser(description="Profile a CSV dataset (Module 1)")
    ap.add_argument("--input", required=True, help="path to a CSV file")
    ap.add_argument("--output", default="results", help="folder for the report and charts")
    args = ap.parse_args()

    # Reads every value as text (so mixed types are not hidden) and
    # detects the separator automatically (, or ; or tab).
    df = read_csv_raw(args.input)

    report = profile_dataset(df, dataset_name=Path(args.input).name)
    numeric_cols = [c["name"] for c in report["columns"] if "numeric_stats" in c]
    report["charts"] = make_all(df, numeric_cols, Path(args.output) / "charts")
    path = save_report(report, args.output)

    d, s = report["dataset"], report["summary"]
    print(f"Profiled {d['rows']} rows x {d['columns']} columns")
    print(f"Missing cells: {d['missing_cells_pct']}%   Duplicate rows: {d['duplicate_rows']}")
    print(f"PII columns: {s['pii_columns']}")
    print(f"Suspicious columns: {s['suspicious_columns']}")
    print(f"Issues: {s['issue_counts']}")
    print(f"Report -> {path}")


if __name__ == "__main__":
    main()
