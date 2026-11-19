"""Module 1 CLI — profile a dataset.

Usage:
    python run_profiling.py --input data/raw/bank_marketing.csv --output results/bank
    python run_profiling.py --input big.csv --output results/big --sample 10000   # random 10k rows
"""
import argparse
import logging
import time
from pathlib import Path

from src.profiling.profiler import profile_dataset, read_csv_raw, sample_rows, save_report
from src.profiling.visualize import make_all

log = logging.getLogger("profiling")


def main():
    ap = argparse.ArgumentParser(description="Profile a CSV dataset (Module 1)")
    ap.add_argument("--input", required=True, help="path to a CSV file")
    ap.add_argument("--output", default="results", help="folder for the report and charts")
    ap.add_argument("--sample", type=int, default=None, metavar="N",
                    help="profile only N random rows (faster for big files; seed 42)")
    args = ap.parse_args()
    if args.sample is not None and args.sample < 1:
        ap.error("--sample must be at least 1")
    # our own messages at INFO; other libraries (matplotlib) only when something is wrong
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    t0 = time.perf_counter()

    # Reads every value as text (so mixed types are not hidden) and
    # detects the separator automatically (, or ; or tab).
    df = read_csv_raw(args.input)
    log.info("Read %d rows in %.2f s", len(df), time.perf_counter() - t0)
    rows_in_file = len(df)
    df = sample_rows(df, args.sample)
    if len(df) < rows_in_file:
        log.info("Profiling a random sample of %d of %d rows", len(df), rows_in_file)

    report = profile_dataset(df, dataset_name=Path(args.input).name, rows_in_file=rows_in_file)
    log.info("Profiling took %.2f s", report["run"]["seconds"])

    t_charts = time.perf_counter()
    report["charts"] = make_all(df, report, Path(args.output) / "charts")
    log.info("Charts took %.2f s", time.perf_counter() - t_charts)
    path = save_report(report, args.output)
    log.info("Total time %.2f s", time.perf_counter() - t0)

    d, s, q = report["dataset"], report["summary"], report["quality_score"]
    print(f"Profiled {d['rows']} rows x {d['columns']} columns")
    print(f"Missing cells: {d['missing_cells_pct']}%   "
          f"True missing (incl. disguised): {d['true_missing_cells_pct']}%   "
          f"Duplicate rows: {d['duplicate_rows']}")
    print(f"PII columns: {s['pii_columns']}")
    print(f"Suspicious columns: {s['suspicious_columns']}")
    print(f"Issues: {s['issue_counts']}   Cleaning suggestions: {len(report['cleaning_suggestions'])}")
    print(f"Quality score: {q['overall']} / 100   {q['dimensions']}")
    print(f"Report -> {path}")


if __name__ == "__main__":
    main()
