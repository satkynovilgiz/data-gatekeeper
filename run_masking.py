"""Sprint 6 CLI — find personal data (PII) in a CSV and protect it.

Usage:
    export GATEKEEPER_PII_SALT=$(openssl rand -hex 32)     # secret for hashing; keep it safe
    python run_masking.py --input data/raw/customers_pii.csv --output results/customers
    python run_masking.py --input ... --action email=mask --action ssn=hash

Outputs in --output: masked_data.csv, pii_report.json (columns, type, confidence, action —
never a data value). The input file is only read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.privacy.mask import ACTIONS, PrivacyError, mask_dataframe
from src.profiling.profiler import read_csv_raw


def parse_actions(items: list[str]) -> dict[str, str]:
    """["email=mask"] -> {"email": "mask"}; raises ValueError for a bad item."""
    out = {}
    for item in items:
        column, sep, action = item.rpartition("=")
        if not sep or not column or action not in ACTIONS:
            raise ValueError(f"--action {item!r}: use COLUMN=ACTION with ACTION one of "
                             f"{sorted(ACTIONS)}")
        out[column] = action
    return out


def main() -> None:
    """Parse the command line, protect the PII columns, write the two outputs."""
    ap = argparse.ArgumentParser(description="Detect and mask PII in a CSV")
    ap.add_argument("--input", required=True, help="CSV file")
    ap.add_argument("--output", default="results/privacy", help="folder for the outputs")
    ap.add_argument("--action", action="append", default=[], metavar="COLUMN=ACTION",
                    help=f"action for one column: {', '.join(sorted(ACTIONS))} (repeatable)")
    args = ap.parse_args()
    if not Path(args.input).is_file():
        ap.error(f"input file not found: {args.input}")
    out_dir = Path(args.output)
    out_path = out_dir / "masked_data.csv"
    if out_path.resolve() == Path(args.input).resolve():
        ap.error("the output would overwrite the input file; choose another --output")
    try:
        masked, report = mask_dataframe(read_csv_raw(args.input), parse_actions(args.action))
    except (PrivacyError, ValueError) as e:
        ap.error(str(e))

    out_dir.mkdir(parents=True, exist_ok=True)
    masked.to_csv(out_path, index=False)
    (out_dir / "pii_report.json").write_text(json.dumps(report, indent=2))
    print(f"PII columns found: {report['pii_columns_found']}")
    for r in report["columns"]:
        print(f"  {r['column']:<16} {r['pii_type']:<15} confidence {r['confidence']:<5} "
              f"-> {r['action']}")
    print(f"masked_data -> {out_path}\npii_report  -> {out_dir / 'pii_report.json'}")


if __name__ == "__main__":
    main()
