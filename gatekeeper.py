"""gatekeeper — data contracts, drift, anomalies, impact and the pipeline (Sprints 7-9).

    python gatekeeper.py contract init <reference.csv> [--name NAME] [--out FILE] [--force]
    python gatekeeper.py validate <data.csv> --contract contracts/NAME.yaml
                         [--reference REF.csv] [--anomalies] [--fail-on-anomaly-share X]
    python gatekeeper.py anomalies <data.csv> [--method iforest|zscore|iqr|all] [--top 10]
    python gatekeeper.py decisions <data.csv> [--target y] [--write decisions.yaml]
    python gatekeeper.py drift <reference.csv> <current.csv> [--contract FILE]
    python gatekeeper.py impact <data.csv> --target y [--folds 5] [--no-robustness]
    python gatekeeper.py run --config pipeline.yaml [--skip impact]

Exit codes (for CI):
    validate  0 = every error rule passed (warnings allowed)
              1 = at least one error rule failed (or, with --fail-on-anomaly-share X,
                  more than X of the rows were flagged as anomalies)
              2 = bad contract or bad input (missing / unreadable file, invalid YAML ...)
    anomalies 0 = report written (1 only with --fail-on-anomaly-share), 2 = bad input
    decisions 0 = table printed (and draft written), 2 = bad input
    drift     0 = no column with major drift, 1 = at least one, 2 = bad input
    contract init: 0 = written, 2 = bad input or the file exists (use --force)
    impact    0 = report written, 2 = bad input
    run       0 = every step passed, 1 = stopped at a gate (contract error / anomaly share
              over fail_on_share / major drift with gate: true), 2 = bad config or a step crashed

Reports are written to reports/ (change with --reports-dir):
    reports/<dataset>_validation_report.json (contract + drift + anomalies, Sprint 9;
    docs/validation_report.schema.json), reports/<dataset>_contract.json (the contract part,
    as in Sprint 7), reports/<current>_drift.json, reports/<dataset>_anomalies.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.profiling.profiler import read_csv_raw
from src.validation.contract import (
    ContractError,
    contract_to_yaml,
    draft_contract,
    load_contract,
)
from src.validation.drift import drift_report

DEFAULT_LEVELS = "0,0.05,0.10,0.20"

EXIT_OK, EXIT_FAIL, EXIT_BAD_INPUT = 0, 1, 2
CELL = 48           # widest table cell; longer text is cut with "…" (the JSON has it all)


class InputError(ValueError):
    """A data file is missing or cannot be read as CSV."""


def read_data(path: str) -> pd.DataFrame:
    """Read a CSV as text, turning every read problem into an InputError."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"data file not found: {p}")
    try:
        return read_csv_raw(p)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise InputError(f"{p}: cannot read it as CSV ({e.__class__.__name__})") from None


def table(rows: list[list[str]], header: list[str]) -> str:
    """A plain fixed-width text table."""
    def cut(v: str) -> str:
        return v if len(v) <= CELL else v[:CELL - 1] + "…"
    rows = [[cut(str(v)) for v in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), line]
    out += ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def write_json(report: dict, reports_dir: str, filename: str) -> Path:
    """Save a report as pretty JSON and return its path."""
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    return path


def cmd_init(args: argparse.Namespace) -> int:
    """`contract init`: draft a contract YAML from a reference CSV."""
    df = read_data(args.csv)
    name = args.name or Path(args.csv).stem
    out = Path(args.out or f"contracts/{name}.yaml")
    if out.exists() and not args.force:
        raise InputError(f"{out} already exists (use --force to overwrite)")
    contract = draft_contract(df, name, source=Path(args.csv).name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(contract_to_yaml(contract, Path(args.csv).name), encoding="utf-8")
    load_contract(out)                    # the draft must pass our own format check
    print(f"Drafted contract for {len(contract['columns'])} columns, {len(df)} rows -> {out}")
    print("Review and edit it before using it as a gate.")
    return EXIT_OK


def _val(v: object) -> str:
    """A number as %g, a category as itself."""
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _anomaly_lines(section: dict, method: str, rows: int = 3) -> list[str]:
    """A few printed lines about one method's top rows (row position + reasons)."""
    m = section["methods"][method]
    out = [f"{method}: {m['flagged']} of {section['rows']} rows flagged "
           f"({m['flagged_share']:.2%}), fitted on {section['fitted_on']}"]
    for t in m["top"][:rows]:
        why = "; ".join(f"{r['column']}={_val(r['value'])} (typical {_val(r['median'])}, z "
                        f"{'-' if r['robust_z'] is None else r['robust_z']})"
                        for r in t["reasons"] if r["value"] is not None)
        out.append(f"  #{t['rank']} row {t['row']} score {t['score']:.3f}: {why}")
    return out


def cmd_validate(args: argparse.Namespace) -> int:
    """`validate`: contract (+ drift, + anomalies) -> one validation report; exit 1 on fail."""
    from src.validation.report import parse_methods, validation_report

    contract = load_contract(args.contract)
    df = read_data(args.csv)
    reference = read_data(args.reference) if args.reference else None
    name = Path(args.csv).stem
    try:
        methods = parse_methods(args.anomaly_method)
        report = validation_report(
            df, contract, name, reference=reference,
            reference_name=Path(args.reference).name if args.reference else "reference",
            anomalies=args.anomalies, methods=methods, contamination=args.contamination,
            exclude=args.exclude, fail_on_share=args.fail_on_anomaly_share)
    except ValueError as e:
        raise InputError(str(e)) from None
    part = report["contract"]
    part["contract"] = str(args.contract)

    order = {"fail": 0, "warn": 1, "skipped": 2, "pass": 3}
    shown = sorted(part["results"], key=lambda r: order[r["status"]])
    if args.failures_only:
        shown = [r for r in shown if r["status"] in ("fail", "warn")]
    print(table([[r["rule"], r["column"] or "(table)", r["severity"], r["status"].upper(),
                  r["observed"], r["expected"]] for r in shown],
                ["rule", "column", "severity", "status", "observed", "expected"]))
    c = part["counts"]
    write_json(part, args.reports_dir, f"{name}_contract.json")
    path = write_json(report, args.reports_dir, f"{name}_validation_report.json")
    print(f"\ncontract  {part['status'].upper()}: {c['passed']} passed, {c['failed_error']} "
          f"failed (error), {c['failed_warn']} warnings, {c['skipped']} skipped of "
          f"{c['rules']} rules  |  {part['rows']} rows")
    if report["drift"]:
        d = report["drift"]
        print(f"drift     {d['status'].upper()}: {d['counts']['major']} major, "
              f"{d['counts']['moderate']} moderate, {d['counts']['stable']} stable"
              + (f" ({', '.join(d['major_columns'])})" if d["major_columns"] else ""))
    if report["anomalies"]:
        gate = report["gates"]["anomalies"]
        lines = _anomaly_lines(report["anomalies"], gate["method"])
        limit = "" if gate["limit"] is None else f" (limit {gate['limit']:.2%}: " \
                f"{gate['status'].upper()})"
        print("anomalies " + lines[0] + limit)
        print("\n".join(lines[1:]))
    print(f"\n{report['status'].upper()}  |  Report -> {path}")
    return EXIT_FAIL if report["status"] == "fail" else EXIT_OK


def cmd_anomalies(args: argparse.Namespace) -> int:
    """`anomalies`: score rows, explain the top ones; never fails unless a limit is set."""
    from src.validation.anomaly import detect
    from src.validation.report import parse_methods

    df = read_data(args.csv)
    reference = read_data(args.reference) if args.reference else None
    try:
        methods = parse_methods(args.method)
        if args.fail_on_anomaly_share is not None and not 0 <= args.fail_on_anomaly_share <= 1:
            raise ValueError("--fail-on-anomaly-share must be between 0 and 1")
        section = detect(df, methods=methods, top=args.top, contamination=args.contamination,
                         reference=reference, exclude=args.exclude, encoding=args.encoding)
    except ValueError as e:
        raise InputError(str(e)) from None
    name = Path(args.csv).stem
    path = write_json(section, args.reports_dir, f"{name}_anomalies.json")
    print(f"features: {', '.join(section['features'])}")
    if section["categorical_features"]:
        print(f"categorical (iforest_with_categories only, {args.encoding} encoding): "
              f"{', '.join(section['categorical_features'])}")
    for m in methods:
        print()
        print("\n".join(_anomaly_lines(section, m, rows=args.top)))
    share = section["methods"][methods[0]]["flagged_share"]
    failed = args.fail_on_anomaly_share is not None and share > args.fail_on_anomaly_share
    print(f"\nReport -> {path}")
    return EXIT_FAIL if failed else EXIT_OK


def _fmt(v: float | None, digits: int = 4) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def cmd_drift(args: argparse.Namespace) -> int:
    """`drift`: PSI + KS per column; exit 1 when any column drifted majorly."""
    contract = load_contract(args.contract) if args.contract else None
    ref, cur = read_data(args.reference), read_data(args.current)
    name = Path(args.current).stem
    report = drift_report(ref, cur, contract, reference_name=Path(args.reference).name,
                          current_name=Path(args.current).name)
    rows = []
    for e in report["columns"]:
        note = e.get("note", "")
        if e.get("new_categories_count"):
            shown = ", ".join(e["new_categories"][:3])
            note = f"{e['new_categories_count']} new: {shown}"
        elif e.get("cur_missing_pct", 0) != e.get("ref_missing_pct", 0):
            note = f"missing {e['ref_missing_pct']}% -> {e['cur_missing_pct']}%"
        if e["pii"]:
            note = ("PII, values hidden; " + note).rstrip("; ")
        rows.append([e["column"], e.get("kind") or "-", _fmt(e["psi"]),
                     _fmt(e.get("ks_statistic")), e["status"].upper(), note])
    print(table(rows, ["column", "kind", "PSI", "KS", "status", "note"]))
    t, c = report["thresholds"], report["counts"]
    path = write_json(report, args.reports_dir, f"{name}_drift.json")
    print(f"\n{report['status'].upper()}: {c['major']} major, {c['moderate']} moderate, "
          f"{c['stable']} stable, {c['skipped']} skipped  |  PSI < {t['psi_moderate']} stable, "
          f"> {t['psi_major']} major  |  {report['reference']['rows']} -> "
          f"{report['current']['rows']} rows")
    print(f"Report -> {path}")
    return EXIT_FAIL if report["counts"]["major"] else EXIT_OK


def stratified_sample(df: pd.DataFrame, target: str, n: int, seed: int = 42) -> pd.DataFrame:
    """n rows with the same class shares as the full file (for quick runs)."""
    if n >= len(df):
        return df
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(df, train_size=n, stratify=df[target], random_state=seed)
    return keep.sort_index()


def parse_levels(text: str) -> tuple[float, ...]:
    """'0,0.05,0.1' -> (0.0, 0.05, 0.1), each between 0 and 1."""
    try:
        levels = tuple(float(x) for x in text.split(","))
    except ValueError:
        raise InputError(f"--levels {text!r}: use numbers like 0,0.05,0.1") from None
    if not levels or any(not 0 <= lv <= 1 for lv in levels):
        raise InputError("--levels: every level must be between 0 and 1")
    return levels if 0.0 in levels else (0.0, *levels)


def cmd_impact(args: argparse.Namespace) -> int:
    """`impact`: does gatekeeper cleaning help a model? A/B/C with 5-fold CV."""
    df = read_data(args.csv)
    if args.target not in df.columns:
        raise InputError(f"target column {args.target!r} not found")
    if args.folds < 2:
        raise InputError("--folds must be at least 2")
    name = Path(args.csv).stem
    if args.sample:
        df = stratified_sample(df[df[args.target].notna()], args.target, args.sample)
        name += f"_sample{args.sample}"
    from src.impact.impact import run_impact
    from src.impact.report import comparison_table, results_table, write_impact_report

    try:
        variants = tuple(v.strip().upper() for v in args.variants.split(",") if v.strip())
        if not variants or set(variants) - {"A", "B", "C"}:
            raise InputError("--variants: use a comma list of A, B, C")
        config = None
        if args.decisions:
            from src.profiling.decisions import DecisionError, load_decisions
            try:
                config = {"decisions": load_decisions(args.decisions)}
            except DecisionError as e:
                raise InputError(str(e)) from None
        report = run_impact(df, args.target, positive=args.positive, folds=args.folds,
                            robustness=not args.no_robustness, variants=variants,
                            levels=parse_levels(args.levels), dataset=name,
                            cleaner_config=config)
    except ValueError as e:
        raise InputError(str(e)) from None
    write_impact_report(report, args.reports_dir, name)

    print(results_table(report))
    print()
    print(comparison_table(report))
    print(f"\nRun time {report['seconds']} s")
    for kind, path in report["files"].items():
        print(f"{kind:9}-> {path}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """`run`: every configured step in order; stops at a failed gate (exit 1)."""
    from src.pipeline.run import ConfigError, load_config, run_pipeline

    try:
        cfg = load_config(args.config)
        code, summary, out = run_pipeline(cfg, skip=set(args.skip))
    except ConfigError as e:
        raise InputError(str(e)) from None
    for s in summary["steps"]:
        detail = s.get("error") or s.get("reason") or ""
        print(f"{s['step']:9} {s['status'].upper():8} {s.get('seconds', '-'):>7}  {detail}")
    print(f"\n{summary['status'].upper()} (exit {code}) in {summary['total_seconds']} s")
    print(f"Summary -> {out / 'summary.md'}")
    return code


def _anomaly_options(p: argparse.ArgumentParser) -> None:
    """Options shared by `validate --anomalies` and `anomalies`."""
    p.add_argument("--contamination", type=float, default=0.01,
                   help="Isolation Forest share used for its flag threshold (default 0.01)")
    p.add_argument("--exclude", action="append", default=[], metavar="COLUMN",
                   help="never use this column (e.g. the label y); repeatable")
    p.add_argument("--fail-on-anomaly-share", type=float, metavar="X",
                   help="exit 1 when more than this share of rows is flagged (e.g. 0.05)")


def cmd_decisions(args: argparse.Namespace) -> int:
    """`decisions`: values the gatekeeper is not sure about, with evidence; draft YAML."""
    from src.profiling.decisions import decisions_yaml, find_decisions
    from src.profiling.profiler import profile_dataset

    df = read_data(args.csv)
    try:
        found = find_decisions(df, profile_dataset(df, Path(args.csv).name), target=args.target,
                               positive=args.positive)
    except ValueError as e:
        raise InputError(str(e)) from None

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{v:.2%}"
    print(table([[d["column"], d["value"], d["kind"], str(d["rows"]), pct(d["label_rate_in"]),
                  pct(d["label_rate_rest"]), "-" if d["p_value"] is None else f"{d['p_value']:.1e}",
                  d["current_default"], d["recommended"] or "a person decides"] for d in found],
                ["column", "value", "kind", "rows", "label in", "label rest", "p", "today",
                 "recommended"]))
    if not found:
        print("No decisions needed: every code the profiler found is handled by its default.")
    if args.target is None:
        print("\nNo --target: counts only. With a target the label rates show which codes matter.")
    if args.write:
        out = Path(args.write)
        if out.exists() and not args.force:
            raise InputError(f"{out} already exists (use --force to overwrite)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(decisions_yaml(found, Path(args.csv).name), encoding="utf-8")
        print(f"\nDraft decisions -> {out} (a person must confirm them)")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser with the contract / validate / drift sub-commands."""
    p = argparse.ArgumentParser(prog="gatekeeper", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract", help="work with data contracts")
    csub = contract.add_subparsers(dest="contract_command", required=True)
    init = csub.add_parser("init", help="draft a contract from a reference CSV")
    init.add_argument("csv", help="reference dataset (CSV)")
    init.add_argument("--name", help="contract name (default: the CSV file name)")
    init.add_argument("--out", help="output YAML (default: contracts/<name>.yaml)")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(func=cmd_init)

    val = sub.add_parser("validate", help="check a CSV against a contract")
    val.add_argument("csv", help="dataset to check (CSV)")
    val.add_argument("--contract", required=True, help="contract YAML file")
    val.add_argument("--reports-dir", default="reports", help="where the JSON report goes")
    val.add_argument("--failures-only", action="store_true",
                     help="print only failed and warned rules (the JSON has all)")
    val.add_argument("--reference", help="reference CSV: adds the drift section (and the "
                     "anomaly models are fitted on it)")
    val.add_argument("--anomalies", action="store_true", help="add the anomaly section")
    val.add_argument("--anomaly-method", default="iforest",
                     help="iforest, zscore, iqr, a comma list or all (first one = the gate's)")
    _anomaly_options(val)
    val.set_defaults(func=cmd_validate)

    ano = sub.add_parser("anomalies", help="find and explain unusual rows")
    ano.add_argument("csv", help="dataset (CSV)")
    ano.add_argument("--method", default="all",
                     help="iforest, zscore, iqr, zscore_mixture, iforest_with_categories, "
                          "a comma list or all (default)")
    ano.add_argument("--encoding", default="frequency", choices=["frequency", "onehot"],
                     help="how iforest_with_categories turns categories into numbers")
    ano.add_argument("--top", type=int, default=10, help="rows to explain per method")
    ano.add_argument("--reference", help="fit on this CSV and judge the dataset against it")
    ano.add_argument("--reports-dir", default="reports", help="where the JSON report goes")
    _anomaly_options(ano)
    ano.set_defaults(func=cmd_anomalies)

    drift = sub.add_parser("drift", help="measure drift of a CSV against a reference CSV")
    drift.add_argument("reference", help="reference dataset (CSV)")
    drift.add_argument("current", help="new dataset (CSV)")
    drift.add_argument("--contract", help="contract YAML (drift thresholds, column dtypes)")
    drift.add_argument("--reports-dir", default="reports", help="where the JSON report goes")
    drift.set_defaults(func=cmd_drift)

    imp = sub.add_parser("impact", help="does cleaning help a model? (A raw / B / C, CV)")
    imp.add_argument("csv", help="raw dataset (CSV)")
    imp.add_argument("--target", required=True, help="label column, e.g. y")
    imp.add_argument("--positive", help="positive class (default: yes / 1 / the rarest)")
    imp.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    imp.add_argument("--no-robustness", action="store_true",
                     help="skip the corrupted-test-data part (faster)")
    imp.add_argument("--levels", default=DEFAULT_LEVELS,
                     help=f"corruption levels for robustness (default {DEFAULT_LEVELS})")
    imp.add_argument("--sample", type=int, help="use N stratified rows (quick runs)")
    imp.add_argument("--decisions", help="YAML with a decisions: block for the cleaner (B, C)")
    imp.add_argument("--variants", default="A,B,C",
                     help="which preparations to compare (default A,B,C; robustness uses A,B)")
    imp.add_argument("--reports-dir", default="reports", help="where the reports go")
    imp.set_defaults(func=cmd_impact)

    dec = sub.add_parser("decisions", help="values the gatekeeper is not sure about")
    dec.add_argument("csv", help="dataset (CSV)")
    dec.add_argument("--target", help="label column: adds label rates as evidence")
    dec.add_argument("--positive", help="positive class (default: yes / 1 / the rarest)")
    dec.add_argument("--write", metavar="FILE", help="write a draft decisions: block (YAML)")
    dec.add_argument("--force", action="store_true", help="overwrite --write FILE")
    dec.set_defaults(func=cmd_decisions)

    run = sub.add_parser("run", help="run the steps of a pipeline YAML in order")
    run.add_argument("--config", required=True, help="pipeline YAML, e.g. pipeline.yaml")
    run.add_argument("--skip", action="append", default=[], metavar="STEP",
                     help="skip a step, e.g. --skip impact (repeatable)")
    run.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ContractError, InputError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except OSError as e:
        # e.g. reports/ not writable inside a container: not a data failure, so not exit 1
        print(f"error: cannot read or write a file ({e.strerror}: {e.filename})",
              file=sys.stderr)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())
