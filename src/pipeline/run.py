"""One command for the whole flow: `gatekeeper run --config pipeline.yaml` (Sprint 8).

The config lists the steps to run, in this order (any subset, never reordered):

    profile -> validate -> clean -> mask_pii -> quality -> anomalies -> drift -> impact

    input: data/raw/bank_marketing.csv
    name: bank_marketing              # optional, default: the input file name
    reports_dir: reports              # optional
    steps:
      - profile
      - validate: {contract: contracts/bank_marketing.yaml}
      - clean: {imputer: iterative}   # optional: median, knn, regression, iterative
      - mask_pii                      # optional: {actions: {column: mask|hash|drop|...}}
      - quality
      - anomalies: {exclude: [y], top: 10}   # optional: method, contamination, fail_on_share
      - drift: {reference: data/demo/bank_reference.csv, gate: false}
      - impact: {target: y, folds: 5, robustness: true}

Every step calls the function the single commands use (profile_dataset, validate_dataframe,
clean_file, mask_dataframe, quality_score, detect, drift_report, run_impact); nothing is
re-implemented here. Paths are relative to the folder you run the command from.

Gates: validate stops the run when an error-level rule fails (exit 1). anomalies stops it
only with `fail_on_share: X` and more than X of the rows flagged; drift only with
`gate: true` and a column with major drift (exit 1). A bad config or a step that
crashes ends the run with exit 2. The summary is written in every case:
reports/run_<UTC timestamp>/summary.json and summary.md, next to each step's own files.
"""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.cleaning.cleaner import IMPUTE_STRATEGIES
from src.privacy.mask import ACTIONS, mask_dataframe
from src.profiling.profiler import profile_dataset, read_csv_raw, save_report
from src.validation.contract import load_contract, validate_dataframe
from src.validation.drift import drift_report

ORDER = ("profile", "validate", "clean", "mask_pii", "quality", "anomalies", "drift", "impact")
OPTIONS = {
    "profile": set(),
    "validate": {"contract"},
    "clean": {"imputer", "decisions"},
    "mask_pii": {"actions"},
    "quality": set(),
    "anomalies": {"method", "top", "contamination", "exclude", "fail_on_share"},
    "drift": {"reference", "contract", "gate"},
    "impact": {"target", "folds", "robustness", "positive", "decisions"},
}
REQUIRED = {"validate": {"contract"}, "drift": {"reference"}, "impact": {"target"}}
TOP_KEYS = {"input", "name", "reports_dir", "steps"}
EXIT_OK, EXIT_STOPPED, EXIT_ERROR = 0, 1, 2


class ConfigError(ValueError):
    """The pipeline config is missing, not YAML, or breaks the format."""


class StepStop(Exception):
    """A gate failed: the run stops here (exit 1)."""


# ---------------------------------------------------------------- config
def parse_config(raw: Any, source: str = "config") -> dict:
    """Check a loaded YAML config -> {input, name, reports_dir, steps: [(name, options)]}."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: the top level must be a mapping (key: value)")
    unknown = set(raw) - TOP_KEYS
    if unknown:
        raise ConfigError(f"{source}: unknown key(s) {sorted(unknown)}; allowed: "
                          f"{sorted(TOP_KEYS)}")
    if not isinstance(raw.get("input"), str) or not raw["input"]:
        raise ConfigError(f"{source}: `input` (the CSV to process) is required")
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ConfigError(f"{source}: `steps` must be a non-empty list")

    steps: list[tuple[str, dict]] = []
    for i, item in enumerate(steps_raw):
        if isinstance(item, str):
            name, opts = item, {}
        elif isinstance(item, dict) and len(item) == 1:
            name, opts = next(iter(item.items()))
            opts = opts or {}
        else:
            raise ConfigError(f"steps[{i}]: write a step name or `name: {{options}}`")
        if name not in ORDER:
            raise ConfigError(f"steps[{i}]: unknown step {name!r}; steps are {', '.join(ORDER)}")
        if not isinstance(opts, dict):
            raise ConfigError(f"steps[{i}] ({name}): options must be a mapping")
        bad = set(opts) - OPTIONS[name] - {"enabled"}
        if bad:
            raise ConfigError(f"steps[{i}] ({name}): unknown option(s) {sorted(bad)}; "
                              f"allowed: {sorted(OPTIONS[name] | {'enabled'})}")
        missing = REQUIRED.get(name, set()) - set(opts)
        if missing:
            raise ConfigError(f"steps[{i}] ({name}): needs {sorted(missing)}")
        steps.append((name, opts))

    names = [n for n, _ in steps]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigError(f"steps listed twice: {dupes}")
    positions = [ORDER.index(n) for n in names]
    if positions != sorted(positions):
        raise ConfigError("steps must keep this order: " + " -> ".join(ORDER) +
                          f" (got {' -> '.join(names)})")
    _check_options(steps)
    return {"input": raw["input"], "name": raw.get("name") or Path(raw["input"]).stem,
            "reports_dir": raw.get("reports_dir", "reports"), "steps": steps}


def _check_options(steps: list[tuple[str, dict]]) -> None:
    for name, o in steps:
        if "enabled" in o and not isinstance(o["enabled"], bool):
            raise ConfigError(f"{name}.enabled must be true or false")
        if name == "clean" and o.get("imputer", "iterative") not in IMPUTE_STRATEGIES - {"none"}:
            raise ConfigError(f"clean.imputer must be one of "
                              f"{sorted(IMPUTE_STRATEGIES - {'none'})}")
        if name == "mask_pii":
            actions = o.get("actions") or {}
            if not isinstance(actions, dict) or set(actions.values()) - ACTIONS:
                raise ConfigError(f"mask_pii.actions: {{column: action}} with action in "
                                  f"{sorted(ACTIONS)}")
        if name == "anomalies":
            if o.get("method", "iforest") not in ("iforest", "zscore", "iqr"):
                raise ConfigError("anomalies.method must be iforest, zscore or iqr")
            top = o.get("top", 10)
            if not isinstance(top, int) or isinstance(top, bool) or top < 1:
                raise ConfigError("anomalies.top must be a whole number >= 1")
            share = o.get("fail_on_share")
            if share is not None and (not isinstance(share, (int, float)) or
                                      isinstance(share, bool) or not 0 <= share <= 1):
                raise ConfigError("anomalies.fail_on_share must be a number from 0 to 1")
            if not isinstance(o.get("exclude", []), list):
                raise ConfigError("anomalies.exclude must be a list of column names")
        if name == "drift" and not isinstance(o.get("gate", False), bool):
            raise ConfigError("drift.gate must be true or false")
        if name == "impact":
            folds = o.get("folds", 5)
            if not isinstance(folds, int) or isinstance(folds, bool) or folds < 2:
                raise ConfigError("impact.folds must be a whole number >= 2")
            if not isinstance(o.get("robustness", True), bool):
                raise ConfigError("impact.robustness must be true or false")


def load_config(path: str | Path) -> dict:
    """Read and check a pipeline YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML ({e})") from None
    return parse_config(raw, source=str(path))


# ----------------------------------------------------------------- steps
class Context:
    """What the steps share: the data, earlier results and the run folder."""

    def __init__(self, cfg: dict, out: Path) -> None:
        self.cfg = cfg
        self.out = out
        self.raw = read_csv_raw(cfg["input"])
        self.profile: dict | None = None
        self.clean_log: dict | None = None
        self.cleaned_path: Path | None = None

    def write(self, filename: str, data: dict) -> str:
        """Save a step's JSON result in the run folder and return its path."""
        path = self.out / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
        return str(path)


def step_profile(ctx: Context, opts: dict) -> dict:
    """Module 1 profile of the input."""
    ctx.profile = profile_dataset(ctx.raw, dataset_name=Path(ctx.cfg["input"]).name)
    path = save_report(ctx.profile, ctx.out)
    d = ctx.profile["dataset"]
    return {"status": "passed", "files": [str(path)],
            "result": {"rows": d["rows"], "columns": d["columns"],
                       "duplicate_rows": d["duplicate_rows"],
                       "issues": len(ctx.profile["issues"]),
                       "quality_score": ctx.profile["quality_score"]["overall"]}}


def step_validate(ctx: Context, opts: dict) -> dict:
    """Sprint 7 contract gate: any error-level failure stops the run."""
    contract = load_contract(opts["contract"])
    report = validate_dataframe(ctx.raw, contract, dataset=ctx.cfg["name"])
    report["contract"] = str(opts["contract"])
    path = ctx.write(f"{ctx.cfg['name']}_contract.json", report)
    c = report["counts"]
    out = {"files": [path], "result": {
        "contract": str(opts["contract"]), "rules": c["rules"], "passed": c["passed"],
        "failed_error": c["failed_error"], "failed_warn": c["failed_warn"],
        "failed_rules": [f"{r['rule']} {r['column'] or '(table)'}: {r['observed']}"
                         for r in report["results"] if r["status"] == "fail"]}}
    if report["status"] == "fail":
        raise StepStop({**out, "status": "failed"})
    return {**out, "status": "warn" if c["failed_warn"] else "passed"}


def step_clean(ctx: Context, opts: dict) -> dict:
    """Module 2 cleaning (clean_file), reusing the profile when it ran."""
    from src.cleaning.cleaner import clean_file

    config = {}
    if opts.get("decisions"):                      # Sprint 11: a YAML with a decisions: block
        from src.profiling.decisions import load_decisions
        config["decisions"] = load_decisions(opts["decisions"])
    if "imputer" in opts:
        method = opts["imputer"]
        config = {"numeric_strategy": method,
                  "categorical_strategy": "mode" if method in ("median", "mean") else method}
    log = clean_file(ctx.cfg["input"], ctx.out / "clean", report=ctx.profile, config=config)
    ctx.clean_log = log
    ctx.cleaned_path = Path(log["outputs"]["cleaned_data"])
    s = log["summary"]
    chosen = [f"{d['column']}={d['value']}: {d['action']}" for d in log.get("decisions", [])
              if d["set_by"] == "config"]
    return {"status": "passed", "files": list(log["outputs"].values()),
            "result": {"rows_before": s["rows_before"], "rows_after": s["rows_after"],
                       "decisions_from_config": chosen,
                       "duplicates_removed": s["duplicates_removed"],
                       "disguised_missing_converted": s["disguised_missing_converted"],
                       "values_imputed": s["values_imputed"],
                       "columns_added": s["columns_added"]}}


def step_mask_pii(ctx: Context, opts: dict) -> dict:
    """Sprint 6 PII detection + protection on the cleaned data (or the input)."""
    data = read_csv_raw(ctx.cleaned_path) if ctx.cleaned_path else ctx.raw
    masked, report = mask_dataframe(data, opts.get("actions") or None)
    files = [ctx.write("pii_report.json", report)]
    if report["pii_columns_found"] or opts.get("actions"):
        masked_path = ctx.out / "masked_data.csv"
        masked.to_csv(masked_path, index=False)
        files.append(str(masked_path))
    return {"status": "passed", "files": files,
            "result": {"pii_columns_found": report["pii_columns_found"],
                       "actions": {r["column"]: r["action"] for r in report["columns"]}}}


def step_quality(ctx: Context, opts: dict) -> dict:
    """Quality score: before/after cleaning when clean ran, else of the input."""
    if ctx.clean_log is not None:
        q = ctx.clean_log["quality"]
        result = {"before": q["before"]["overall"], "after": q["after"]["overall"],
                  "delta": q["delta"]["overall"], "dimensions_after": q["after"]["dimensions"]}
    else:
        score = (ctx.profile or profile_dataset(ctx.raw))["quality_score"]
        result = {"score": score["overall"], "dimensions": score["dimensions"]}
    return {"status": "passed", "files": [ctx.write("quality.json", result)],
            "result": result}


def step_anomalies(ctx: Context, opts: dict) -> dict:
    """Sprint 9 anomaly detection on the input; a gate only with fail_on_share."""
    from src.validation.anomaly import detect

    method = opts.get("method", "iforest")
    section = detect(ctx.raw, methods=(method,), top=opts.get("top", 10),
                     contamination=opts.get("contamination", 0.01),
                     exclude=opts.get("exclude") or [])
    path = ctx.write(f"{ctx.cfg['name']}_anomalies.json", section)
    m = section["methods"][method]
    top = [f"row {t['row']}: " + ", ".join(r["column"] for r in t["reasons"])
           for t in m["top"][:3]]
    out = {"files": [path], "result": {"method": method, "flagged": m["flagged"],
                                       "flagged_share": m["flagged_share"],
                                       "limit": opts.get("fail_on_share"), "top_rows": top}}
    limit = opts.get("fail_on_share")
    if limit is not None and m["flagged_share"] > limit:
        raise StepStop({**out, "status": "failed"})
    return {**out, "status": "passed"}


def step_drift(ctx: Context, opts: dict) -> dict:
    """Sprint 7 drift of the input against a reference file."""
    contract = load_contract(opts["contract"]) if opts.get("contract") else None
    reference = read_csv_raw(opts["reference"])
    report = drift_report(reference, ctx.raw, contract,
                          reference_name=Path(opts["reference"]).name,
                          current_name=Path(ctx.cfg["input"]).name)
    path = ctx.write(f"{ctx.cfg['name']}_drift.json", report)
    out = {"files": [path], "result": {
        "reference": str(opts["reference"]), "counts": report["counts"],
        "major_columns": report["major_columns"],
        "top": [{"column": c["column"], "psi": c["psi"]} for c in report["columns"][:3]]}}
    if report["counts"]["major"] and opts.get("gate", False):
        raise StepStop({**out, "status": "failed"})
    return {**out, "status": "warn" if report["counts"]["major"] else "passed"}


def step_impact(ctx: Context, opts: dict) -> dict:
    """Sprint 8 impact study (A raw / B gatekeeper / C median imputer) on the input."""
    from src.impact.impact import run_impact
    from src.impact.report import write_impact_report

    config = None
    if opts.get("decisions"):
        from src.profiling.decisions import load_decisions
        config = {"decisions": load_decisions(opts["decisions"])}
    report = run_impact(ctx.raw, opts["target"], positive=opts.get("positive"),
                        folds=opts.get("folds", 5), robustness=opts.get("robustness", True),
                        dataset=ctx.cfg["name"], cleaner_config=config)
    files = write_impact_report(report, ctx.out, ctx.cfg["name"])
    headline = {}
    for m, by_metric in report["comparisons"].get("B-A", {}).items():
        c = by_metric["roc_auc"]
        lo, hi = c["ci95"]
        headline[f"{m} ROC-AUC B-A"] = (f"{c['mean_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                                        f"{c['verdict']}")
    return {"status": "passed", "files": list(files.values()), "result": headline}


STEPS: dict[str, Callable[[Context, dict], dict]] = {
    "profile": step_profile, "validate": step_validate, "clean": step_clean,
    "mask_pii": step_mask_pii, "quality": step_quality, "anomalies": step_anomalies,
    "drift": step_drift,
    "impact": step_impact,
}


# ------------------------------------------------------------------- run
def _run_dir(reports_dir: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = Path(reports_dir) / f"run_{stamp}"
    out, n = base, 1
    while out.exists():                       # two runs in the same second
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True)
    return out


def run_pipeline(cfg: dict, skip: set[str] | None = None) -> tuple[int, dict, Path]:
    """Run the configured steps in order -> (exit code, summary dict, run folder)."""
    skip = set(skip or set())
    unknown = skip - set(ORDER)
    if unknown:
        raise ConfigError(f"--skip: unknown step(s) {sorted(unknown)}")
    if not Path(cfg["input"]).is_file():
        raise ConfigError(f"input file not found: {cfg['input']}")
    start = time.perf_counter()
    out = _run_dir(cfg["reports_dir"])
    summary: dict[str, Any] = {
        "name": cfg["name"], "input": cfg["input"],
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(out), "steps": []}
    code = EXIT_OK
    ctx = Context(cfg, out)
    stopped = False
    for name, opts in cfg["steps"]:
        entry: dict[str, Any] = {"step": name}
        if stopped:
            entry.update(status="not_run", reason="an earlier step stopped the run")
        elif name in skip or opts.get("enabled") is False:
            entry.update(status="skipped",
                         reason="--skip" if name in skip else "enabled: false in the config")
        else:
            t0 = time.perf_counter()
            try:
                entry.update(STEPS[name](ctx, opts))
            except StepStop as stop:
                entry.update(stop.args[0])
                code, stopped = EXIT_STOPPED, True
            except Exception as e:  # noqa: BLE001  any crash: record it, stop, exit 2
                entry.update(status="error", error=f"{e.__class__.__name__}: {e}",
                             trace=traceback.format_exc(limit=3))
                code, stopped = EXIT_ERROR, True
            entry["seconds"] = round(time.perf_counter() - t0, 2)
        summary["steps"].append(entry)
    summary["status"] = {EXIT_OK: "passed", EXIT_STOPPED: "stopped at a gate",
                         EXIT_ERROR: "error"}[code]
    summary["exit_code"] = code
    summary["total_seconds"] = round(time.perf_counter() - start, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    return code, summary, out


def _short(result: dict | None) -> str:
    if not result:
        return ""
    parts = []
    for k, v in result.items():
        if isinstance(v, (dict, list)) and k not in ("failed_rules", "major_columns",
                                                     "top_rows", "decisions_from_config"):
            continue
        if isinstance(v, list):
            v = "; ".join(map(str, v)) or "none"
        parts.append(f"{k}: {v}")
    return ", ".join(parts).replace("|", "/")


def summary_markdown(summary: dict) -> str:
    """The run summary as a markdown table."""
    lines = [f"# Gatekeeper run: {summary['name']}", "",
             f"Input `{summary['input']}` · started {summary['started_at']} · "
             f"**{summary['status']}** (exit {summary['exit_code']}) · "
             f"{summary['total_seconds']} s", "",
             "| # | step | status | seconds | result |", "|---|---|---|---|---|"]
    for i, s in enumerate(summary["steps"], 1):
        detail = s.get("error") or s.get("reason") or _short(s.get("result"))
        lines.append(f"| {i} | {s['step']} | {s['status']} | {s.get('seconds', '-')} | "
                     f"{detail} |")
    return "\n".join(lines) + "\n"
