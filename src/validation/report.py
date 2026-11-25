"""One Module 3 report: contract + drift + anomalies -> validation_report.json (Sprint 9).

Used by `gatekeeper validate`, the pipeline and POST /validate, so all three give the same
report (format: docs/validation_report.schema.json).

  contract   always: the Sprint 7 rule check (validate_dataframe)
  drift      when a reference file is given (drift_report); information only, it never
             fails the report (use `gatekeeper drift` or the pipeline's drift gate for that)
  anomalies  when asked (anomaly.detect); fitted on the reference when there is one.
             Information only, unless fail_on_share is set: then a flagged share above it
             fails the report.
status = "fail" when an error-level contract rule fails or the anomaly gate trips.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.validation.anomaly import METHODS, TOP_K, detect
from src.validation.contract import validate_dataframe
from src.validation.drift import drift_report

REPORT_VERSION = "1.0"
DEFAULT_METHOD = "iforest"


def validation_report(df: pd.DataFrame, contract: dict, dataset: str,
                      reference: pd.DataFrame | None = None, reference_name: str = "reference",
                      anomalies: bool = False, methods: tuple[str, ...] = (DEFAULT_METHOD,),
                      top: int = TOP_K, contamination: float = 0.01,
                      exclude: list[str] | None = None,
                      fail_on_share: float | None = None) -> dict:
    """Build the combined report for one dataset (all frames read as text)."""
    if fail_on_share is not None and not 0 <= fail_on_share <= 1:
        raise ValueError("the anomaly share limit must be between 0 and 1")
    if fail_on_share is not None and not anomalies:
        raise ValueError("an anomaly share limit needs the anomaly section")
    contract_part = validate_dataframe(df, contract, dataset=dataset)
    drift_part = None
    if reference is not None:
        drift_part = drift_report(reference, df, contract, reference_name=reference_name,
                                  current_name=dataset)
    anomaly_part = None
    gate = None
    if anomalies:
        anomaly_part = detect(df, methods=methods, top=top, contamination=contamination,
                              reference=reference, exclude=exclude, contract=contract)
        share = anomaly_part["methods"][methods[0]]["flagged_share"]
        gate = {"method": methods[0], "flagged_share": share, "limit": fail_on_share,
                "status": "pass" if fail_on_share is None or share <= fail_on_share
                else "fail"}
    failed = contract_part["status"] == "fail" or (gate is not None and gate["status"] == "fail")
    return {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": dataset, "rows": int(len(df)), "columns": int(df.shape[1])},
        "status": "fail" if failed else "pass",
        "gates": {"contract": contract_part["status"], "anomalies": gate},
        "contract": contract_part,
        "drift": drift_part,
        "anomalies": anomaly_part,
    }


def parse_methods(text: str) -> tuple[str, ...]:
    """'all' or 'iforest' or 'zscore,iqr' -> a tuple of method names (first = the gate's)."""
    if text == "all":
        return METHODS
    methods = tuple(m.strip() for m in text.split(",") if m.strip())
    bad = set(methods) - set(METHODS)
    if not methods or bad:
        raise ValueError(f"method must be one of {', '.join(METHODS)} or all")
    return methods
