"""Data contracts: a YAML file that says what a dataset must look like, and the check.

A contract has three parts (see contracts/example.yaml):

  table:    min_rows, max_duplicate_rows_pct, extra_columns (severity for columns the
            contract does not list)
  columns:  one entry per expected column: dtype, required, nullable / max_missing_pct,
            allowed_values, min, max, regex, unique, pii
  drift:    psi_moderate, psi_major, bins (used by `gatekeeper drift`, see drift.py)

Every rule has a severity, `error` or `warn`. A rule value is written either plain
(`min: 18`, the column's `severity` is used, default error) or with its own severity
(`min: {value: 18, severity: warn}`). A failed error rule fails the dataset; a failed warn
rule is only reported.

Missing means an empty cell. Placeholders such as "unknown" are ordinary values here
(list them in allowed_values or not); the profiler reports them as disguised missing.

Privacy: values of PII columns (marked `pii: true` in the contract, or detected by
src/privacy/detect.py) are masked with src/privacy/mask.py before they reach a report,
and `contract init` never writes values of a PII column into the contract.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.privacy.detect import detect_column
from src.privacy.mask import mask_value
from src.profiling.profiler import profile_dataset
from src.profiling.semantic import PATTERNS

CONTRACT_VERSION = 1
DTYPES = ("integer", "number", "string", "boolean", "date")
SEVERITIES = ("error", "warn")
COLUMN_RULES = ("dtype", "required", "nullable", "max_missing_pct", "allowed_values",
                "min", "max", "regex", "unique")
COLUMN_KEYS = {"name", "severity", "pii", "description", *COLUMN_RULES}
TABLE_RULES = ("min_rows", "max_duplicate_rows_pct")
TABLE_KEYS = {*TABLE_RULES, "extra_columns"}
DRIFT_DEFAULTS = {"psi_moderate": 0.1, "psi_major": 0.25, "bins": 10}
TOP_KEYS = {"contract_version", "dataset", "description", "table", "columns", "drift",
            "decisions"}

BOOL_WORDS = {"true", "false", "yes", "no", "y", "n", "t", "f", "1", "0"}
INTEGER = re.compile(r"^[+-]?\d+(\.0+)?$")
EXAMPLES = 3              # bad values shown per failed rule
MAX_CATEGORIES = 50       # init: more distinct values than this -> free text, no allowed list
# profiler semantic type -> PII detector type (a regex is drafted only when both agree,
# e.g. an SSN column whose values also fit the loose phone pattern gets no phone regex)
SEMANTIC_PII = {"email": "email", "phone": "phone", "ip_address": "ip_address",
                "postcode": "zip_code"}
RANGE_MARGIN = 0.10       # init: widen the observed min/max by 10% of the range
ROWS_SHARE = 0.10         # init: min_rows = 10% of the reference row count


class ContractError(ValueError):
    """The contract file is missing, not YAML, or breaks the contract format."""


# ------------------------------------------------------------------ loading
def _rule(value: Any, default_severity: str, where: str) -> tuple[Any, str]:
    """Split `x` or `{value: x, severity: s}` into (value, severity)."""
    if isinstance(value, dict):
        unknown = set(value) - {"value", "severity"}
        if unknown or "value" not in value:
            raise ContractError(f"{where}: write either a plain value or "
                                f"{{value: ..., severity: error|warn}}")
        severity = value.get("severity", default_severity)
        if severity not in SEVERITIES:
            raise ContractError(f"{where}: severity must be error or warn, not {severity!r}")
        return value["value"], severity
    return value, default_severity


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_value(rule: str, value: Any, dtype: str | None, where: str) -> Any:
    """Type-check one rule value; returns it normalised (e.g. allowed values as text)."""
    if rule == "dtype":
        if value not in DTYPES:
            raise ContractError(f"{where}: dtype must be one of {', '.join(DTYPES)}, "
                                f"not {value!r}")
    elif rule in ("required", "nullable", "unique"):
        if not isinstance(value, bool):
            raise ContractError(f"{where}: must be true or false")
    elif rule in ("max_missing_pct", "max_duplicate_rows_pct"):
        if not _is_number(value) or not 0 <= value <= 100:
            raise ContractError(f"{where}: must be a number from 0 to 100")
    elif rule == "min_rows":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractError(f"{where}: must be a whole number >= 0")
    elif rule in ("min", "max"):
        if not _is_number(value):
            raise ContractError(f"{where}: must be a number")
        if dtype not in ("integer", "number"):
            raise ContractError(f"{where}: min/max need dtype integer or number")
    elif rule == "allowed_values":
        if not isinstance(value, list) or not value:
            raise ContractError(f"{where}: must be a non-empty list")
        if any(isinstance(v, bool) for v in value):
            # YAML 1.1 reads yes/no/on/off/true/false as booleans
            raise ContractError(f"{where}: contains true/false; quote words like "
                                f"\"yes\" and \"no\" so YAML keeps them as text")
        if any(v is None or isinstance(v, (dict, list)) for v in value):
            raise ContractError(f"{where}: values must be plain text or numbers")
        return [str(v) for v in value]          # the data is read as text
    elif rule == "regex":
        if not isinstance(value, str):
            raise ContractError(f"{where}: must be a text pattern")
        try:
            re.compile(value)
        except re.error as e:
            raise ContractError(f"{where}: invalid regular expression ({e})") from None
    return value


def _parse_column(raw: Any, i: int) -> dict:
    """One `columns:` entry -> {name, pii, rules: {rule: (value, severity)}}."""
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"]:
        raise ContractError(f"columns[{i}]: needs a `name`")
    name = raw["name"]
    where = f"columns[{i}] ({name})"
    unknown = set(raw) - COLUMN_KEYS
    if unknown:
        raise ContractError(f"{where}: unknown key(s) {sorted(unknown)}; "
                            f"allowed: {sorted(COLUMN_KEYS)}")
    severity = raw.get("severity", "error")
    if severity not in SEVERITIES:
        raise ContractError(f"{where}: severity must be error or warn, not {severity!r}")
    pii = raw.get("pii", False)
    if not isinstance(pii, bool):
        raise ContractError(f"{where}: pii must be true or false")

    rules: dict[str, tuple[Any, str]] = {}
    dtype = None
    if "dtype" in raw:
        dtype = _rule(raw["dtype"], severity, f"{where}.dtype")[0]
    for rule in COLUMN_RULES:
        if rule in raw:
            value, sev = _rule(raw[rule], severity, f"{where}.{rule}")
            rules[rule] = (_check_value(rule, value, dtype, f"{where}.{rule}"), sev)

    # nullable is the short form of max_missing_pct: nullable false == at most 0% missing
    if "nullable" in rules:
        nullable, sev = rules.pop("nullable")
        if "max_missing_pct" in rules:
            if not nullable and rules["max_missing_pct"][0] > 0:
                raise ContractError(f"{where}: nullable is false but max_missing_pct > 0")
        elif not nullable:
            rules["max_missing_pct"] = (0, sev)
    if "min" in rules and "max" in rules and rules["min"][0] > rules["max"][0]:
        raise ContractError(f"{where}: min is larger than max")
    rules.setdefault("required", (True, severity))
    return {"name": name, "pii": pii, "rules": rules}


def parse_contract(raw: Any, source: str = "contract") -> dict:
    """Check a loaded YAML document and return the normalised contract.

    Raises ContractError with a message that names the broken part.
    """
    if not isinstance(raw, dict):
        raise ContractError(f"{source}: the top level must be a mapping (key: value)")
    unknown = set(raw) - TOP_KEYS
    if unknown:
        raise ContractError(f"{source}: unknown key(s) {sorted(unknown)}; "
                            f"allowed: {sorted(TOP_KEYS)}")
    version = raw.get("contract_version", CONTRACT_VERSION)
    if version != CONTRACT_VERSION:
        raise ContractError(f"{source}: contract_version {version!r} is not supported "
                            f"(this tool reads version {CONTRACT_VERSION})")

    columns_raw = raw.get("columns")
    if not isinstance(columns_raw, list) or not columns_raw:
        raise ContractError(f"{source}: `columns` must be a non-empty list")
    columns = [_parse_column(c, i) for i, c in enumerate(columns_raw)]
    names = [c["name"] for c in columns]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ContractError(f"{source}: column(s) listed twice: {dupes}")

    table_raw = raw.get("table") or {}
    if not isinstance(table_raw, dict):
        raise ContractError(f"{source}: `table` must be a mapping")
    unknown = set(table_raw) - TABLE_KEYS
    if unknown:
        raise ContractError(f"table: unknown key(s) {sorted(unknown)}; "
                            f"allowed: {sorted(TABLE_KEYS)}")
    table: dict[str, tuple[Any, str]] = {}
    for rule in TABLE_RULES:
        if rule in table_raw:
            value, sev = _rule(table_raw[rule], "error", f"table.{rule}")
            table[rule] = (_check_value(rule, value, None, f"table.{rule}"), sev)
    extra = table_raw.get("extra_columns", "warn")
    if extra not in (*SEVERITIES, "ignore"):
        raise ContractError("table.extra_columns: must be error, warn or ignore")

    drift_raw = raw.get("drift") or {}
    if not isinstance(drift_raw, dict) or set(drift_raw) - set(DRIFT_DEFAULTS):
        raise ContractError(f"drift: allowed keys are {sorted(DRIFT_DEFAULTS)}")
    drift = {**DRIFT_DEFAULTS, **drift_raw}
    if not all(_is_number(drift[k]) and drift[k] > 0 for k in ("psi_moderate", "psi_major")):
        raise ContractError("drift: psi_moderate and psi_major must be positive numbers")
    if drift["psi_moderate"] >= drift["psi_major"]:
        raise ContractError("drift: psi_moderate must be smaller than psi_major")
    if not isinstance(drift["bins"], int) or isinstance(drift["bins"], bool) \
            or not 2 <= drift["bins"] <= 100:
        raise ContractError("drift: bins must be a whole number from 2 to 100")

    # Sprint 11: decisions a person confirmed for the cleaner (src/profiling/decisions.py)
    from src.profiling.decisions import DecisionError, normalise_decisions
    try:
        decisions = normalise_decisions(raw.get("decisions"), source=f"{source}: decisions")
    except DecisionError as e:
        raise ContractError(str(e)) from None
    return {"dataset": raw.get("dataset"), "columns": columns, "table": table,
            "extra_columns": extra, "drift": drift, "decisions": decisions}


def load_contract(path: str | Path) -> dict:
    """Read and check a contract YAML file (ContractError when it is missing or invalid)."""
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"contract file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ContractError(f"{path}: not valid YAML ({e})") from None
    return parse_contract(raw, source=str(path))


# --------------------------------------------------------------- validation
def _show(values: pd.Series | list, pii_type: str | None) -> str:
    """A few example values for a report; PII values are masked first."""
    shown = []
    for v in list(dict.fromkeys(str(x) for x in values))[:EXAMPLES]:
        shown.append(mask_value(v, pii_type) if pii_type else v)
    return ", ".join(repr(v) for v in shown)


def _bad_dtype(s: pd.Series, dtype: str) -> pd.Series:
    """Boolean mask: True where a non-null value is not of the dtype."""
    text = s.astype(str).str.strip()
    if dtype == "integer":
        return ~text.str.match(INTEGER)
    if dtype == "number":
        return pd.to_numeric(text, errors="coerce").isna()
    if dtype == "boolean":
        return ~text.str.lower().isin(BOOL_WORDS)
    if dtype == "date":
        return pd.to_datetime(text, errors="coerce", format="mixed").isna()
    return pd.Series(False, index=s.index)          # string: anything goes


def _result(rule: str, column: str | None, severity: str, ok: bool | None,
            observed: str, expected: str, failing_rows: int = 0) -> dict:
    """One row of the validation table."""
    if ok is None:
        status = "skipped"
    else:
        status = "pass" if ok else ("fail" if severity == "error" else "warn")
    return {"rule": rule, "column": column, "severity": severity, "status": status,
            "observed": observed, "expected": expected, "failing_rows": int(failing_rows)}


def _check_column(series: pd.Series, spec: dict, pii_type: str | None) -> list[dict]:
    """Run every rule of one column spec on the column's values."""
    name, rules = spec["name"], spec["rules"]
    n = len(series)
    non_null = series.dropna()
    out = []

    sev = rules["required"][1]
    out.append(_result("required", name, sev, True, "present", "present"))

    if "dtype" in rules:
        dtype, sev = rules["dtype"]
        bad = non_null[_bad_dtype(non_null, dtype)]
        obs = f"{len(bad)} of {len(non_null)} not {dtype}" + (f": {_show(bad, pii_type)}"
                                                               if len(bad) else "")
        out.append(_result("dtype", name, sev, bad.empty, obs, dtype, len(bad)))

    if "max_missing_pct" in rules:
        limit, sev = rules["max_missing_pct"]
        missing = n - len(non_null)
        pct = missing / n * 100 if n else 0.0
        out.append(_result("max_missing_pct", name, sev, pct <= limit,
                           f"{pct:.2f}% ({missing} rows)", f"<= {limit}%", missing))

    if "allowed_values" in rules:
        allowed, sev = rules["allowed_values"]
        text = non_null.astype(str)
        bad = text[~text.isin(set(allowed))]
        obs = (f"{len(bad)} rows, {bad.nunique()} values not allowed: {_show(bad, pii_type)}"
               if len(bad) else "all values allowed")
        exp = f"{len(allowed)} values" if pii_type else _short(allowed)
        out.append(_result("allowed_values", name, sev, bad.empty, obs, exp, len(bad)))

    numbers = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce").dropna()
    for rule, op in (("min", "lt"), ("max", "gt")):
        if rule not in rules:
            continue
        limit, sev = rules[rule]
        bad = numbers[numbers < limit] if op == "lt" else numbers[numbers > limit]
        if numbers.empty:
            out.append(_result(rule, name, sev, None, "no numeric values", f"{rule} {limit}"))
            continue
        actual = numbers.min() if rule == "min" else numbers.max()
        shown = mask_value(str(actual), pii_type) if pii_type else f"{actual:g}"
        obs = f"{rule} {shown}" + (f", {len(bad)} rows outside" if len(bad) else "")
        sign = ">=" if rule == "min" else "<="
        out.append(_result(rule, name, sev, bad.empty, obs, f"{sign} {limit}", len(bad)))

    if "regex" in rules:
        pattern, sev = rules["regex"]
        text = non_null.astype(str)
        bad = text[~text.str.fullmatch(pattern)]
        obs = (f"{len(bad)} of {len(text)} do not match: {_show(bad, pii_type)}"
               if len(bad) else "all match")
        out.append(_result("regex", name, sev, bad.empty, obs, pattern, len(bad)))

    if rules.get("unique", (False,))[0]:
        sev = rules["unique"][1]
        dup = non_null[non_null.duplicated(keep=False)]
        obs = (f"{non_null.duplicated().sum()} repeated values: {_show(dup, pii_type)}"
               if len(dup) else "all unique")
        out.append(_result("unique", name, sev, dup.empty, obs, "unique", len(dup)))
    return out


def _short(values: list[str], limit: int = 6) -> str:
    """'a, b, c, ... (+5 more)' for a long list of allowed values."""
    head = ", ".join(values[:limit])
    return head + (f", ... (+{len(values) - limit} more)" if len(values) > limit else "")


def pii_columns(df: pd.DataFrame, contract: dict | None = None) -> dict[str, str]:
    """{column: pii_type} from PII detection plus columns marked `pii: true`."""
    found = {}
    for c in df.columns:
        hit = detect_column(str(c), df[c])
        if hit:
            found[str(c)] = hit["pii_type"]
    for spec in (contract or {}).get("columns", []):
        if spec["pii"]:
            found.setdefault(spec["name"], "pii")
    return found


def validate_dataframe(df: pd.DataFrame, contract: dict, dataset: str = "dataset") -> dict:
    """Check a DataFrame (read as text) against a parsed contract -> report dict."""
    pii = pii_columns(df, contract)
    results: list[dict] = []
    n = len(df)

    if "min_rows" in contract["table"]:
        limit, sev = contract["table"]["min_rows"]
        results.append(_result("min_rows", None, sev, n >= limit, str(n), f">= {limit}"))
    if "max_duplicate_rows_pct" in contract["table"]:
        limit, sev = contract["table"]["max_duplicate_rows_pct"]
        dups = int(df.duplicated().sum())
        pct = dups / n * 100 if n else 0.0
        results.append(_result("max_duplicate_rows_pct", None, sev, pct <= limit,
                               f"{pct:.2f}% ({dups} rows)", f"<= {limit}%", dups))

    listed = [c["name"] for c in contract["columns"]]
    extra = [str(c) for c in df.columns if str(c) not in listed]
    if contract["extra_columns"] != "ignore":
        results.append(_result("extra_columns", None, contract["extra_columns"], not extra,
                               ", ".join(extra) if extra else "none", "none"))

    for spec in contract["columns"]:
        name = spec["name"]
        if name not in df.columns:
            required, sev = spec["rules"]["required"]
            results.append(_result("required", name, sev, False if required else None,
                                   "column missing", "present" if required else "optional"))
            continue
        results.extend(_check_column(df[name], spec, pii.get(name)))

    counts = {s: sum(r["status"] == s for r in results)
              for s in ("pass", "fail", "warn", "skipped")}
    missing_cols = [r["column"] for r in results
                    if r["rule"] == "required" and r["observed"] == "column missing"]
    return {
        "dataset": dataset,
        "rows": n,
        "columns": int(df.shape[1]),
        "status": "fail" if counts["fail"] else "pass",
        "counts": {"rules": len(results), "passed": counts["pass"],
                   "failed_error": counts["fail"], "failed_warn": counts["warn"],
                   "skipped": counts["skipped"]},
        "column_set": {"missing": missing_cols, "extra": extra},
        "pii_columns": sorted(pii),
        "results": results,
    }


# ---------------------------------------------------------------- init
def _floor(x: float, integer: bool) -> float:
    return math.floor(x) if integer else math.floor(round(x * 1e4, 6)) / 1e4


def _ceil(x: float, integer: bool) -> float:
    return math.ceil(x) if integer else math.ceil(round(x * 1e4, 6)) / 1e4


def _num(x: float) -> int | float:
    """Plain int when the number is whole (cleaner YAML: 17 not 17.0)."""
    return int(x) if float(x).is_integer() else float(x)


def draft_contract(df: pd.DataFrame, name: str, source: str | None = None) -> dict:
    """Draft a contract from a reference dataset (read as text) using the profiler.

    The profiler decides each column's type and meaning; ranges and category lists come
    from all observed values (the profiler's stats leave sentinels such as pdays = 999
    out, and list only the top 10 categories). The draft is a starting point to edit:
      - every column required; nullable: false when nothing is missing, otherwise
        max_missing_pct = observed x 1.5 + 1
      - numeric: integer / number, min / max = observed range widened by 10% (not below
        0 when nothing was negative)
      - categorical (<= 50 values): allowed_values = every observed value
      - email / phone / IP / postcode / URL: a regex, severity warn
      - id columns that are fully unique: unique: true
      - PII columns: pii: true and no values written (no allowed list, no range)
    """
    report = profile_dataset(df, dataset_name=name)
    pii = pii_columns(df)
    columns = []
    for col in report["columns"]:
        cname = col["name"]
        s = df[cname]
        non_null = s.dropna().astype(str).str.strip()
        spec: dict[str, Any] = {"name": cname}
        kind = col["inferred_type"]
        is_pii = cname in pii or col["is_pii"]

        if kind == "numeric":
            numbers = pd.to_numeric(non_null, errors="coerce")
            integer = bool(non_null[numbers.notna()].str.match(INTEGER).all())
            spec["dtype"] = "integer" if integer else "number"
        elif kind == "datetime":
            spec["dtype"] = "date"
        elif kind == "boolean":
            spec["dtype"] = "boolean"
        else:
            spec["dtype"] = "string"
        spec["required"] = True
        missing_pct = col["missing_pct"]
        if missing_pct == 0:
            spec["nullable"] = False
        else:
            spec["max_missing_pct"] = round(min(100.0, missing_pct * 1.5 + 1), 1)

        if is_pii:
            spec["pii"] = True
        elif spec["dtype"] in ("integer", "number"):
            numbers = pd.to_numeric(non_null, errors="coerce").dropna()
            if not numbers.empty:
                lo, hi = float(numbers.min()), float(numbers.max())
                margin = (hi - lo) * RANGE_MARGIN
                low = lo - margin if lo < 0 else max(0.0, lo - margin)
                integer = spec["dtype"] == "integer"
                spec["min"] = _num(_floor(low, integer))
                spec["max"] = _num(_ceil(hi + margin, integer))
        elif kind == "categorical" and 0 < non_null.nunique() <= MAX_CATEGORIES:
            spec["allowed_values"] = sorted(s.dropna().astype(str).unique())

        semantic = col["semantic"]["semantic_type"]
        if semantic in PATTERNS and pii.get(cname, SEMANTIC_PII.get(semantic)) == \
                SEMANTIC_PII.get(semantic):
            compiled = PATTERNS[semantic]
            pattern = re.sub(r"^\^|\$$", "", compiled.pattern)   # the check uses fullmatch
            if not compiled.pattern.endswith("$"):                  # url: only a prefix
                pattern += ".*"
            if compiled.flags & re.IGNORECASE:
                pattern = "(?i)" + pattern
            spec["regex"] = {"value": pattern, "severity": "warn"}
        if semantic == "id" and col["unique_count"] == len(non_null) > 0:
            spec["unique"] = True
        columns.append(spec)

    rows = report["dataset"]["rows"]
    dup_pct = report["dataset"]["duplicate_rows"] / rows * 100 if rows else 0.0
    return {
        "contract_version": CONTRACT_VERSION,
        "dataset": source or name,
        "table": {
            "min_rows": max(1, math.ceil(rows * ROWS_SHARE)),
            "max_duplicate_rows_pct": {"value": min(100.0, max(1.0, round(dup_pct * 2, 2))),
                                       "severity": "warn"},
            "extra_columns": "warn",
        },
        "columns": columns,
        "drift": dict(DRIFT_DEFAULTS),
    }


def contract_to_yaml(contract: dict, reference: str) -> str:
    """YAML text of a drafted contract, with a short header on how to edit it."""
    header = (f"# Data contract drafted by `gatekeeper contract init` from {reference}.\n"
              "# It describes what the reference data looked like: edit it to say what\n"
              "# new data MUST look like (tighten or loosen ranges, add regexes, set\n"
              "# severity: warn on rules that should not fail a run).\n"
              "# Rule forms:  min: 18   or   min: {value: 18, severity: warn}\n")
    body = yaml.safe_dump(contract, sort_keys=False, allow_unicode=True,
                          default_flow_style=None, width=100)
    return header + body


def contract_ranges(contract: dict) -> dict[str, list[float]]:
    """{column: [min, max]} for columns whose contract sets both (for step 8b on new rows)."""
    out = {}
    for spec in contract["columns"]:
        rules = spec["rules"]
        if "min" in rules and "max" in rules:
            out[spec["name"]] = [float(rules["min"][0]), float(rules["max"][0])]
    return out
