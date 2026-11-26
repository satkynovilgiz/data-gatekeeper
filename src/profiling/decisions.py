"""Decisions for a person: values the gatekeeper is NOT sure how to treat (Sprint 11).

Sprint 10 showed on UCI Adult that two automatic choices can destroy real signal:
  capital-gain = 99999  a top code (the highest amount the census records), not "no value";
                        all 244 rows earn >50K, but the profiler called it a sentinel -> missing
  workclass = "?"       a missing answer that MEANS something (9.47% >50K vs 24.81%), imputed away
So the profiler now lists such values as DECISIONS, with evidence, instead of deciding alone:

  top_code_or_sentinel  a numeric value the profiler flagged as a sentinel (999-style, the
                        column's maximum or minimum). Listed always.
  informative_missing   a disguised-missing code ("?", "unknown", ...) of a column. With a target:
                        listed only when its rows have a CLEARLY different label rate. Without a
                        target: every code is listed with counts only.

"Clearly different" (DECISION_RULE, fixed before any run and never tuned on results):
  |rate in the rows - rate in the other rows| > 5 percentage points  AND  a two-proportion
  z-test p-value < 0.01  AND  at least 30 rows hold the value.
  5 points is a difference a person would care about; p < 0.01 keeps chance out on large files;
  30 rows keeps a handful of rows from deciding anything.

Every decision has the action used today (current_default) and a recommendation:
  keep_as_value      the number stays a number (it carries signal)
  treat_as_missing   the value becomes NaN (and may be imputed)
  keep_as_category   the code stays a category of its own ("unknown"), never imputed
  impute             the code becomes NaN and is filled in (today's default for text codes)
  special_code       already handled by a config rule (Bank: pdays 999 -> previously_contacted)
A person confirms them in the contract / config (`decisions:`); the cleaner obeys and logs them.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import norm

DECISION_RULE = {"min_diff": 0.05, "max_p": 0.01, "min_rows": 30}
ACTIONS = ("keep_as_value", "treat_as_missing", "keep_as_category", "impute")


class DecisionError(ValueError):
    """A decisions block is not valid."""


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """(difference p1 - p2, two-sided p-value) of a pooled two-proportion z-test."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, 1.0
    return p1 - p2, float(2 * norm.sf(abs((p1 - p2) / se)))


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _evidence(rows: pd.Series, y: pd.Series | None, rule: dict) -> dict:
    """Counts, and with a target the label rates in / out and whether they clearly differ."""
    n = int(rows.sum())
    out: dict[str, Any] = {"rows": n, "share": round(n / len(rows), 6) if len(rows) else 0.0}
    if y is None:
        out.update(label_rate_in=None, label_rate_rest=None, diff_pp=None, p_value=None,
                   clearly_different=None)
        return out
    known = y.notna()
    inside, rest = rows & known, ~rows & known
    k1, n1 = int(y[inside].sum()), int(inside.sum())
    k2, n2 = int(y[rest].sum()), int(rest.sum())
    diff, p = two_proportion_test(k1, n1, k2, n2)
    clear = abs(diff) > rule["min_diff"] and p < rule["max_p"] and n1 >= rule["min_rows"]
    out.update(label_rate_in=round(k1 / n1, 4) if n1 else None,
               label_rate_rest=round(k2 / n2, 4) if n2 else None,
               diff_pp=round(diff * 100, 2), p_value=p, clearly_different=bool(clear))
    return out


def find_decisions(df: pd.DataFrame, report: dict, target: str | None = None,
                   positive: str | None = None, config: dict | None = None,
                   rule: dict | None = None) -> list[dict]:
    """The decisions a person should confirm, with evidence (df read as text).

    report: the Module 1 profile of df. config: the cleaning config (DEFAULT_CONFIG merged), to
    tell today's default (special codes, keep_as_category, decisions already set).
    """
    from src.cleaning.cleaner import DEFAULT_CONFIG

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rule = {**DECISION_RULE, **(rule or {})}
    y = None
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"target column {target!r} not found")
        from src.impact.impact import positive_label

        pos = positive_label(df[target], positive)
        y = (df[target].astype(str) == pos).astype(float).where(df[target].notna())
    set_by_config = normalise_decisions(cfg.get("decisions") or {})
    out = []
    for col in report["columns"]:
        c = col["name"]
        if c == target:
            continue
        s = df[c].astype(str).str.strip()
        for code in col["disguised_missing"]["values"]:
            rows = (s == str(code)) & df[c].notna()
            numeric = _is_number(str(code)) and col.get("numeric_stats") is not None
            ev = _evidence(rows, y, rule)
            special = cfg["special_codes"].get(c)
            if numeric:
                kind = "top_code_or_sentinel"
                if special and float(code) == float(special["code"]):
                    default = "special_code"
                    recommended = "special_code"
                    reason = (f"already a config rule: {c} = {code} -> "
                              f"{special['flag_column']}; the signal is kept in that flag")
                else:
                    default = "treat_as_missing"
                    if ev["clearly_different"]:
                        recommended = "keep_as_value"
                        reason = "its rows have a clearly different label rate: it carries signal"
                    elif y is None:
                        recommended = None
                        reason = "no target given: check whether it is a code or a real value"
                    else:
                        recommended = "treat_as_missing"
                        reason = "label rate not clearly different: a 'no value' code"
            else:
                if y is not None and not ev["clearly_different"]:
                    continue                     # uninformative: imputing it is fine
                kind = "informative_missing"
                default = "keep_as_category" if c in cfg["keep_as_category"] else "impute"
                recommended = "keep_as_category" if ev["clearly_different"] else None
                reason = ("its rows have a clearly different label rate: the missing answer "
                          "means something" if ev["clearly_different"]
                          else "no target given: counts only")
            chosen = set_by_config.get(c, {}).get(str(code))
            out.append({"column": c, "value": str(code), "kind": kind, **ev,
                        "current_default": default, "recommended": recommended,
                        "configured": chosen, "reason": reason})
    return out


# ------------------------------------------------------------------ config
def normalise_decisions(raw: Any, source: str = "decisions") -> dict[str, dict[str, str]]:
    """Check a `decisions:` block -> {column: {value as text: action}}."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise DecisionError(f"{source}: must be a mapping column -> {{value: action}}")
    out: dict[str, dict[str, str]] = {}
    for column, choices in raw.items():
        if not isinstance(choices, dict) or not choices:
            raise DecisionError(f"{source}.{column}: write {{value: action}}, e.g. {{'?': "
                                f"keep_as_category}}")
        for value, action in choices.items():
            if action not in ACTIONS:
                raise DecisionError(f"{source}.{column}.{value}: action must be one of "
                                    f"{', '.join(ACTIONS)}, not {action!r}")
            if isinstance(value, bool) or value is None:
                raise DecisionError(f"{source}.{column}: quote the value (YAML read {value!r})")
            text = str(value)
            if isinstance(value, float) and value.is_integer():
                text = str(int(value))
            out.setdefault(str(column), {})[text] = action
    return out


def load_decisions(path: str | Path) -> dict[str, dict[str, str]]:
    """The `decisions:` block of a YAML file (a contract or a decisions-only file)."""
    path = Path(path)
    if not path.is_file():
        raise DecisionError(f"decisions file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise DecisionError(f"{path}: not valid YAML ({e})") from None
    if not isinstance(raw, dict) or "decisions" not in raw:
        raise DecisionError(f"{path}: no `decisions:` block")
    return normalise_decisions(raw["decisions"], source=f"{path}: decisions")


def decisions_yaml(decisions: list[dict], source: str) -> str:
    """A DRAFT `decisions:` block from the recommendations, for a person to confirm."""
    block: dict[str, dict[str, str]] = {}
    lines = [f"# DRAFT decisions for {source}, written by `gatekeeper decisions`.",
             "# A person must confirm each one: the recommendation comes from label rates,",
             "# not from knowing what the value means. Actions: " + ", ".join(ACTIONS) + "."]
    for d in decisions:
        action = d["recommended"]
        if action not in ACTIONS:
            continue
        block.setdefault(d["column"], {})[d["value"]] = action
        rate = ("" if d["label_rate_in"] is None else
                f": label rate {d['label_rate_in']:.2%} vs {d['label_rate_rest']:.2%}")
        lines.append(f"#   {d['column']} = {d['value']} ({d['rows']} rows){rate}")
    body = yaml.safe_dump({"decisions": block}, sort_keys=False, default_flow_style=None,
                          allow_unicode=True)
    return "\n".join(lines) + "\n" + body
