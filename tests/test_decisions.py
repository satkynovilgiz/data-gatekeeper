"""Sprint 11: decisions for a person — detector, config, cleaner, CLI."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.cleaning.cleaner import clean_dataset
from src.profiling.decisions import (
    DECISION_RULE,
    DecisionError,
    decisions_yaml,
    find_decisions,
    load_decisions,
    normalise_decisions,
    two_proportion_test,
)
from src.profiling.profiler import profile_dataset
from src.validation.contract import ContractError, parse_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def people():
    """1,000 rows: a top code 99999 that is always 'yes', an informative '?' in job and an
    uninformative '?' in country."""
    rng = np.random.default_rng(7)
    n = 1000
    gain = rng.integers(0, 5000, n).astype(str)
    y = rng.random(n) < 0.25
    top = rng.choice(n, 40, replace=False)
    gain[top] = "99999"
    y[top] = True
    job = rng.choice(["clerk", "sales", "tech"], n).astype(object)
    no_job = rng.choice(np.setdiff1d(np.arange(n), top), 120, replace=False)
    job[no_job] = "?"
    y[no_job] = rng.random(120) < 0.03                           # "?" rows rarely 'yes'
    country = rng.choice(["US", "MX", "IN"], n).astype(object)
    country[rng.choice(n, 80, replace=False)] = "?"              # independent of y
    return pd.DataFrame({"customer_id": [f"C{i}" for i in range(n)],   # no duplicate rows
                         "gain": gain, "job": job, "country": country,
                         "y": np.where(y, "yes", "no")})


def test_rule_is_fixed_and_test_by_hand():
    assert DECISION_RULE == {"min_diff": 0.05, "max_p": 0.01, "min_rows": 30}
    # 30/100 vs 20/100: pooled 0.25, se = sqrt(.25*.75*.02) = 0.06124, z = 1.633, p = 0.1025
    diff, p = two_proportion_test(30, 100, 20, 100)
    assert diff == pytest.approx(0.10) and p == pytest.approx(0.1025, abs=1e-3)


def test_top_code_and_informative_missing_are_listed_uninformative_is_not(people):
    found = find_decisions(people, profile_dataset(people), target="y")
    by = {(d["column"], d["value"]): d for d in found}
    top = by[("gain", "99999")]
    assert top["kind"] == "top_code_or_sentinel" and top["rows"] == 40
    assert top["label_rate_in"] == 1.0 and top["clearly_different"]
    assert top["current_default"] == "treat_as_missing" and top["recommended"] == "keep_as_value"
    q = by[("job", "?")]
    assert q["kind"] == "informative_missing" and q["recommended"] == "keep_as_category"
    assert q["current_default"] == "impute"
    assert ("country", "?") not in by                             # not clearly different


def test_without_target_counts_only(people):
    no_y = people.drop(columns="y")
    found = find_decisions(no_y, profile_dataset(no_y))
    by = {(d["column"], d["value"]): d for d in found}
    assert set(by) == {("gain", "99999"), ("job", "?"), ("country", "?")}
    assert all(d["label_rate_in"] is None and d["p_value"] is None for d in found)
    assert by[("gain", "99999")]["rows"] == 40 and by[("country", "?")]["rows"] == 80
    assert all(d["recommended"] is None for d in found)


def test_special_code_is_reported_as_kept():
    rng = np.random.default_rng(1)
    pdays = np.where(rng.random(500) < 0.9, "999", rng.integers(0, 20, 500).astype(str))
    df = pd.DataFrame({"pdays": pdays, "y": np.where(pdays == "999", "no", "yes")})
    (d,) = find_decisions(df, profile_dataset(df), target="y")
    assert d["column"] == "pdays" and d["current_default"] == d["recommended"] == "special_code"


def test_config_decisions_are_obeyed_and_logged(people):
    cfg = {"numeric_strategy": "median", "categorical_strategy": "mode",
           "decisions": {"gain": {99999: "keep_as_value"}, "job": {"?": "keep_as_category"}}}
    out, log = clean_dataset(people, profile_dataset(people), cfg)
    assert (out["gain"] == 99999).sum() == 40                      # kept as a number
    assert (out["job"] == "unknown").sum() == 120                  # kept as a category
    assert out["country"].ne("?").all() and out["country"].notna().all()   # default: imputed
    entries = {(d["column"], d["value"]): d for d in log["decisions"]}
    assert entries[("gain", "99999")] == {"column": "gain", "value": "99999",
                                          "action": "keep_as_value", "set_by": "config"}
    assert entries[("job", "?")]["set_by"] == "config"
    assert entries[("country", "?")] == {"column": "country", "value": "?",
                                         "action": "impute", "set_by": "default"}
    _, default_log = clean_dataset(people, profile_dataset(people),
                                   {"numeric_strategy": "median", "categorical_strategy": "mode"})
    assert all(d["set_by"] == "default" for d in default_log["decisions"])


def test_decisions_format_errors_and_contract(tmp_path):
    assert normalise_decisions({"a": {99999: "keep_as_value", 1.0: "impute"}}) == {
        "a": {"99999": "keep_as_value", "1": "impute"}}
    with pytest.raises(DecisionError, match="action must be one of"):
        normalise_decisions({"a": {"?": "drop"}})
    with pytest.raises(DecisionError, match="quote the value"):
        normalise_decisions({"a": {True: "impute"}})
    with pytest.raises(DecisionError, match="write"):
        normalise_decisions({"a": "impute"})
    c = parse_contract({"columns": [{"name": "a"}], "decisions": {"a": {"?": "impute"}}})
    assert c["decisions"] == {"a": {"?": "impute"}}
    with pytest.raises(ContractError, match="action must be one of"):
        parse_contract({"columns": [{"name": "a"}], "decisions": {"a": {"?": "x"}}})
    assert load_decisions(ROOT / "contracts/adult_decisions.yaml") == {
        "capital-gain": {"99999": "keep_as_value"}, "workclass": {"?": "keep_as_category"},
        "occupation": {"?": "keep_as_category"}}
    (tmp_path / "x.yaml").write_text("columns: []\n")
    with pytest.raises(DecisionError, match="no `decisions:` block"):
        load_decisions(tmp_path / "x.yaml")


def test_draft_yaml_round_trips(people):
    found = find_decisions(people, profile_dataset(people), target="y")
    text = decisions_yaml(found, "people.csv")
    assert text.startswith("# DRAFT")
    assert normalise_decisions(yaml.safe_load(text)["decisions"]) == {
        "gain": {"99999": "keep_as_value"}, "job": {"?": "keep_as_category"}}


def test_cli_decisions(tmp_path, people):
    csv = tmp_path / "people.csv"
    people.to_csv(csv, index=False)
    out = tmp_path / "draft.yaml"
    r = subprocess.run([sys.executable, str(ROOT / "gatekeeper.py"), "decisions", str(csv),
                        "--target", "y", "--write", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "keep_as_value" in r.stdout and "99999" in r.stdout and out.exists()
    r = subprocess.run([sys.executable, str(ROOT / "gatekeeper.py"), "decisions", str(csv),
                        "--target", "nope"], capture_output=True, text=True)
    assert r.returncode == 2
