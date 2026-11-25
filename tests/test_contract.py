"""Data contracts (Sprint 7): parsing + format errors, every rule pass/fail, init."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.validation.contract import (
    ContractError,
    contract_to_yaml,
    draft_contract,
    load_contract,
    parse_contract,
    validate_dataframe,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "contracts/example.yaml"


def contract(*columns, table=None, **top):
    """A parsed contract built from plain dicts."""
    return parse_contract({"columns": list(columns), "table": table or {}, **top})


def run(df, *columns, table=None, extra="ignore"):
    """Validate df; returns {(rule, column): result}."""
    c = contract(*columns, table={**(table or {}), "extra_columns": extra})
    report = validate_dataframe(df, c)
    return report, {(r["rule"], r["column"]): r for r in report["results"]}


# ------------------------------------------------------------------ parsing
def test_example_contract_loads():
    c = load_contract(EXAMPLE)
    by_name = {col["name"]: col for col in c["columns"]}
    assert by_name["amount"]["rules"]["max"] == (1000, "warn")      # per-rule severity
    assert by_name["amount"]["rules"]["min"] == (0, "error")        # column default
    assert by_name["order_id"]["rules"]["max_missing_pct"] == (0, "error")  # nullable: false
    assert by_name["coupon"]["rules"]["required"] == (False, "error")
    assert by_name["email"]["pii"] is True
    assert c["drift"]["bins"] == 5


@pytest.mark.parametrize("raw, message", [
    ([1, 2], "top level must be a mapping"),
    ({"columns": []}, "non-empty list"),
    ({"columns": [{"dtype": "integer"}]}, "needs a `name`"),
    ({"columns": [{"name": "a", "nulable": False}]}, "unknown key"),
    ({"colums": [], "columns": [{"name": "a"}]}, "unknown key"),
    ({"columns": [{"name": "a", "dtype": "int"}]}, "dtype must be one of"),
    ({"columns": [{"name": "a", "severity": "fatal"}]}, "severity must be error or warn"),
    ({"columns": [{"name": "a", "dtype": "number", "min": {"value": 1, "severity": "x"}}]},
     "severity must be error or warn"),
    ({"columns": [{"name": "a", "dtype": "number", "min": 5, "max": 1}]}, "min is larger"),
    ({"columns": [{"name": "a", "dtype": "string", "min": 1}]}, "need dtype integer or number"),
    ({"columns": [{"name": "a", "dtype": "number", "min": "low"}]}, "must be a number"),
    ({"columns": [{"name": "a", "regex": "[unclosed"}]}, "invalid regular expression"),
    ({"columns": [{"name": "a", "allowed_values": [True, False]}]}, "quote words like"),
    ({"columns": [{"name": "a", "allowed_values": []}]}, "non-empty list"),
    ({"columns": [{"name": "a", "nullable": "no"}]}, "true or false"),
    ({"columns": [{"name": "a", "nullable": False, "max_missing_pct": 5}]},
     "nullable is false but max_missing_pct > 0"),
    ({"columns": [{"name": "a", "max_missing_pct": 150}]}, "from 0 to 100"),
    ({"columns": [{"name": "a"}, {"name": "a"}]}, "listed twice"),
    ({"columns": [{"name": "a"}], "table": {"min_rows": -1}}, "whole number >= 0"),
    ({"columns": [{"name": "a"}], "table": {"extra_columns": "maybe"}}, "error, warn or ignore"),
    ({"columns": [{"name": "a"}], "drift": {"psi_moderate": 0.3, "psi_major": 0.2}},
     "smaller than psi_major"),
    ({"columns": [{"name": "a"}], "drift": {"bins": 1}}, "bins must be"),
    ({"columns": [{"name": "a"}], "contract_version": 2}, "not supported"),
])
def test_bad_contracts_give_clear_errors(raw, message):
    with pytest.raises(ContractError, match=message):
        parse_contract(raw)


def test_load_contract_file_errors(tmp_path):
    with pytest.raises(ContractError, match="not found"):
        load_contract(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("columns: [\n  - name: a\n")
    with pytest.raises(ContractError, match="not valid YAML"):
        load_contract(bad)


def test_unquoted_yes_no_is_caught_but_quoted_works(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("columns:\n  - name: y\n    allowed_values: [no, yes]\n")
    with pytest.raises(ContractError, match="quote"):
        load_contract(f)
    f.write_text("columns:\n  - name: y\n    allowed_values: ['no', 'yes']\n")
    assert load_contract(f)["columns"][0]["rules"]["allowed_values"][0] == ["no", "yes"]


# --------------------------------------------------------- each rule type
@pytest.mark.parametrize("dtype, good, bad", [
    ("integer", ["1", "-2", "30", "4.0"], ["1", "2.5"]),
    ("number", ["1", "2.5", "-3e2"], ["1", "abc"]),
    ("boolean", ["yes", "No", "true", "0"], ["yes", "maybe"]),
    ("date", ["2026-11-01", "2026-01-31"], ["2026-11-01", "not a date"]),
])
def test_dtype_rule(dtype, good, bad):
    _, ok = run(pd.DataFrame({"a": good}), {"name": "a", "dtype": dtype})
    assert ok[("dtype", "a")]["status"] == "pass"
    _, fail = run(pd.DataFrame({"a": bad}), {"name": "a", "dtype": dtype})
    assert fail[("dtype", "a")]["status"] == "fail"
    assert fail[("dtype", "a")]["failing_rows"] == 1


def test_string_dtype_accepts_anything():
    _, r = run(pd.DataFrame({"a": ["x", "1", "2026-01-01"]}), {"name": "a", "dtype": "string"})
    assert r[("dtype", "a")]["status"] == "pass"


def test_missing_rules():
    df = pd.DataFrame({"a": ["1", None, "3", "4"]})                  # 25% missing
    _, r = run(df, {"name": "a", "nullable": False})
    assert r[("max_missing_pct", "a")]["status"] == "fail"
    assert r[("max_missing_pct", "a")]["observed"].startswith("25.00%")
    _, r = run(df, {"name": "a", "max_missing_pct": 25})
    assert r[("max_missing_pct", "a")]["status"] == "pass"
    _, r = run(df, {"name": "a", "nullable": True})                 # no limit at all
    assert ("max_missing_pct", "a") not in r


def test_allowed_values_rule():
    _, r = run(pd.DataFrame({"c": ["KG", "KZ", None]}),
               {"name": "c", "allowed_values": ["KG", "KZ"]})
    assert r[("allowed_values", "c")]["status"] == "pass"          # missing is not "not allowed"
    _, r = run(pd.DataFrame({"c": ["KG", "RU", "RU"]}),
               {"name": "c", "allowed_values": ["KG", "KZ"]})
    res = r[("allowed_values", "c")]
    assert res["status"] == "fail" and res["failing_rows"] == 2 and "'RU'" in res["observed"]


def test_numbers_in_allowed_values_match_text():
    _, r = run(pd.DataFrame({"n": ["1", "2", "3"]}),
               {"name": "n", "allowed_values": [1, 2, 3]})
    assert r[("allowed_values", "n")]["status"] == "pass"


@pytest.mark.parametrize("rule, limit, values, status", [
    ("min", 0, ["0", "5"], "pass"),
    ("min", 0, ["-1", "5"], "fail"),
    ("max", 10, ["10", "5"], "pass"),
    ("max", 10, ["11", "5"], "fail"),
])
def test_range_rules(rule, limit, values, status):
    _, r = run(pd.DataFrame({"x": values}), {"name": "x", "dtype": "number", rule: limit})
    assert r[(rule, "x")]["status"] == status


def test_range_skipped_when_no_numbers():
    _, r = run(pd.DataFrame({"x": ["a", "b"]}), {"name": "x", "dtype": "number", "min": 0})
    assert r[("min", "x")]["status"] == "skipped"


def test_regex_rule_is_a_full_match():
    col = {"name": "code", "regex": r"[A-Z]{2}\d{3}"}
    _, r = run(pd.DataFrame({"code": ["AB123", "CD456"]}), col)
    assert r[("regex", "code")]["status"] == "pass"
    _, r = run(pd.DataFrame({"code": ["AB123", "AB1234"]}), col)      # extra digit: no match
    assert r[("regex", "code")]["status"] == "fail"


def test_unique_rule():
    _, r = run(pd.DataFrame({"id": ["1", "2", "3"]}), {"name": "id", "unique": True})
    assert r[("unique", "id")]["status"] == "pass"
    _, r = run(pd.DataFrame({"id": ["1", "2", "2"]}), {"name": "id", "unique": True})
    assert r[("unique", "id")]["status"] == "fail"
    assert r[("unique", "id")]["failing_rows"] == 2                # both copies


def test_required_and_optional_columns():
    df = pd.DataFrame({"a": ["1"]})
    report, r = run(df, {"name": "a"}, {"name": "b"}, {"name": "c", "required": False})
    assert r[("required", "b")]["status"] == "fail"
    assert r[("required", "c")]["status"] == "skipped"
    assert report["column_set"]["missing"] == ["b", "c"]
    assert report["status"] == "fail"


def test_extra_columns_severity():
    df = pd.DataFrame({"a": ["1"], "surprise": ["x"]})
    report, r = run(df, {"name": "a"}, extra="warn")
    assert r[("extra_columns", None)]["status"] == "warn"
    assert report["column_set"]["extra"] == ["surprise"] and report["status"] == "pass"
    report, r = run(df, {"name": "a"}, extra="error")
    assert report["status"] == "fail"
    _, r = run(df, {"name": "a"}, extra="ignore")
    assert ("extra_columns", None) not in r


def test_table_rules():
    df = pd.DataFrame({"a": ["1", "1", "2", "3"]})                   # 1 duplicate row = 25%
    _, r = run(df, {"name": "a"}, table={"min_rows": 5, "max_duplicate_rows_pct": 10})
    assert r[("min_rows", None)]["status"] == "fail"
    assert r[("max_duplicate_rows_pct", None)]["status"] == "fail"
    _, r = run(df, {"name": "a"}, table={"min_rows": 4, "max_duplicate_rows_pct": 25})
    assert r[("min_rows", None)]["status"] == "pass"
    assert r[("max_duplicate_rows_pct", None)]["status"] == "pass"


def test_warn_rules_do_not_fail_the_dataset():
    df = pd.DataFrame({"x": ["5", "50"]})
    report, r = run(df, {"name": "x", "dtype": "number", "max": {"value": 10, "severity": "warn"}})
    assert r[("max", "x")]["status"] == "warn"
    assert report["status"] == "pass"
    assert report["counts"]["failed_warn"] == 1 and report["counts"]["failed_error"] == 0


def test_column_severity_is_the_default_for_its_rules():
    report, r = run(pd.DataFrame({"x": ["-1"]}),
                    {"name": "x", "dtype": "number", "min": 0, "severity": "warn"})
    assert r[("min", "x")]["severity"] == "warn" and report["status"] == "pass"


def test_pii_values_are_masked_in_the_report():
    df = pd.DataFrame({"email": ["aida@example.com", "bek@example.com", "not-an-email"],
                       "code": ["Z1", "Z2", "Z3"]})
    report, r = run(df, {"name": "email", "regex": r"\S+@\S+\.\w+"},
                    {"name": "code", "pii": True, "allowed_values": ["Z1"]})
    text = str(report)
    assert "not-an-email" not in text and "Z2" not in text and "Z3" not in text
    assert "'n***'" in r[("regex", "email")]["observed"]
    assert set(report["pii_columns"]) == {"email", "code"}


# ---------------------------------------------------------------- init
@pytest.fixture
def reference():
    return pd.DataFrame({
        "id": [str(i) for i in range(1, 41)],
        "age": [str(20 + i % 30) for i in range(40)],
        "score": [f"{0.5 + i / 10:.2f}" for i in range(40)],
        "branch": ["Bishkek", "Osh", "Naryn", "Osh"] * 10,
        "note": [None if i % 10 == 0 else "ok" for i in range(40)],
        "email": [f"user{i}@example.com" for i in range(40)],
    })


def test_draft_contract_from_profile(reference):
    draft = draft_contract(reference, "ref")
    cols = {c["name"]: c for c in draft["columns"]}
    assert cols["age"]["dtype"] == "integer" and cols["score"]["dtype"] == "number"
    # range 20..49 widened by 10% of 29 (2.9) -> 17..52, never below 0
    assert (cols["age"]["min"], cols["age"]["max"]) == (17, 52)
    assert cols["score"]["min"] == 0.11 and cols["score"]["max"] >= 4.4
    assert cols["branch"]["allowed_values"] == ["Bishkek", "Naryn", "Osh"]
    assert cols["age"]["nullable"] is False
    assert cols["note"]["max_missing_pct"] == 16.0                   # 10% x 1.5 + 1
    assert cols["id"]["unique"] is True
    assert cols["email"]["pii"] is True
    assert "allowed_values" not in cols["email"]                    # no PII values written
    assert cols["email"]["regex"]["severity"] == "warn"
    assert draft["table"]["min_rows"] == 4                          # 10% of 40 rows


def test_draft_roundtrips_and_reference_passes_its_own_contract(reference):
    text = contract_to_yaml(draft_contract(reference, "ref"), "ref.csv")
    assert "user1@example.com" not in text
    parsed = parse_contract(yaml.safe_load(text))
    report = validate_dataframe(reference, parsed)
    assert report["status"] == "pass", [r for r in report["results"] if r["status"] != "pass"]


def test_draft_quotes_yes_no_categories():
    df = pd.DataFrame({"y": ["yes", "no"] * 5})
    text = contract_to_yaml(draft_contract(df, "d"), "d.csv")
    parsed = parse_contract(yaml.safe_load(text))                    # no bool error
    assert parsed["columns"][0]["rules"]["allowed_values"][0] == ["no", "yes"]
