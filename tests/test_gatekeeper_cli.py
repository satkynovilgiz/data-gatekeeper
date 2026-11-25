"""gatekeeper.py CLI end to end (Sprint 7): run it like CI would and check exit codes."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
EXAMPLE = ROOT / "contracts/example.yaml"


def gk(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    """Run `python gatekeeper.py <args>` and capture its output."""
    return subprocess.run([sys.executable, str(ROOT / "gatekeeper.py"), *map(str, args)],
                          cwd=cwd, capture_output=True, text=True)


def test_validate_passing_file_exits_0_and_writes_report(tmp_path):
    r = gk("validate", FIXTURES / "orders_ok.csv", "--contract", EXAMPLE,
           "--reports-dir", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS:" in r.stdout
    report = json.loads((tmp_path / "orders_ok_contract.json").read_text())
    assert report["status"] == "pass" and report["counts"]["failed_error"] == 0


def test_validate_failing_file_exits_1_with_the_failed_rules(tmp_path):
    r = gk("validate", FIXTURES / "orders_bad.csv", "--contract", EXAMPLE,
           "--reports-dir", tmp_path)
    assert r.returncode == 1
    report = json.loads((tmp_path / "orders_bad_contract.json").read_text())
    failed = {(x["rule"], x["column"]) for x in report["results"] if x["status"] == "fail"}
    assert failed == {("unique", "order_id"), ("allowed_values", "country"),
                      ("min", "amount"), ("dtype", "paid"), ("dtype", "order_date")}
    warned = {(x["rule"], x["column"]) for x in report["results"] if x["status"] == "warn"}
    assert warned == {("regex", "email"), ("max", "amount")}
    assert "FAIL" in r.stdout and "rule" in r.stdout and "observed" in r.stdout
    # the PII email that broke the regex is masked on screen and in the file
    for text in (r.stdout, json.dumps(report)):
        assert "carlos.at.example.com" not in text


def test_bad_contract_exits_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("columns:\n  - name: a\n    dtype: int\n")
    r = gk("validate", FIXTURES / "orders_ok.csv", "--contract", bad, "--reports-dir", tmp_path)
    assert r.returncode == 2 and "dtype must be one of" in r.stderr
    assert "Traceback" not in r.stderr
    r = gk("validate", FIXTURES / "orders_ok.csv", "--contract", tmp_path / "missing.yaml")
    assert r.returncode == 2 and "not found" in r.stderr


def test_bad_input_exits_2(tmp_path):
    r = gk("validate", tmp_path / "nope.csv", "--contract", EXAMPLE)
    assert r.returncode == 2 and "data file not found" in r.stderr
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    r = gk("validate", empty, "--contract", EXAMPLE)
    assert r.returncode == 2 and "cannot read it as CSV" in r.stderr
    r = gk("validate", FIXTURES / "orders_ok.csv")                    # --contract missing
    assert r.returncode == 2


def test_init_then_validate_then_drift(tmp_path):
    ref = tmp_path / "ref.csv"
    rows = ["id,age,plan"] + [f"{i},{20 + i % 40},{'pro' if i % 3 else 'basic'}"
                              for i in range(300)]
    ref.write_text("\n".join(rows) + "\n")
    out = tmp_path / "contracts" / "people.yaml"

    r = gk("contract", "init", ref, "--name", "people", "--out", out)
    assert r.returncode == 0, r.stderr
    assert out.exists() and "plan" in out.read_text()
    assert gk("contract", "init", ref, "--out", out).returncode == 2      # no overwrite
    assert gk("contract", "init", ref, "--out", out, "--force").returncode == 0

    r = gk("validate", ref, "--contract", out, "--reports-dir", tmp_path / "rep")
    assert r.returncode == 0, r.stdout                                    # its own reference

    # same data -> no drift (exit 0); ages +30 and a new plan -> major drift (exit 1)
    r = gk("drift", ref, ref, "--contract", out, "--reports-dir", tmp_path / "rep")
    assert r.returncode == 0 and "STABLE" in r.stdout
    cur = tmp_path / "cur.csv"
    rows = ["id,age,plan"] + [f"{i},{50 + i % 40},{'pro' if i % 3 else 'gold'}"
                              for i in range(300)]
    cur.write_text("\n".join(rows) + "\n")
    r = gk("drift", ref, cur, "--contract", out, "--reports-dir", tmp_path / "rep")
    assert r.returncode == 1, r.stdout
    report = json.loads((tmp_path / "rep" / "cur_drift.json").read_text())
    assert set(report["major_columns"]) >= {"age", "plan"}
    plan = next(c for c in report["columns"] if c["column"] == "plan")
    assert plan["new_categories"] == ["gold"] and plan["missing_categories"] == ["basic"]
    # the summary table is sorted by PSI, biggest first
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("age", "plan", "id"))]
    assert lines[0].startswith(("age", "plan"))


def test_drift_bad_input_exits_2(tmp_path):
    r = gk("drift", tmp_path / "a.csv", FIXTURES / "orders_ok.csv")
    assert r.returncode == 2 and "not found" in r.stderr


def test_drift_output_hides_pii(tmp_path):
    r = gk("drift", ROOT / "data/raw/customers_pii.csv", FIXTURES / "orders_ok.csv",
           "--reports-dir", tmp_path)
    assert r.returncode in (0, 1)
    text = r.stdout + (tmp_path / "orders_ok_drift.json").read_text()
    for raw in ("aida@example.com", "bek@example.com", "hasan@example.com"):
        assert raw not in text
