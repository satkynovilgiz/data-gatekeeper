"""`gatekeeper run` (Sprint 8): config parsing, step order, gates, summary files, CLI."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.pipeline.run import ORDER, ConfigError, load_config, parse_config, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures"
CI_CONFIG = FIX / "pipeline_ci.yaml"


def config(tmp_path, steps, input_file=FIX / "orders_ok.csv", **extra):
    """A parsed config whose reports go to tmp_path."""
    return parse_config({"input": str(input_file), "reports_dir": str(tmp_path / "reports"),
                         "steps": steps, **extra})


def summary_of(out: Path) -> dict:
    return json.loads((out / "summary.json").read_text())


# ---------------------------------------------------------------- parsing
def test_example_configs_parse():
    for path in (ROOT / "pipeline.yaml", CI_CONFIG):
        cfg = load_config(path)
        assert [n for n, _ in cfg["steps"]] == list(ORDER)
    cfg = load_config(ROOT / "pipeline.yaml")
    assert cfg["name"] == "bank_marketing"
    assert dict(cfg["steps"])["impact"]["target"] == "y"


def test_steps_as_names_or_mappings_and_default_name(tmp_path):
    cfg = config(tmp_path, ["profile", {"validate": {"contract": "c.yaml"}}, {"quality": None}])
    assert cfg["steps"] == [("profile", {}), ("validate", {"contract": "c.yaml"}),
                            ("quality", {})]
    assert cfg["name"] == "orders_ok"


@pytest.mark.parametrize("raw, message", [
    ([], "must be a mapping"),
    ({"steps": ["profile"]}, "`input`"),
    ({"input": "x.csv", "steps": []}, "non-empty list"),
    ({"input": "x.csv", "steps": ["profile"], "outputs": "r"}, "unknown key"),
    ({"input": "x.csv", "steps": ["train"]}, "unknown step"),
    ({"input": "x.csv", "steps": [{"clean": {"imputr": "knn"}}]}, "unknown option"),
    ({"input": "x.csv", "steps": [{"clean": {"imputer": "magic"}}]}, "clean.imputer"),
    ({"input": "x.csv", "steps": ["validate"]}, "needs \\['contract'\\]"),
    ({"input": "x.csv", "steps": [{"impact": {"target": "y", "folds": 1}}]}, "folds"),
    ({"input": "x.csv", "steps": ["clean", "profile"]}, "must keep this order"),
    ({"input": "x.csv", "steps": ["quality", "quality"]}, "listed twice"),
    ({"input": "x.csv", "steps": [{"profile": {}, "clean": {}}]}, "a step name or"),
    ({"input": "x.csv", "steps": [{"drift": {"reference": "r", "gate": "yes"}}]},
     "drift.gate"),
    ({"input": "x.csv", "steps": [{"mask_pii": {"actions": {"email": "burn"}}}]},
     "mask_pii.actions"),
])
def test_bad_configs(raw, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_load_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("steps: [\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


# -------------------------------------------------------------------- run
def test_full_run_writes_summary_with_every_step(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)                 # the CI config uses repo-relative paths
    cfg = load_config(CI_CONFIG)
    cfg["reports_dir"] = str(tmp_path)
    code, summary, out = run_pipeline(cfg, skip={"impact"})
    assert code == 0 and summary["status"] == "passed"
    steps = [s["step"] for s in summary["steps"]]
    assert steps == list(ORDER)                                  # in the configured order
    status = {s["step"]: s["status"] for s in summary["steps"]}
    assert status == {"profile": "passed", "validate": "passed", "clean": "passed",
                      "mask_pii": "passed", "quality": "passed", "anomalies": "passed",
                      "drift": "passed", "impact": "skipped"}
    assert out.parent == tmp_path and out.name.startswith("run_")
    assert summary_of(out) == json.loads(json.dumps(summary, default=str))
    md = (out / "summary.md").read_text()
    assert "| 1 | profile | passed" in md and "| 8 | impact | skipped" in md
    assert "| 6 | anomalies | passed" in md
    for f in ("profiling_report.json", "orders_contract.json", "pii_report.json",
              "masked_data.csv", "quality.json", "orders_anomalies.json", "orders_drift.json",
              "clean/cleaned_data.csv"):
        assert (out / f).exists(), f
    assert all("seconds" in s for s in summary["steps"] if s["status"] != "skipped")
    masked = (out / "masked_data.csv").read_text()
    assert "aida@example.com" not in masked and "a***@e***.com" in masked


def test_contract_gate_stops_the_run_with_exit_1(tmp_path):
    contract = str(ROOT / "contracts/example.yaml")
    cfg = config(tmp_path, ["profile", {"validate": {"contract": contract}}, "clean", "quality"],
                 input_file=FIX / "orders_bad.csv")
    code, summary, out = run_pipeline(cfg)
    assert code == 1 and summary["status"] == "stopped at a gate"
    status = [s["status"] for s in summary["steps"]]
    assert status == ["passed", "failed", "not_run", "not_run"]
    failed = summary["steps"][1]["result"]["failed_rules"]
    assert any(r.startswith("allowed_values country") for r in failed)
    assert not (out / "clean").exists()                          # nothing ran after the gate
    assert (out / "summary.md").exists() and summary_of(out)["exit_code"] == 1


def test_drift_gate_only_stops_when_asked(tmp_path):
    shifted = tmp_path / "shifted.csv"
    rows = ["order_id,amount"] + [f"{i},{1000 + i}" for i in range(50)]
    shifted.write_text("\n".join(rows) + "\n")
    ref = tmp_path / "ref.csv"
    ref.write_text("\n".join(["order_id,amount"] + [f"{i},{i}" for i in range(50)]) + "\n")
    warn = config(tmp_path, [{"drift": {"reference": str(ref)}}], input_file=shifted)
    code, summary, _ = run_pipeline(warn)
    assert code == 0 and summary["steps"][0]["status"] == "warn"
    gate = config(tmp_path, [{"drift": {"reference": str(ref), "gate": True}}],
                  input_file=shifted)
    code, summary, _ = run_pipeline(gate)
    assert code == 1 and summary["steps"][0]["status"] == "failed"
    assert "amount" in summary["steps"][0]["result"]["major_columns"]


def test_skip_and_enabled_false(tmp_path):
    cfg = config(tmp_path, ["profile", {"clean": {"enabled": False}}, "quality"])
    code, summary, _ = run_pipeline(cfg, skip={"profile"})
    assert code == 0
    assert [s["status"] for s in summary["steps"]] == ["skipped", "skipped", "passed"]
    assert "score" in summary["steps"][2]["result"]              # quality of the input
    with pytest.raises(ConfigError, match="--skip"):
        run_pipeline(cfg, skip={"train"})


def test_a_crashing_step_is_recorded_and_exits_2(tmp_path):
    cfg = config(tmp_path, ["profile", {"validate": {"contract": str(tmp_path / "none.yaml")}},
                            "quality"])
    code, summary, out = run_pipeline(cfg)
    assert code == 2 and summary["status"] == "error"
    assert summary["steps"][1]["status"] == "error"
    assert "not found" in summary["steps"][1]["error"]
    assert summary["steps"][2]["status"] == "not_run"
    assert (out / "summary.json").exists()


def test_missing_input_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="input file not found"):
        run_pipeline(config(tmp_path, ["profile"], input_file=tmp_path / "nope.csv"))


# -------------------------------------------------------------------- CLI
def gk(*args, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "gatekeeper.py"), *map(str, args)],
                          cwd=cwd, capture_output=True, text=True)


def _ci_config_in(tmp_path: Path, **changes) -> Path:
    raw = yaml.safe_load(CI_CONFIG.read_text())
    raw["reports_dir"] = str(tmp_path / "reports")
    raw.update(changes)
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_cli_run_end_to_end(tmp_path):
    r = gk("run", "--config", _ci_config_in(tmp_path), "--skip", "impact")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASSED (exit 0)" in r.stdout
    runs = list((tmp_path / "reports").glob("run_*"))
    assert len(runs) == 1 and (runs[0] / "summary.md").exists()


def test_cli_run_gate_and_bad_config_exit_codes(tmp_path):
    bad_input = _ci_config_in(tmp_path, input="tests/fixtures/orders_bad.csv")
    r = gk("run", "--config", bad_input, "--skip", "impact")
    assert r.returncode == 1 and "FAILED" in r.stdout and "NOT_RUN" in r.stdout
    broken = tmp_path / "broken.yaml"
    broken.write_text("input: x.csv\nsteps: [clean, profile]\n")
    r = gk("run", "--config", broken)
    assert r.returncode == 2 and "must keep this order" in r.stderr
    assert "Traceback" not in r.stderr


def test_cli_impact_on_a_small_sample(tmp_path):
    r = gk("impact", "data/raw/bank_marketing_sample.csv", "--target", "y", "--folds", "2",
           "--sample", "300", "--levels", "0.1", "--reports-dir", tmp_path)
    assert r.returncode == 0, r.stderr
    name = "bank_marketing_sample_sample300"
    report = json.loads((tmp_path / f"{name}_impact.json").read_text())
    assert report["rows"] == 300 and report["robustness"]["levels"] == [0.0, 0.1]
    assert (tmp_path / f"{name}_robustness.png").stat().st_size > 1000
    assert "B-A" in r.stdout
    r = gk("impact", "data/raw/bank_marketing_sample.csv", "--target", "nope")
    assert r.returncode == 2 and "not found" in r.stderr


# ----------------------------------------------------------- anomalies (Sprint 9)
def test_anomalies_step_reports_and_gates(tmp_path):
    sample = ROOT / "data/raw/bank_marketing_sample.csv"
    cfg = config(tmp_path, ["quality", {"anomalies": {"exclude": ["y"], "top": 3}}],
                 input_file=sample)
    code, summary, out = run_pipeline(cfg)
    assert code == 0
    step = summary["steps"][1]
    assert step["step"] == "anomalies" and step["status"] == "passed"
    assert step["result"]["flagged"] == 10 and len(step["result"]["top_rows"]) == 3
    section = json.loads((out / "bank_marketing_sample_anomalies.json").read_text())
    assert "y" not in section["features"]
    assert "anomalies | passed" in (out / "summary.md").read_text()

    gate = config(tmp_path, [{"anomalies": {"exclude": ["y"], "fail_on_share": 0.005}},
                             {"drift": {"reference": str(sample)}}], input_file=sample)
    code, summary, _ = run_pipeline(gate)
    assert code == 1
    assert [s["status"] for s in summary["steps"]] == ["failed", "not_run"]


@pytest.mark.parametrize("opts, message", [
    ({"method": "lof"}, "anomalies.method"),
    ({"top": 0}, "anomalies.top"),
    ({"fail_on_share": 1.5}, "fail_on_share"),
    ({"exclude": "y"}, "anomalies.exclude"),
    ({"limit": 3}, "unknown option"),
])
def test_bad_anomaly_options(opts, message):
    with pytest.raises(ConfigError, match=message):
        parse_config({"input": "x.csv", "steps": [{"anomalies": opts}]})


def test_anomalies_must_come_after_quality_and_before_drift():
    with pytest.raises(ConfigError, match="must keep this order"):
        parse_config({"input": "x.csv", "steps": [{"drift": {"reference": "r"}}, "anomalies"]})
