"""run_profiling.py CLI — run it like a user would, on a tiny CSV in a temp folder."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run `python run_profiling.py <args>` from the repo root and capture its output."""
    return subprocess.run([sys.executable, "run_profiling.py", *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli_writes_report_and_charts(tmp_path):
    csv = tmp_path / "tiny.csv"
    csv.write_text("id,age,city\n"
                   "1,25,London\n2,31,Paris\n3,unknown,Paris\n"
                   "4,40,London\n5,28,Rome\n6,35,Paris\n")
    out_dir = tmp_path / "out"

    result = run_cli("--input", str(csv), "--output", str(out_dir))

    assert result.returncode == 0, result.stderr
    report_file = out_dir / "profiling_report.json"
    assert report_file.exists()
    report = json.loads(report_file.read_text())
    assert report["dataset"]["name"] == "tiny.csv"
    assert report["dataset"]["rows"] == 6
    assert (out_dir / "charts" / report["charts"]["missing_heatmap"]).exists()
    assert "Report ->" in result.stdout and "Quality score" in result.stdout


def test_cli_fails_for_missing_file(tmp_path):
    result = run_cli("--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path))
    assert result.returncode != 0
    assert not (tmp_path / "profiling_report.json").exists()
