"""Sprint 12: the demo script in --fixture mode (no network), as a smoke test."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demo_fixture_mode_writes_two_reports():
    before = set((ROOT / "reports/demo").glob("run_*"))
    r = subprocess.run(["bash", str(ROOT / "scripts/demo.sh"), "--fixture", "--local"],
                       cwd=ROOT, capture_output=True, text=True, env={
                           "PATH": "/usr/bin:/bin", "PYTHON": sys.executable}, timeout=600)
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert "2-3. --fixture: no download" in r.stdout and "done locally" in r.stdout
        paths = re.findall(r"^\s+(reports/demo/run_\S+/report\.html)\s+\(exit 0\)$", r.stdout, re.M)
        assert len(paths) == 2
        for p in paths:
            page = (ROOT / p).read_text()
            assert "This file passed every check that was run" in page
    finally:
        after = set((ROOT / "reports/demo").glob("run_*"))
        for d in after - before:
            shutil.rmtree(d, ignore_errors=True)


def test_demo_rejects_unknown_options():
    r = subprocess.run(["bash", str(ROOT / "scripts/demo.sh"), "--fast"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "unknown option" in r.stderr


def test_demo_is_wired_into_make_and_docker():
    assert "demo:" in (ROOT / "Makefile").read_text()
    assert "pipeline_adult.yaml" in (ROOT / "Dockerfile").read_text()
    assert "demo.sh --fixture" in (ROOT / ".github/workflows/tests.yml").read_text()
