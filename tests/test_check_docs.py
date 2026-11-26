"""Sprint 12: the documentation checker on small, made-up doc trees."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("checker", ROOT / "scripts/check_docs_commands.py")
checker = importlib.util.module_from_spec(spec)
sys.modules["checker"] = checker          # dataclasses look their module up here
spec.loader.exec_module(checker)


def test_commands_are_extracted_from_bash_blocks():
    text = ("intro\n```bash\npython gatekeeper.py validate a.csv \\\n   --contract c.yaml  # note\n"
            "# a comment\nmake demo\n```\n```python\nprint(1)\n```\n")
    assert checker.commands_in(text) == ["python gatekeeper.py validate a.csv --contract c.yaml",
                                         "make demo"]


def test_classify():
    c = checker.classify
    assert c("curl -F file=@x.csv http://localhost/run")[0] == "skip"
    assert c("make demo")[0] == "skip"
    assert c("python gatekeeper.py impact data/raw/adult.csv --target income")[0] == "parse"
    assert c("python gatekeeper.py run --config pipeline.yaml")[0] == "parse"
    assert c("python gatekeeper.py validate tests/fixtures/orders_ok.csv "
             "--contract contracts/example.yaml") == ("quick", "read-only, committed inputs")
    example = "python gatekeeper.py validate new.csv --contract contracts/example.yaml"
    assert c(example)[0] == "parse"
    assert c("python scripts/download_adult.py --split")[0] == "help"          # has argparse
    assert c("python scripts/evaluate_imputer_speed.py")[0] == "skip"          # would run in full
    assert c("python scripts/nope.py")[0] == "fail"
    assert c("python scripts/check_docs_commands.py")[0] == "skip"           # never runs itself


def test_parser_check_catches_wrong_options():
    assert checker.parse_gatekeeper("python gatekeeper.py decisions x.csv --target y") is None
    assert "rejected" in checker.parse_gatekeeper("python gatekeeper.py decisions x.csv --tagret y")
    assert "rejected" in checker.parse_gatekeeper("python gatekeeper.py nonsense")


def test_links_numbers_and_plan(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/sprint-01.md").write_text("ROC-AUC 0.9141 and 0.1234\n")
    (tmp_path / "docs/sprint_plan.md").write_text("### Sprint 1 ✅ — a\n### Sprint 2 ⏳ — b\n")
    readme = tmp_path / "README.md"
    readme.write_text("[ok](docs/sprint-01.md) [anchor](docs/sprint-01.md#x) [bad](docs/no.md) "
                      "[web](https://example.org)\nB 0.9141, A −0.1234, wrong 0.5555\n")
    links = checker.check_links([readme], root=tmp_path)
    assert len(links.ok) == 2 and links.failed == ["README.md -> docs/no.md"]
    assert links.skipped == ["README.md -> https://example.org  (external, not tried)"]
    numbers = checker.check_numbers(tmp_path)
    assert numbers.failed == ["0.5555"] and len(numbers.ok) == 2           # −0.1234 matches 0.1234
    plan = checker.check_plan(tmp_path)
    assert plan.ok == ["Sprint 1 ✅ — a"] and plan.failed == ["Sprint 2 ⏳ — b"]
