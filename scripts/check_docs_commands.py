"""Check that the documentation still matches the code (Sprint 12).

    python scripts/check_docs_commands.py            # exit 0 = everything checked is fine
    python scripts/check_docs_commands.py --online   # also try the external links

What it checks, in README.md and every docs/*.md:
  1. commands in ```bash blocks:
       - every `python gatekeeper.py ...` is parsed by the real argument parser (sub-command
         and options must exist) without running it
       - QUICK commands are run for real: read-only gatekeeper commands whose input files are
         committed, with --reports-dir sent to a temporary folder; and `python <script> --help`
         for scripts that use argparse (other scripts would run in full: only checked to exist)
       - the rest is listed as SKIPPED with the reason (needs downloaded data, Docker, a server,
         network, or minutes of run time)
  2. links: every relative [text](path) must point to an existing file (anchors ignored);
     external links are listed (tried only with --online)
  3. numbers: every number with 4 decimals in README.md must appear in a sprint report or a
     presentation (so the README never shows a number its source does not have)
  4. docs/sprint_plan.md: no sprint still marked ⏳ or 🔄
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUICK_SUBCOMMANDS = {"validate", "drift", "anomalies", "decisions", "report"}
SLOW_SUBCOMMANDS = {"impact": "takes minutes (cross-validation)",
                    "run": "runs a whole pipeline (minutes on real data)",
                    "contract": "writes a contract file into the repo"}
DOWNLOADED = ("data/raw/bank_marketing.csv", "data/raw/adult.csv", "data/demo/")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
NUMBER = re.compile(r"(?<![\w.])[−-]?\d+\.\d{4}(?!\d)")


@dataclass
class Result:
    """What the checker found."""

    ok: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def doc_files(root: Path = ROOT) -> list[Path]:
    """README.md and every markdown file in docs/."""
    return [root / "README.md", *sorted((root / "docs").glob("*.md"))]


def commands_in(text: str) -> list[str]:
    """Shell commands from ```bash blocks (continuation lines joined, comments dropped)."""
    out = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        block = re.sub(r"\\\n\s*", " ", block)
        for line in block.splitlines():
            line = re.sub(r"\s+", " ", re.sub(r"\s+#.*$", "", line)).strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def classify(cmd: str, root: Path = ROOT) -> tuple[str, str]:
    """('quick' | 'parse' | 'skip', reason) for one command."""
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return "skip", "not a simple command"
    if not argv:
        return "skip", "empty"
    head = argv[0]
    if head in ("curl", "docker", "uvicorn", "export") or "=" in head:
        return "skip", f"needs {'a running server' if head == 'curl' else head}"
    if head in ("make", "pytest", "pip", "source", "bash", "ruff"):
        return "skip", f"`{head}` is run by CI / the test suite itself"
    if head not in ("python", "python3") or len(argv) < 2:
        return "skip", "not a project command"
    script = argv[1]
    if script == "-m":
        return "skip", "module run (covered by CI)"
    if not (root / script).exists():
        return "fail", f"script {script} does not exist"
    if script != "gatekeeper.py":
        # only scripts with argparse answer --help; others would run in full (minutes,
        # and some rewrite data/demo/), so they are only checked to exist
        if re.search(r"^import argparse", (root / script).read_text(encoding="utf-8"), re.M):
            return "help", "script: run with --help"
        return "skip", "script without --help: runs in full (exists)"
    sub = argv[2] if len(argv) > 2 else ""
    if any(d in cmd for d in DOWNLOADED):
        return "parse", "needs downloaded data (parsed only)"
    if sub in SLOW_SUBCOMMANDS:
        return "parse", SLOW_SUBCOMMANDS[sub] + " (parsed only)"
    if sub in QUICK_SUBCOMMANDS and "--write" not in argv and "<" not in cmd:
        files = [a for a in argv[3:] if not a.startswith("-") and (
            "/" in a or a.endswith((".csv", ".yaml", ".yml", ".json")))]
        if all((root / f).exists() for f in files):
            return "quick", "read-only, committed inputs"
        return "parse", "example file names, not in the repo (parsed only)"
    return "parse", "parsed only"


def parse_gatekeeper(cmd: str) -> str | None:
    """None when the real parser accepts the command, else the error text."""
    from gatekeeper import build_parser

    argv = shlex.split(cmd)[2:]
    argv = [a for a in argv if not a.startswith("<")]
    try:
        build_parser().parse_args(argv)
    except SystemExit as e:
        return None if e.code == 0 else f"parser rejected: {' '.join(argv)}"
    return None


def run_quick(cmd: str, root: Path = ROOT) -> str | None:
    """Run a quick command with reports in a temporary folder; None when it exits 0 or 1."""
    argv = shlex.split(cmd)
    with tempfile.TemporaryDirectory() as tmp:
        if argv[2] in ("validate", "drift", "anomalies"):
            argv = [a for i, a in enumerate(argv) if not (a == "--reports-dir" or (
                i > 0 and argv[i - 1] == "--reports-dir"))] + ["--reports-dir", tmp]
        r = subprocess.run([sys.executable, *argv[1:]], cwd=root, capture_output=True,
                           text=True, timeout=300)
    # exit 1 = the data failed a gate on purpose (e.g. the bad fixture): the command works
    return None if r.returncode in (0, 1) else f"exit {r.returncode}: {r.stderr.strip()[-200:]}"


def check_commands(files: list[Path], root: Path = ROOT) -> Result:
    """Check every command of every file."""
    res = Result()
    seen = set()
    for f in files:
        for cmd in commands_in(f.read_text(encoding="utf-8")):
            if cmd in seen:
                continue
            seen.add(cmd)
            where = f"{f.relative_to(root)}: {cmd}"
            kind, why = classify(cmd, root)
            if kind == "fail":
                res.failed.append(f"{where}  ({why})")
            elif kind == "skip":
                res.skipped.append(f"{where}  ({why})")
            elif kind == "help":
                r = subprocess.run([sys.executable, shlex.split(cmd)[1], "--help"], cwd=root,
                                   capture_output=True, text=True, timeout=120)
                (res.ok if r.returncode == 0 else res.failed).append(f"{where}  (--help)")
            else:
                err = parse_gatekeeper(cmd)
                if err:
                    res.failed.append(f"{where}  ({err})")
                elif kind == "quick":
                    err = run_quick(cmd, root)
                    (res.failed if err else res.ok).append(
                        f"{where}  ({err or 'ran'})")
                else:
                    res.ok.append(f"{where}  (parsed; {why})")
    return res


def check_links(files: list[Path], online: bool = False, root: Path = ROOT) -> Result:
    """Relative links must exist; external ones are tried only when online."""
    res = Result()
    for f in files:
        for target in LINK.findall(f.read_text(encoding="utf-8")):
            where = f"{f.relative_to(root)} -> {target}"
            if target.startswith(("http://", "https://")):
                if not online:
                    res.skipped.append(f"{where}  (external, not tried)")
                    continue
                try:
                    req = urllib.request.Request(target, method="HEAD",
                                                 headers={"User-Agent": "docs-check"})
                    urllib.request.urlopen(req, timeout=20)
                    res.ok.append(where)
                except Exception as e:  # noqa: BLE001  any network problem is reported
                    res.failed.append(f"{where}  ({type(e).__name__})")
                continue
            if target.startswith(("#", "mailto:")):
                continue
            path = (f.parent / target.split("#")[0]).resolve()
            (res.ok if path.exists() else res.failed).append(where)
    return res


def check_numbers(root: Path = ROOT) -> Result:
    """Every 4-decimal number in the README appears in a sprint report or presentation."""
    res = Result()
    sources = "\n".join(p.read_text(encoding="utf-8") for p in sorted((root / "docs").glob("*.md"))
                        if p.name.startswith(("sprint", "module", "final")))
    sources = sources.replace("−", "-")
    for n in sorted(set(NUMBER.findall((root / "README.md").read_text(encoding="utf-8")))):
        plain = n.replace("−", "-")
        (res.ok if plain.lstrip("-") in sources else res.failed).append(n)
    return res


def check_plan(root: Path = ROOT) -> Result:
    """Every sprint heading in the plan is ✅."""
    res = Result()
    for line in (root / "docs/sprint_plan.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("### Sprint"):
            (res.ok if "✅" in line else res.failed).append(line.strip("# ").strip())
    return res


def main() -> int:
    """Run all checks, print a summary, exit 1 when something failed."""
    files = doc_files()
    parts = {"commands": check_commands(files), "links": check_links(files, "--online" in sys.argv),
             "numbers": check_numbers(), "plan": check_plan()}
    failed = 0
    for name, res in parts.items():
        print(f"\n## {name}: {len(res.ok)} ok, {len(res.skipped)} skipped, "
              f"{len(res.failed)} failed")
        for line in res.failed:
            print(f"  FAILED  {line}")
        if name == "commands":
            for line in res.skipped:
                print(f"  skipped {line}")
        failed += len(res.failed)
    print(f"\n{'OK' if not failed else 'PROBLEMS'}: {failed} failed check(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        sys.exit(0)
    sys.exit(main())
