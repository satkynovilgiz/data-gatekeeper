"""An independent test of the decision detector on a dataset this project never used (Sprint 12).

    python scripts/independent_decision_test.py

Sprint 11 could not test the detector independently: its Adult decisions came from Sprint 10's
ablation on the same data. Here the data is NEW: UCI Mushroom (agaricus-lepiota, 8,124 rows,
target `class` = edible / poisonous). Its description file documents "?" as the missing code
(in stalk-root); nothing else about the data was looked at before this file was committed.

SUCCESS CRITERIA, WRITTEN AND COMMITTED BEFORE THE DATA WAS DOWNLOADED:
  1. Split the rows at random into two halves (seed 42). Run the detector (src/profiling/
     decisions.py, its unchanged Sprint 11 rule) with the target on half A only.
  2. A listed decision HOLDS UP when half B, which the detector never saw, also shows a
     clearly different label rate (> 5 points, p < 0.01, >= 30 rows) in the SAME direction.
  3. Outcome: SUCCESS = at least one decision listed on A and all of them hold up on B;
     PARTIAL = some hold up; FAIL = none holds up; NOTHING = no decision listed on A
     (reported as "no evidence either way", not as a success).
The "positive" class is `p` (poisonous). The download's SHA-256 is printed and stored, not
pinned (it was unknown before downloading). Data goes to data/raw/mushroom.csv (not in git).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.profiling.decisions import DECISION_RULE, _evidence, find_decisions  # noqa: E402
from src.profiling.profiler import profile_dataset  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/73/mushroom.zip"
COLUMNS = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
           "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
           "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
           "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
           "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
OUT = ROOT / "data/raw/mushroom.csv"
SEED = 42


def download() -> tuple[pd.DataFrame, str]:
    """The data as text (no header in the source) and the zip's SHA-256."""
    with urllib.request.urlopen(URL, timeout=60) as resp:
        data = resp.read()
    raw = zipfile.ZipFile(io.BytesIO(data)).read("agaricus-lepiota.data").decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), header=None, names=COLUMNS, dtype=str,
                     keep_default_na=False)
    return df, hashlib.sha256(data).hexdigest()


def main() -> None:
    """Download, split, detect on A, check on B, print the outcome."""
    df, digest = download()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    half_a = df.sample(frac=0.5, random_state=SEED)
    half_b = df.drop(half_a.index)
    found = find_decisions(half_a, profile_dataset(half_a, "mushroom_half_a"), target="class",
                           positive="p")
    y_b = (half_b["class"] == "p").astype(float)
    checks = []
    for d in found:
        rows_b = half_b[d["column"]].astype(str).str.strip() == d["value"]
        ev_b = _evidence(rows_b, y_b, DECISION_RULE)
        same_dir = (d["diff_pp"] or 0) * (ev_b["diff_pp"] or 0) > 0
        checks.append({"decision": {k: d[k] for k in ("column", "value", "kind", "rows",
                                                      "label_rate_in", "label_rate_rest",
                                                      "diff_pp", "p_value", "recommended")},
                       "half_b": {k: ev_b[k] for k in ("rows", "label_rate_in", "label_rate_rest",
                                                       "diff_pp", "p_value", "clearly_different")},
                       "holds_up": bool(ev_b["clearly_different"] and same_dir)})
    held = sum(c["holds_up"] for c in checks)
    outcome = ("NOTHING" if not checks else "SUCCESS" if held == len(checks)
               else "PARTIAL" if held else "FAIL")
    out = {"source": URL, "zip_sha256": digest, "rows": len(df), "half_a": len(half_a),
           "half_b": len(half_b), "positive": "p",
           "positive_rate": round(float((df["class"] == "p").mean()), 4),
           "codes_seen_by_profiler": {c: v for c, v in (
               (col["name"], col["disguised_missing"]["values"])
               for col in profile_dataset(half_a, "a")["columns"]) if v},
           "checks": checks, "outcome": outcome}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/independent_decision_test.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
