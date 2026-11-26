"""Score the three anomaly methods on the planted anomalies (Sprint 9).

    python scripts/make_anomaly_demo.py && python scripts/evaluate_anomalies.py

SETTINGS, FIXED BEFORE THE FIRST RUN (committed before any result existed; never tuned):
  - methods with their defaults: iforest (100 trees, seed 42, contamination 0.01),
    zscore (robust z, MAD = 0 columns unused), iqr (fences 1.5 x IQR)
  - Sprint 10 adds, with settings fixed in src/validation/anomaly.py before its first run:
    zscore_mixture (1-3 Gaussian bumps per column by BIC, seed 42, bump SD >= 0.1 x scale,
    flag at 3.5) and iforest_with_categories (frequency encoding, same forest settings).
    The planted data (scripts/make_anomaly_demo.py, seed 42) is the Sprint 9 file, unchanged.
  - features chosen by src/validation/anomaly.py (label y excluded), fitted on the file itself
  - overall: k = number of planted rows (412); per kind: the normal rows + that kind's rows,
    k = that kind's count; precision@k, recall@k, ROC-AUC, PR-AUC
  - runtime = scoring time of each method on all 41,188 rows (shared preparation apart)
Writes reports/anomaly_evaluation.json and .md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.profiling.profiler import read_csv_raw  # noqa: E402
from src.validation.anomaly import METHODS, detect, evaluate  # noqa: E402
from src.validation.anomaly_inject import KINDS  # noqa: E402


def main() -> None:
    """Run every method once, then score it overall and per kind."""
    demo = ROOT / "data/demo"
    df = read_csv_raw(demo / "bank_anomalies.csv")
    truth = pd.read_csv(demo / "bank_anomalies_labels.csv")
    report = detect(df, methods=METHODS, top=5, exclude=["y"], keep_scores=True)
    scores = report.pop("_scores")
    label = pd.Series(0, index=df.index)
    label[truth["row"]] = 1
    kind = pd.Series("normal", index=df.index)
    kind[truth["row"]] = truth["kind"].to_numpy()

    rows = []
    for m in METHODS:
        s = scores[m]
        for group in ("all", *KINDS):
            keep = kind.isin(["normal", group]) if group != "all" else kind.notna()
            res = evaluate(s[keep], label[keep])
            rows.append({"method": m, "kind": group, "planted": int(label[keep].sum()), **res,
                         "seconds": report["methods"][m]["seconds"]})
    out = {"rows": len(df), "planted": int(label.sum()), "prepare_seconds":
           report["prepare_seconds"], "features": report["features"], "results": rows,
           "flagged": {m: report["methods"][m]["flagged"] for m in METHODS},
           "top": {m: report["methods"][m]["top"] for m in METHODS},
           "top_truth": {m: [str(kind[t["row"]]) for t in report["methods"][m]["top"]]
                         for m in METHODS}}
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "anomaly_evaluation.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["| method | kind | planted | k | precision@k | recall@k | ROC-AUC | PR-AUC | "
             "seconds |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['kind']} | {r['planted']} | {r['k']} | "
                     f"{r['precision_at_k']:.4f} | {r['recall_at_k']:.4f} | {r['roc_auc']:.4f} | "
                     f"{r['pr_auc']:.4f} | {r['seconds']} |")
    table = "\n".join(lines)
    (reports / "anomaly_evaluation.md").write_text(table + "\n")
    print(table)
    print(f"\nprepare {report['prepare_seconds']} s; flagged: {out['flagged']}")
    for m in METHODS:
        print(m, "top-5 truth:", out["top_truth"][m])


if __name__ == "__main__":
    main()
