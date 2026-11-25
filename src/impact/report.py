"""Impact report as markdown tables and the robustness line chart (Sprint 8)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.impact.impact import LOWER_IS_BETTER  # noqa: E402

METRIC_NAMES = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1 @0.5", "brier": "Brier ↓"}
MODEL_NAMES = {"logreg": "LogisticRegression", "hgb": "HistGradientBoosting"}


def _pm(s: dict) -> str:
    return f"{s['mean']:.4f} ± {s['std']:.4f}"


def results_table(report: dict) -> str:
    """CV metrics, mean ± std over folds, one row per model x variant."""
    head = "| model | variant | " + " | ".join(METRIC_NAMES[m] for m in report["metrics"]) + " |"
    lines = [head, "|" + "---|" * (2 + len(report["metrics"]))]
    for m in report["models"]:
        for v, label in report["variants"].items():
            cells = [_pm(report["results"][v][m][k]) for k in report["metrics"]]
            lines.append(f"| {MODEL_NAMES.get(m, m)} | {v}: {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison_table(report: dict) -> str:
    """Paired fold differences with the 95% CI, Wilcoxon p and a verdict."""
    lines = ["| comparison | model | metric | mean diff | 95% CI | Wilcoxon p | verdict |",
             "|---|---|---|---|---|---|---|"]
    for pair, by_model in report["comparisons"].items():
        for m, by_metric in by_model.items():
            for k, c in by_metric.items():
                p = "-" if c["wilcoxon_p"] is None else f"{c['wilcoxon_p']:.4f}"
                lines.append(f"| {pair} | {MODEL_NAMES.get(m, m)} | {METRIC_NAMES[k]} | "
                             f"{c['mean_diff']:+.4f} | [{c['ci95'][0]:+.4f}, "
                             f"{c['ci95'][1]:+.4f}] | {p} | {c['verdict']} |")
    brier = ("Brier: lower is better, so a negative difference is an improvement. "
             if "brier" in LOWER_IS_BETTER else "")
    note = ("\nDifference = first minus second variant, per fold, then averaged. " + brier +
            f"With {report['folds']} folds the smallest two-sided Wilcoxon p is "
            f"{2 / 2 ** report['folds']:.4f}.")
    return "\n".join(lines) + note


def robustness_table(report: dict, kind: str = "all") -> str:
    """ROC-AUC per corruption level for A and B (mean over folds, drop vs. clean)."""
    rob = report["robustness"]
    rows = [r for r in rob["rows"] if r["kind"] == kind]
    lines = ["| model | variant | " + " | ".join(f"{lv:.0%}" for lv in rob["levels"]) +
             " | drop at max |", "|" + "---|" * (3 + len(rob["levels"]))]
    for m in report["models"]:
        for v in ("A", "B"):
            mine = {r["level"]: r for r in rows if r["model"] == m and r["variant"] == v}
            if not mine:
                continue
            cells = [f"{mine[lv]['roc_auc_mean']:.4f}" for lv in rob["levels"]]
            drop = mine[max(rob["levels"])]["roc_auc_drop"]
            lines.append(f"| {MODEL_NAMES.get(m, m)} | {v} | " + " | ".join(cells) +
                         f" | {-drop:+.4f} |")
    return "\n".join(lines)


def to_markdown(report: dict) -> str:
    """The whole impact report as markdown."""
    parts = [f"# Impact: {report['dataset']} (target `{report['target']}` = "
             f"{report['positive_label']!r}, {report['positive_rate']:.2%} positive)",
             f"{report['rows']} rows, {report['folds']}-fold stratified CV, seed {report['seed']}, "
             f"run time {report['seconds']} s.", "", "## Cross-validated metrics (mean ± std)",
             results_table(report), "", "## Paired comparison", comparison_table(report)]
    if "robustness" in report:
        parts += ["", "## Robustness: ROC-AUC on corrupted test folds (all corruption kinds)",
                  robustness_table(report, "all")]
        for kind in report["robustness"]["kinds"][1:]:
            parts += ["", f"### only {kind}", robustness_table(report, kind)]
    return "\n".join(parts) + "\n"


def robustness_chart(report: dict, path: str | Path) -> Path:
    """Line chart: ROC-AUC vs corruption level, A dashed / B solid, one colour per model."""
    rob = report["robustness"]
    rows = [r for r in rob["rows"] if r["kind"] == "all"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colours = {"logreg": "#3b6fb6", "hgb": "#d9822b"}
    for m in report["models"]:
        for v, style in (("A", "--"), ("B", "-")):
            pts = sorted((r["level"], r["roc_auc_mean"]) for r in rows
                         if r["model"] == m and r["variant"] == v)
            if pts:
                ax.plot([p[0] * 100 for p in pts], [p[1] for p in pts], style, marker="o",
                        color=colours.get(m), label=f"{MODEL_NAMES.get(m, m)}, "
                        f"{'A raw' if v == 'A' else 'B gatekeeper'}")
    ax.set_xlabel("corrupted share of test cells / rows (%)")
    ax.set_ylabel("ROC-AUC (mean of folds)")
    ax.set_title(f"{report['dataset']}: robustness to dirty test data")
    ax.set_xticks([lv * 100 for lv in rob["levels"]])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
