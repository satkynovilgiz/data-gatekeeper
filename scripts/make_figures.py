"""Make the Module 2 presentation charts in docs/figures/ from the real result files.

Run first (they write the inputs this script reads):
    python run_profiling.py / run_cleaning.py for results/bank and results/hr
    python run_cleaning.py ... --evaluate-imputers   (results/bank/imputer_evaluation.json)
    python run_masking.py for results/customers and results/hr/privacy
Then:
    python scripts/make_figures.py

Out: docs/figures/quality_scores.png, imputer_comparison.png, pii_columns.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.profiling.profiler import profile_dataset, read_csv_raw  # noqa: E402
from src.quality.score import quality_score  # noqa: E402

OUT = ROOT / "docs" / "figures"
RESULTS = ROOT / "results"
# the same validated categorical palette as the Sprint 5 imputer chart, in fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#222222", "#666666", "#e5e5e5"


def tidy(ax):
    """Recessive axes: light grid, no box."""
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def quality_chart():
    """Bars per dataset: score before cleaning, after (Sprint 5 way), after (Sprint 6 fix)."""
    rows = []
    for name, label in [("hr", "HR sample"), ("bank", "Bank Marketing")]:
        log = json.loads((RESULTS / name / "cleaning_log.json").read_text())
        after = profile_dataset(read_csv_raw(RESULTS / name / "cleaned_data.csv"))
        rows.append((label, log["quality"]["before"]["overall"],
                     quality_score(after)["overall"],               # flags counted (old)
                     log["quality"]["after"]["overall"]))          # flags left out (new)
    series = ["before cleaning", "after, flags counted (Sprint 5)",
              "after, flags left out (Sprint 6)"]
    fig, axes = plt.subplots(1, len(rows), figsize=(9, 2.6), sharey=True)
    for ax, (label, *scores) in zip(axes, rows):
        bars = ax.barh(series[::-1], scores[::-1], height=0.6, color=PALETTE[:3][::-1])
        for b, s in zip(bars, scores[::-1]):
            ax.annotate(f"{s:.2f}", (b.get_width(), b.get_y() + b.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
        ax.set_xlim(0, 112)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        tidy(ax)
    fig.suptitle("Quality score (0-100) before and after cleaning", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "quality_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


def imputer_chart():
    """Two panels for Bank Marketing: numeric error (NRMSE) and category accuracy."""
    ev = json.loads((RESULTS / "bank" / "imputer_evaluation.json").read_text())
    methods = list(ev["methods"])
    nrmse = [ev["methods"][m]["mean_nrmse"] for m in methods]
    acc = [ev["methods"][m]["mean_accuracy"] * 100 for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, values, title, fmt in [
            (axes[0], nrmse, "Numeric error, mean NRMSE (lower is better)", "{:.3f}"),
            (axes[1], acc, "Category accuracy % (higher is better)", "{:.1f}%")]:
        bars = ax.barh(methods[::-1], values[::-1], height=0.6,
                       color=PALETTE[:len(methods)][::-1])
        for b, v in zip(bars, values[::-1]):
            ax.annotate(fmt.format(v), (b.get_width(), b.get_y() + b.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, max(values) * 1.2)
        tidy(ax)
    fig.suptitle(f"Imputers on Bank Marketing: {ev['hidden_cells']:,} hidden known values",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "imputer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(zip(methods, zip(nrmse, acc)))


def pii_chart():
    """Stacked bars: PII columns found per dataset, split by the action taken."""
    actions = ["hash", "mask", "generalize", "drop"]
    data = [("Bank Marketing", None), ("HR sample", RESULTS / "hr" / "privacy"),
            ("Fake customers", RESULTS / "customers")]
    counts = []
    for _label, folder in data:
        report = json.loads((folder / "pii_report.json").read_text()) if folder else \
            {"columns": []}                     # Bank: checked by test_bank_marketing_has_no_pii
        counts.append([sum(r["action"] == a for r in report["columns"]) for a in actions])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    left = [0] * len(data)
    for k, a in enumerate(actions):
        widths = [c[k] for c in counts]
        ax.barh([d[0] for d in data], widths, left=left, height=0.55, color=PALETTE[k],
                edgecolor="white", linewidth=2, label=a)
        left = [lo + w for lo, w in zip(left, widths)]
    for i, total in enumerate(left):
        ax.annotate(f"{total} column{'s' if total != 1 else ''}", (total, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(left) + 2)
    ax.set_title("PII columns found, by action taken", loc="left", fontsize=11, color=INK)
    tidy(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, -0.45))
    fig.tight_layout()
    fig.savefig(OUT / "pii_columns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(zip([d[0] for d in data], counts))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("quality:", quality_chart())
    print("imputers:", imputer_chart())
    print("pii:", pii_chart())
    print(f"charts -> {OUT}")
