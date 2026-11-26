"""A first fairness MEASUREMENT (Sprint 11): do group gaps change with the gatekeeper's cleaning?

Measurement only: nothing here changes a model, and the sensitive columns are used exactly as
before (as ordinary features). Uses the out-of-fold predictions of the impact study, so every
row is predicted once, by a model that never saw it in training, pooled over the 5 folds.

Per group of a column (e.g. sex: Female / Male):
  rows, actual positive rate, predicted positive rate (threshold 0.5), TPR (share of real
  positives predicted positive), FPR (share of real negatives predicted positive), ROC-AUC
  inside the group. Rates come with a Wilson 95% interval, which stays sensible for small
  groups.
Small groups (decided before any run): fewer than 50 positives or 50 negatives, so their TPR or
FPR rests on too few people. They are shown, flagged, and left out of the gaps.
Gap = max - min over the groups that are not small. Comparing gaps of A (raw) and B (gatekeeper)
answers one question: does the cleaning make group differences bigger or smaller? It does not
say whether a model is fair.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.5
MIN_CLASS_ROWS = 50
Z95 = 1.959964
RATES = ("predicted_positive_rate", "tpr", "fpr", "roc_auc")


def wilson(k: int, n: int) -> list[float] | None:
    """Wilson 95% interval of k successes in n trials (None for n = 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / denom
    return [round(centre - half, 4), round(centre + half, 4)]


def group_metrics(y: np.ndarray, proba: np.ndarray, groups: pd.Series,
                  threshold: float = THRESHOLD) -> list[dict]:
    """One dict per group: counts, rates with Wilson intervals, ROC-AUC, small-group flag."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(proba) >= threshold
    out = []
    for g in sorted(groups.dropna().astype(str).unique()):
        m = (groups.astype(str) == g).to_numpy()
        n, pos = int(m.sum()), int(y[m].sum())
        neg = n - pos
        tp = int((pred & m & (y == 1)).sum())
        fp = int((pred & m & (y == 0)).sum())
        pp = int((pred & m).sum())
        auc = float(roc_auc_score(y[m], proba[m])) if 0 < pos < n else None
        out.append({
            "group": g, "rows": n, "positives": pos, "negatives": neg,
            "positive_rate": round(pos / n, 4) if n else None,
            "predicted_positive_rate": round(pp / n, 4) if n else None,
            "predicted_positive_rate_ci": wilson(pp, n),
            "tpr": round(tp / pos, 4) if pos else None, "tpr_ci": wilson(tp, pos),
            "fpr": round(fp / neg, 4) if neg else None, "fpr_ci": wilson(fp, neg),
            "roc_auc": None if auc is None else round(auc, 4),
            "small": pos < MIN_CLASS_ROWS or neg < MIN_CLASS_ROWS})
    return out


def gaps(groups: list[dict]) -> dict:
    """max - min of each rate over the groups that are not small (None with < 2 groups)."""
    big = [g for g in groups if not g["small"]]
    out = {}
    for key in RATES:
        vals = [g[key] for g in big if g[key] is not None]
        out[key] = round(max(vals) - min(vals), 4) if len(vals) >= 2 else None
    out["groups_used"] = [g["group"] for g in big]
    return out


def fairness_report(y: np.ndarray, predictions: dict, data: pd.DataFrame,
                    columns: list[str]) -> dict:
    """Per column, variant and model: group table, gaps; plus B - A (and C - A) gap changes.

    predictions: {variant: {model: out-of-fold probabilities}}, rows aligned with data / y.
    """
    out: dict = {"threshold": THRESHOLD, "min_class_rows": MIN_CLASS_ROWS, "columns": {}}
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"fairness column {col!r} not found")
        per: dict = {}
        for v, by_model in predictions.items():
            for m, proba in by_model.items():
                table = group_metrics(y, np.asarray(proba), data[col])
                per.setdefault(v, {})[m] = {"groups": table, "gaps": gaps(table)}
        change = {}
        for v in per:
            if v == "A":
                continue
            for m in per[v]:
                change.setdefault(f"{v}-A", {})[m] = {
                    k: (None if per[v][m]["gaps"][k] is None or per["A"][m]["gaps"][k] is None
                        else round(per[v][m]["gaps"][k] - per["A"][m]["gaps"][k], 4))
                    for k in RATES}
        out["columns"][col] = {"by_variant": per, "gap_change": change}
    return out
