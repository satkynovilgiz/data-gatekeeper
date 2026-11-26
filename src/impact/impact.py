"""Does Data Gatekeeper's cleaning help a model? Measured with cross-validation (Sprint 8).

Three ways to prepare the same raw file, compared with the same models on the same folds:

  A raw          the minimum a model needs: numbers read as numbers, gaps filled with the
                 median / most common value, one-hot categories, numbers standardised.
                 "unknown" stays a category, pdays = 999 stays the number 999, duplicates stay.
  B cleaned      the gatekeeper: Module 1 profile + schema, cleaning steps 1-8 (special code
                 pdays 999 -> previously_contacted, disguised missing -> NaN, "default"
                 keeps "unknown", ...), duplicate rows removed from the TRAIN part, the
                 imputer chosen in Sprint 5 (iterative / MICE), and the Sprint 6 transform
                 (ordinal education, Yeo-Johnson / log, scalers, one-hot).
  C cleaned, median   as B but the gaps are filled with the median / mode, so B - C is the
                 effect of the imputer alone.

No leakage: everything that learns from data is fitted inside each fold on the train rows
only, and only applied to the test rows: the profile + schema (which values are missing
codes, which categories exist), the imputer, the encoders and scalers, the model. The test
rows are the same rows for A, B and C (B and C do not drop test rows), so the fold scores
are paired.

Models: LogisticRegression and HistGradientBoostingClassifier. 5-fold stratified CV, seed 42.
Metrics: ROC-AUC, PR-AUC (average precision), F1 at threshold 0.5, Brier score (lower = better).
Paired comparison per metric: mean fold difference with a 95% t-interval and a Wilcoxon
signed-rank p-value. With 5 folds the smallest possible two-sided Wilcoxon p is 0.0625, so
the t-interval is the main evidence; "better" is only claimed when it excludes 0.

Robustness: the models trained on clean folds (A and B) score test folds with injected
corruption (src/impact/corruption.py) at several levels.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.cleaning.cleaner import (
    DEFAULT_CONFIG,
    ROW_STEPS,
    CleaningRun,
    _strategy,
    apply_decisions,
    feature_columns,
    learn_from_train,
)
from src.cleaning.imputers import FoldImputer
from src.cleaning.schema import infer_schema
from src.impact.corruption import KINDS, corrupt
from src.profiling.profiler import profile_dataset
from src.profiling.semantic import numeric_parse_rate
from src.transform.pipeline import build_pipeline
from src.transform.plan import plan_columns

SEED = 42
FOLDS = 5
LEVELS = (0.0, 0.05, 0.10, 0.20)
NUMERIC_SHARE = 0.95
VARIANTS = {"A": "raw (median/mode + one-hot)",
            "B": "gatekeeper (clean + iterative imputer + transform)",
            "C": "gatekeeper, median imputer"}
MODELS = ("logreg", "hgb")
METRICS = ("roc_auc", "pr_auc", "f1", "brier")
LOWER_IS_BETTER = {"brier"}
PAIRS = (("B", "A"), ("C", "A"), ("B", "C"))
MIN_WILCOXON_FOLDS = 5


def make_model(name: str) -> Any:
    """A fresh, seeded model."""
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED)
    raise ValueError(f"unknown model {name!r}; use one of {MODELS}")


# ------------------------------------------------------------ preparation
def _to_numbers(X: pd.DataFrame) -> pd.DataFrame:
    return X.apply(pd.to_numeric, errors="coerce")


def numeric_columns(X: pd.DataFrame) -> list[str]:
    """Columns where >= 95% of the non-empty values read as numbers."""
    return [c for c in X.columns if numeric_parse_rate(X[c]) >= NUMERIC_SHARE]


class RawPrep:
    """Variant A: the minimum handling so a scikit-learn model can run (fit on train)."""

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn: no parameters."""
        return {}

    def set_params(self, **params: object) -> RawPrep:
        """scikit-learn: no parameters."""
        return self

    def fit(self, X: pd.DataFrame, y: object = None) -> RawPrep:
        """Decide numeric vs categorical columns and learn fills/encoders on train."""
        self.numeric_ = numeric_columns(X)
        self.categorical_ = [c for c in X.columns if c not in self.numeric_]
        self.ct_ = ColumnTransformer([
            ("num", Pipeline([("read", FunctionTransformer(_to_numbers)),
                              ("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), self.numeric_),
            ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                       sparse_output=False))]),
             self.categorical_),
        ]).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply what was learned on train."""
        return self.ct_.transform(X)


class RowCleaner:
    """Variants B/C, part 1: Module 1 profile + schema learned on train, steps 1-8b applied.

    fit() also works out, exactly as the cleaner's impute step does, which columns get
    imputed (impute_targets_) and which ones the imputer may learn from (impute_features_),
    and learns the train range + seen sentinels of every numeric column, so transform()
    turns values the train rows never had into NaN (step 8b, out_of_range_to_missing).
    ranges: {column: [min, max]} that replace the learned range (e.g. from a contract).
    value_checks: False skips the steps that remove values for being impossible or out of
    range (impossible_value_to_nan, out_of_range_to_missing): anomaly detection wants to SEE
    an age of 137, not a gap.
    """

    VALUE_CHECKS = ("impossible_value_to_nan", "out_of_range_to_missing")

    def __init__(self, config: dict | None = None,
                 ranges: dict[str, list[float]] | None = None, value_checks: bool = True) -> None:
        self.config = config
        self.ranges = ranges
        self.value_checks = value_checks

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn: the constructor arguments."""
        return {"config": self.config, "ranges": self.ranges, "value_checks": self.value_checks}

    def set_params(self, **params: object) -> RowCleaner:
        """scikit-learn: change constructor arguments."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _run(self, X: pd.DataFrame) -> CleaningRun:
        run = CleaningRun(df=X.copy(), schema=self.schema_, config=self.config_,
                          suggestions=self.suggestions_)
        for step in ROW_STEPS:
            if self.value_checks or step.__name__ not in self.VALUE_CHECKS:
                step(run)
        return run

    def fit(self, X: pd.DataFrame, y: object = None) -> RowCleaner:
        """Profile the train rows and learn the schema (no label is used)."""
        self.config_ = {**DEFAULT_CONFIG, **(self.config or {}), "learned": None}
        report = profile_dataset(X, dataset_name="train_fold")
        self.schema_ = infer_schema(X, report)
        self.decisions_log_ = apply_decisions(self.schema_, self.config_)   # Sprint 11
        self.suggestions_ = report.get("cleaning_suggestions", [])
        run = self._run(X)                       # learned=None: step 8b does nothing here
        self.columns_ = list(run.df.columns)
        self.impute_features_ = feature_columns(run)
        self.impute_targets_ = [c for c in run.df.columns if _strategy(run, c) != "none"]
        # the profiler's numeric sentinels per column (e.g. pdays 999) are candidates too
        extra = {}
        for c, col in self.schema_["columns"].items():
            codes = pd.to_numeric(pd.Series(col.get("missing_codes", []), dtype=object),
                                  errors="coerce").dropna()
            if len(codes):
                extra[c] = [float(v) for v in codes]
        self.config_["learned"] = learn_from_train(
            run.df, run.cols("numeric"), self.config_["common_sentinels"], self.ranges, extra)
        self.out_of_range_ = {}
        self.sentinel_log_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean rows with the train schema; same columns as on train.

        out_of_range_ = {column: cells set to NaN by step 8b} for this call.
        """
        run = self._run(X)
        self.sentinel_log_ = list(run.sentinel_decisions)       # code, shares, decision
        self.out_of_range_ = {}
        for st in run.steps:
            if st["step"] == "out_of_range_to_missing":
                self.out_of_range_[st["column"]] = (self.out_of_range_.get(st["column"], 0)
                                                    + st["cells_changed"])
        df = run.df.reindex(columns=self.columns_)
        for c in df.columns:              # numbers as float (Int64 / object mixes upset sklearn)
            if c in self.schema_["columns"] and \
                    self.schema_["columns"][c]["expected_type"] == "numeric":
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        return df


class PlanTransform:
    """Variants B/C, part 3: the Sprint 6 per-column plan + ColumnTransformer, fit on train."""

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn: no parameters."""
        return {}

    def set_params(self, **params: object) -> PlanTransform:
        """scikit-learn: no parameters."""
        return self

    def fit(self, X: pd.DataFrame, y: object = None) -> PlanTransform:
        """Plan every column on the train rows and fit the transformers."""
        self.plan_, self.dropped_ = plan_columns(X)
        self.ct_ = build_pipeline(self.plan_).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transform."""
        return self.ct_.transform(X)


# ---------------------------------------------------------------- metrics
def score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """ROC-AUC, PR-AUC, F1 (threshold 0.5) and Brier score for predicted probabilities."""
    return {"roc_auc": float(roc_auc_score(y_true, proba)),
            "pr_auc": float(average_precision_score(y_true, proba)),
            "f1": float(f1_score(y_true, proba >= 0.5, zero_division=0)),
            "brier": float(brier_score_loss(y_true, proba))}


def paired(a: list[float], b: list[float], lower_is_better: bool = False) -> dict:
    """Fold-by-fold difference a - b: mean, 95% t-interval, Wilcoxon p, plain verdict."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    p: float | None = None          # fewer than 5 folds: a Wilcoxon p-value says nothing
    if n >= MIN_WILCOXON_FOLDS:
        if not np.any(d != 0):
            p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # tie / zero-handling notices
                p = float(stats.wilcoxon(d).pvalue)
    lo, hi = mean - half, mean + half
    if lo > 0:
        verdict = "worse" if lower_is_better else "better"
    elif hi < 0:
        verdict = "better" if lower_is_better else "worse"
    else:
        verdict = "no clear difference"
    return {"mean_diff": mean, "ci95": [lo, hi], "wilcoxon_p": p, "folds": n,
            "verdict": verdict}


def _summary(values: list[float]) -> dict:
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "folds": [float(v) for v in a]}


# ------------------------------------------------------------------ run
@dataclass
class FoldModels:
    """Everything fitted on one train fold, reused for the robustness test."""

    raw: dict[str, Pipeline]
    cleaner: RowCleaner
    imputers: dict[str, FoldImputer]
    cleaned: dict[str, dict[str, Pipeline]]          # variant -> model -> transform+model


def positive_label(y: pd.Series, positive: str | None = None) -> str:
    """The class counted as 'positive': given, else yes / 1 / true, else the rarest."""
    values = y.dropna().astype(str)
    if positive is not None:
        if positive not in set(values):
            raise ValueError(f"positive label {positive!r} not found in the target")
        return positive
    for cand in ("yes", "1", "true", "True", "Yes"):
        if cand in set(values):
            return cand
    return str(values.value_counts().idxmin())


def _fit_fold(Xtr: pd.DataFrame, ytr: np.ndarray, models: tuple[str, ...],
              variants: tuple[str, ...]) -> tuple[FoldModels, dict]:
    t0 = time.perf_counter()
    raw = {m: Pipeline([("prep", RawPrep()), ("model", make_model(m))]).fit(Xtr, ytr)
           for m in models} if "A" in variants else {}
    timings = {"A": time.perf_counter() - t0}

    t0 = time.perf_counter()
    cleaner = RowCleaner().fit(Xtr)
    ctr = cleaner.transform(Xtr)
    dup = ctr.assign(__label__=ytr).duplicated().to_numpy()      # exact copies, label included
    ctr, ytr_c = ctr[~dup], ytr[~dup]
    prep_seconds = time.perf_counter() - t0
    imputers, cleaned = {}, {}
    for variant, method in (("B", "iterative"), ("C", "median")):
        if variant not in variants:
            continue
        t0 = time.perf_counter()
        imp = FoldImputer(method, cleaner.impute_targets_, cleaner.impute_features_)
        filled = imp.fit_transform(ctr)
        imputers[variant] = imp
        cleaned[variant] = {m: Pipeline([("transform", PlanTransform()),
                                         ("model", make_model(m))]).fit(filled, ytr_c)
                            for m in models}
        timings[variant] = prep_seconds + time.perf_counter() - t0
    info = {"train_rows": int(len(Xtr)), "train_duplicates_removed": int(dup.sum()),
            "seconds": {k: round(v, 2) for k, v in timings.items()}}
    return FoldModels(raw, cleaner, imputers, cleaned), info


def _predict(fm: FoldModels, variant: str, model: str, X: pd.DataFrame) -> np.ndarray:
    if variant == "A":
        return fm.raw[model].predict_proba(X)[:, 1]
    filled = fm.imputers[variant].transform(fm.cleaner.transform(X))
    return fm.cleaned[variant][model].predict_proba(filled)[:, 1]


def run_impact(df: pd.DataFrame, target: str, positive: str | None = None,
               folds: int = FOLDS, seed: int = SEED, models: tuple[str, ...] = MODELS,
               variants: tuple[str, ...] = tuple(VARIANTS), robustness: bool = True,
               levels: tuple[float, ...] = LEVELS, dataset: str = "dataset") -> dict:
    """Cross-validate A / B / C (and optionally the robustness test) -> report dict.

    df is the raw file read as text (read_csv_raw); rows without a target are left out.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found")
    start = time.perf_counter()
    df = df[df[target].notna()].reset_index(drop=True)
    pos = positive_label(df[target], positive)
    y = (df[target].astype(str) == pos).astype(int).to_numpy()
    X = df.drop(columns=[target])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    scores: dict[str, dict[str, dict[str, list[float]]]] = {
        v: {m: {k: [] for k in METRICS} for m in models} for v in variants}
    robust_rows: list[dict] = []
    fold_info = []
    # numpy with macOS Accelerate raises false "divide by zero / overflow in matmul" flags
    # inside the models (same as noted in src/cleaning/imputers.py); scores are finite
    with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore",
                                                invalid="ignore"):
        warnings.simplefilter("ignore", ConvergenceWarning)
        for k, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
            fm, info = _fit_fold(Xtr, ytr, models, variants)
            info.update(fold=k, test_rows=int(len(te)))
            fold_info.append(info)
            for v in variants:
                for m in models:
                    for metric, value in score(yte, _predict(fm, v, m, Xte)).items():
                        scores[v][m][metric].append(value)
            if robustness:
                robust_rows += _robustness_fold(fm, Xte, yte, k, seed, levels, models,
                                                [v for v in ("A", "B") if v in variants])

    results = {v: {m: {metric: _summary(vals) for metric, vals in ms.items()}
                   for m, ms in vm.items()} for v, vm in scores.items()}
    comparisons = {}
    for a, b in PAIRS:
        if a in variants and b in variants:
            comparisons[f"{a}-{b}"] = {
                m: {metric: paired(scores[a][m][metric], scores[b][m][metric],
                                   metric in LOWER_IS_BETTER) for metric in METRICS}
                for m in models}
    report = {
        "dataset": dataset, "target": target, "positive_label": pos,
        "rows": int(len(df)), "positive_rate": float(y.mean()),
        "folds": folds, "seed": seed, "variants": {v: VARIANTS[v] for v in variants},
        "models": list(models), "metrics": list(METRICS),
        "results": results, "comparisons": comparisons, "fold_info": fold_info,
    }
    if robustness:
        report["robustness"] = _robustness_summary(robust_rows, levels, models)
    report["seconds"] = round(time.perf_counter() - start, 1)
    return report


# ------------------------------------------------------------ robustness
def _robustness_fold(fm: FoldModels, Xte: pd.DataFrame, yte: np.ndarray, fold: int,
                     seed: int, levels: tuple[float, ...], models: tuple[str, ...],
                     variants: list[str]) -> list[dict]:
    """Score A and B on corrupted copies of one clean test fold."""
    test = Xte.reset_index(drop=True).assign(__label__=yte)
    features = list(Xte.columns)
    numeric = fm.raw[models[0]].named_steps["prep"].numeric_ if fm.raw else \
        numeric_columns(Xte)
    rows = []
    for li, level in enumerate(levels):
        for kind in ("all", *KINDS):
            if level == 0 and kind != "all":
                continue
            kinds = KINDS if kind == "all" else (kind,)
            dirty, changed = corrupt(test, level, seed + 1000 * fold + 10 * li, numeric,
                                     features, kinds)
            yd = dirty.pop("__label__").to_numpy().astype(int)
            for v in variants:
                for m in models:
                    rows.append({"fold": fold, "level": level, "kind": kind, "variant": v,
                                 "model": m, "changed": changed,
                                 **score(yd, _predict(fm, v, m, dirty))})
    return rows


def _robustness_summary(rows: list[dict], levels: tuple[float, ...],
                        models: tuple[str, ...]) -> dict:
    """Mean / std over folds per (kind, level, variant, model); drop from the clean level."""
    df = pd.DataFrame(rows)
    out = []
    keys = ["kind", "level", "variant", "model"]
    clean = df[df["level"] == 0].groupby(["variant", "model"])["roc_auc"].mean()
    for (kind, level, v, m), g in df.groupby(keys):
        out.append({"kind": kind, "level": float(level), "variant": v, "model": m,
                    "roc_auc_mean": float(g["roc_auc"].mean()),
                    "roc_auc_std": float(g["roc_auc"].std(ddof=1)),
                    "pr_auc_mean": float(g["pr_auc"].mean()),
                    "brier_mean": float(g["brier"].mean()),
                    "roc_auc_drop": float(clean[(v, m)] - g["roc_auc"].mean())})
    # level 0 belongs to every kind's curve
    zero = [r for r in out if r["level"] == 0]
    for kind in KINDS:
        out += [{**r, "kind": kind} for r in zero]
    out.sort(key=lambda r: (r["kind"], r["model"], r["variant"], r["level"]))
    return {"levels": list(levels), "kinds": ["all", *KINDS], "rows": out,
            "note": "models trained on clean train folds; only the test folds are corrupted; "
                    "'all' applies every kind at the given level"}
