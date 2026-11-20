"""Imputer evaluation (src/cleaning/evaluation.py)."""
import json

import numpy as np
import pandas as pd

from src.cleaning.cleaner import clean_file
from src.cleaning.evaluation import evaluate_imputers, hide_cells, save_evaluation


def related(n: int = 300) -> pd.DataFrame:
    """No missing values. y = 2x + noise, zone depends on x: ML should beat the median."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 100, n)
    return pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 2, n),
                         "zone": np.where(x < 50, "low", "high")})


def test_hide_cells_hides_ten_percent_with_a_fixed_seed():
    df = related()
    masked, hidden = hide_cells(df, ["x", "y", "zone"])
    assert hidden.to_numpy().sum() == 90                      # 10% of 300 x 3 cells
    assert masked.isna().to_numpy().sum() == 90
    _, again = hide_cells(df, ["x", "y", "zone"])
    assert again.equals(hidden)                                # same seed -> same cells
    assert df.notna().all().all()                              # the input is not changed


def test_evaluation_fills_hidden_values_and_ranks_methods():
    df = related()
    result = evaluate_imputers(df, ["x", "y", "zone"], ["x", "y", "zone"])
    assert result["rows_used"] == 300 and result["hidden_cells"] == 90
    for r in result["methods"].values():
        assert r["seconds"] >= 0
        y = r["columns"]["y"]
        assert y["hidden"] > 0 and y["rmse"] >= y["mae"] >= 0  # RMSE >= MAE always
        assert 0 <= r["columns"]["zone"]["accuracy"] <= 1
    m = result["methods"]
    # y is predictable from x: every ML method beats the median baseline
    for method in ("knn", "regression", "iterative"):
        assert m[method]["mean_nrmse"] < m["median"]["mean_nrmse"]
    # zone follows x too: every ML method beats the mode
    for method in ("knn", "regression", "iterative"):
        assert m[method]["mean_accuracy"] > m["median"]["mean_accuracy"]
    assert result["best"]["numeric"] in ("knn", "regression", "iterative")
    assert result["best"]["categorical"] is not None


def test_evaluation_uses_complete_rows_only_and_caps_rows():
    df = related()
    df.loc[:49, "y"] = np.nan                                  # 50 incomplete rows
    result = evaluate_imputers(df, ["y"], ["x", "y"], max_rows=100)
    assert result["rows_complete"] == 250
    assert result["rows_used"] == 100


def test_save_evaluation_writes_json_and_chart(tmp_path):
    result = evaluate_imputers(related(), ["y", "zone"], ["x", "y", "zone"])
    files = save_evaluation(result, tmp_path, title="related.csv")
    assert json.loads((tmp_path / "imputer_evaluation.json").read_text())["best"]
    assert (tmp_path / "imputer_evaluation.png").stat().st_size > 10_000
    assert set(files) == {"json", "chart"}


def test_pipeline_runs_evaluation_when_asked(tmp_path):
    csv = tmp_path / "hr.csv"
    rows = [f"{25 + i % 30},{3 + i % 30},{['Sales', 'HR'][i % 2]}" for i in range(80)]
    csv.write_text("age,years_experience,dept\n" + "\n".join(rows) + "\n")
    log = clean_file(csv, tmp_path / "out", config={"evaluate_imputers": True})
    ev = log["imputer_evaluation"]
    assert set(ev["methods"]) == {"median", "knn", "regression", "iterative"}
    assert "columns" not in ev["methods"]["knn"]               # details live in the json file
    assert (tmp_path / "out" / "imputer_evaluation.png").exists()
    # without the option nothing is evaluated
    assert "imputer_evaluation" not in clean_file(csv, tmp_path / "out2")
