"""Sprint 3 — per-column charts: histogram per numeric column, top-10 bar chart per categorical."""
import pandas as pd

from src.profiling.profiler import profile_dataset
from src.profiling.visualize import make_all, safe_file_name


def _df():
    return pd.DataFrame({
        "age": [str(20 + i % 40) for i in range(60)],
        "pdays": ["999"] * 50 + [str(i) for i in range(10)],     # has disguised missing
        "job": (["admin", "tech", "unknown"] * 20),
        "emp.var.rate": [str(1.1 - i * 0.05) for i in range(60)],
    })


def test_one_chart_per_column(tmp_path):
    df = _df()
    charts = make_all(df, profile_dataset(df), tmp_path)
    assert set(charts["histograms"]) == {"age", "pdays", "emp.var.rate"}
    assert charts["bar_charts"] == {"job": "bar_job.png"}
    # every file named in the report really exists
    names = list(charts["histograms"].values()) + list(charts["bar_charts"].values())
    names += [charts[k] for k in ("missing_heatmap", "correlation_heatmap",
                                  "outlier_distribution")]
    for name in names:
        assert (tmp_path / name).stat().st_size > 0


def test_safe_file_name():
    assert safe_file_name("emp.var.rate") == "emp.var.rate"
    assert safe_file_name("my col/x") == "my_col_x"
    assert safe_file_name("***") == "column"


def test_no_numeric_columns(tmp_path):
    df = pd.DataFrame({"city": ["A", "B", "A"]})
    charts = make_all(df, profile_dataset(df), tmp_path)
    assert charts["histograms"] == {}
    assert "correlation_heatmap" not in charts and "outlier_distribution" not in charts
