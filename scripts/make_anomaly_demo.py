"""Make the Sprint 9 anomaly demo: bank_marketing.csv with 1% planted anomalies (seed 42).

    python scripts/make_anomaly_demo.py          # needs data/raw/bank_marketing.csv

Writes (data/demo/ is not committed; this script rebuilds it):
  data/demo/bank_anomalies.csv          the data with 412 changed rows (row count unchanged)
  data/demo/bank_anomalies_labels.csv   ground truth: row (0 = first data row), kind, columns
The three kinds are described in src/validation/anomaly_inject.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.profiling.profiler import read_csv_raw  # noqa: E402
from src.validation.anomaly import prepare  # noqa: E402
from src.validation.anomaly_inject import SEED, SHARE, inject, usable_columns  # noqa: E402

OUT = ROOT / "data/demo"


def main() -> None:
    """Plant the anomalies and write the data + labels."""
    src = ROOT / "data/raw/bank_marketing.csv"
    if not src.exists():
        sys.exit(f"{src} not found: run python scripts/download_bank_marketing.py first")
    df = read_csv_raw(src)
    clean = prepare(df)
    columns = usable_columns(clean, skip=["y", "pdays"])
    changed, truth = inject(df, clean, columns, SHARE, SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    changed.to_csv(OUT / "bank_anomalies.csv", index=False)
    truth.to_csv(OUT / "bank_anomalies_labels.csv", index=False)
    print(f"usable columns: {columns}")
    print(f"{len(truth)} of {len(df)} rows changed ({len(truth) / len(df):.2%}): "
          + ", ".join(f"{k} {n}" for k, n in truth["kind"].value_counts().sort_index().items()))
    combo = truth[truth["kind"] == "combination"]["columns"].iloc[0]
    print(f"combination pair: {combo}")


if __name__ == "__main__":
    main()
