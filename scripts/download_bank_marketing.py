"""Download the UCI Bank Marketing dataset (finance domain) into data/raw/.

Source : https://archive.ics.uci.edu/dataset/222/bank+marketing
License: CC BY 4.0 (see docs/dataset.md)

Usage:
    python scripts/download_bank_marketing.py            # full file (41,188 rows)
    python scripts/download_bank_marketing.py --sample   # also rewrite the 1,000-row random sample

The full CSV (~5.8 MB) is NOT stored in git (.gitignore). Only the small
sample is committed so tests and demos work without internet.
"""
import argparse
import io
import random
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
INNER_ZIP = "bank-additional.zip"                       # the download is a zip inside a zip
CSV_IN_ZIP = "bank-additional/bank-additional-full.csv"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
FULL_PATH = RAW_DIR / "bank_marketing.csv"
SAMPLE_PATH = RAW_DIR / "bank_marketing_sample.csv"
SAMPLE_ROWS = 1000
SEED = 42   # fixed seed -> the same "random" sample every time


def download() -> bytes:
    """Return the raw bytes of bank-additional-full.csv."""
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))
    return inner.read(CSV_IN_ZIP)


def main():
    """Download the full Bank Marketing CSV into data/raw/ (and the 1,000-row sample
    when --sample is given)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", action="store_true",
                    help=f"also write {SAMPLE_ROWS} random rows to {SAMPLE_PATH.name}")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = download()
    FULL_PATH.write_bytes(data)          # kept exactly as published (';' separated)
    print(f"Saved {FULL_PATH}  ({len(data) / 1e6:.1f} MB)")

    if args.sample:
        header, *rows = data.decode("utf-8").splitlines(keepends=True)
        # The file is sorted by date (first rows are all May), so take a random
        # sample instead of the first rows; keep the original order.
        idx = sorted(random.Random(SEED).sample(range(len(rows)), SAMPLE_ROWS))
        SAMPLE_PATH.write_text(header + "".join(rows[i] for i in idx))
        print(f"Saved {SAMPLE_PATH}  ({SAMPLE_ROWS} rows)")


if __name__ == "__main__":
    main()
