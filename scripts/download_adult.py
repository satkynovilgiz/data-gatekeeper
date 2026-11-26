"""Download UCI Adult (Census Income) into data/raw/adult.csv (Sprint 10).

Source : https://archive.ics.uci.edu/dataset/2/adult  (zip: URL below)
License: CC BY 4.0 (see docs/dataset.md)

    python scripts/download_adult.py

What the script does, and nothing more (the "mess" is left for the gatekeeper to find):
  1. downloads the official zip and checks its SHA-256 against ZIP_SHA256 (stops on a mismatch)
  2. joins adult.data (32,561 rows) and adult.test (16,281 rows) into one table with a header
  3. removes only FILE-FORMAT artefacts, not data problems:
       - the ", " separator leaves a space before every value -> stripped
       - adult.test starts with the line "|1x3 Cross validator" and ends with empty lines
       - adult.test writes the label as ">50K." / "<=50K." -> the trailing "." is removed,
         so the target column `income` has the same two values in both parts
     The "?" missing values, duplicates and everything else stay exactly as published.
  4. writes data/raw/adult.csv and its SHA-256 to data/raw/adult.csv.sha256

The CSV (~5 MB) is NOT stored in git (.gitignore).
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ZIP_SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = RAW_DIR / "adult.csv"


class ChecksumError(RuntimeError):
    """The downloaded file is not the one this script was written for."""


def fetch(url: str = URL, timeout: int = 60) -> bytes:
    """The raw bytes at url."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def sha256(data: bytes) -> str:
    """Hex SHA-256 of some bytes."""
    return hashlib.sha256(data).hexdigest()


def verify(data: bytes, expected: str) -> None:
    """Raise ChecksumError when the bytes do not have the expected SHA-256."""
    got = sha256(data)
    if got != expected:
        raise ChecksumError(f"SHA-256 mismatch: expected {expected}, got {got}. The file at the "
                            "source changed; check it before updating ZIP_SHA256.")


def rows_from(text: str) -> list[list[str]]:
    """Data rows of adult.data / adult.test: separator spaces and the test-file
    artefacts (header line, empty lines, trailing '.' on the label) removed."""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("|"):
            continue
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(COLUMNS):
            raise ValueError(f"expected {len(COLUMNS)} values, got {len(values)}")
        values[-1] = values[-1].rstrip(".")
        out.append(values)
    return out


def build_csv(zip_bytes: bytes) -> tuple[bytes, dict]:
    """The joined CSV bytes and a small summary {train_rows, test_rows}."""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    train = rows_from(z.read("adult.data").decode("utf-8"))
    test = rows_from(z.read("adult.test").decode("utf-8"))
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(train + test)
    return buf.getvalue().encode("utf-8"), {"train_rows": len(train), "test_rows": len(test)}


def main(out: Path = OUT, url: str = URL, expected: str = ZIP_SHA256) -> dict:
    """Download, verify, join and write adult.csv (+ .sha256); returns a summary."""
    print(f"Downloading {url} ...")
    data = fetch(url)
    verify(data, expected)
    csv_bytes, info = build_csv(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(csv_bytes)
    digest = sha256(csv_bytes)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    info.update(rows=info["train_rows"] + info["test_rows"], csv_sha256=digest, path=str(out))
    print(f"Saved {out}: {info['rows']} rows ({info['train_rows']} train + {info['test_rows']} "
          f"test), SHA-256 {digest}")
    return info


if __name__ == "__main__":
    try:
        main()
    except ChecksumError as e:
        sys.exit(f"error: {e}")
