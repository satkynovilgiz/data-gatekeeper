"""scripts/download_adult.py with a mocked network: checksum check, file-format clean-up."""
import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_adult", ROOT / "scripts/download_adult.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

TRAIN = ("39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, "
         "White, Male, 2174, 0, 40, United-States, <=50K\n"
         "52, ?, 209642, HS-grad, 9, Married-civ-spouse, ?, Husband, White, Male, 0, 0, 45, "
         "United-States, >50K\n")
TEST = ("|1x3 Cross validator\n"
        "25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, Own-child, Black, "
        "Male, 0, 0, 40, ?, >50K.\n\n")


def fake_zip(train: str = TRAIN, test: str = TEST) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("adult.data", train)
        z.writestr("adult.test", test)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, data: bytes):
        self.data = data

    def read(self) -> bytes:
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def network(monkeypatch):
    """Every urlopen returns the fake zip; no real network is used."""
    data = fake_zip()
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        return FakeResponse(data)
    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    return data, calls


def test_download_joins_cleans_format_and_writes_checksum(tmp_path, network):
    data, calls = network
    out = tmp_path / "adult.csv"
    info = dl.main(out=out, expected=hashlib.sha256(data).hexdigest())
    assert calls == [dl.URL]
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(df.columns) == dl.COLUMNS
    assert len(df) == 3 and info == {**info, "train_rows": 2, "test_rows": 1, "rows": 3}
    assert df["income"].tolist() == ["<=50K", ">50K", ">50K"]          # "." removed
    assert df.loc[0, "workclass"] == "State-gov"                       # separator space gone
    assert (df == "?").sum().sum() == 3                                # real "?" kept
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert (tmp_path / "adult.csv.sha256").read_text() == f"{digest}  adult.csv\n"
    assert info["csv_sha256"] == digest


def test_checksum_mismatch_stops_before_writing(tmp_path, network):
    out = tmp_path / "adult.csv"
    with pytest.raises(dl.ChecksumError, match="SHA-256 mismatch"):
        dl.main(out=out, expected="0" * 64)
    assert not out.exists()


def test_pinned_checksum_is_a_sha256():
    assert len(dl.ZIP_SHA256) == 64 and int(dl.ZIP_SHA256, 16) >= 0
    assert dl.URL.startswith("https://archive.ics.uci.edu/")


def test_broken_rows_are_refused():
    with pytest.raises(ValueError, match="expected 15 values"):
        dl.rows_from("1, 2, 3\n")
