"""Sprint 2 — stronger PII detection (credit cards, IP addresses, dates of birth)."""
import pandas as pd

from src.profiling.profiler import profile_dataset
from src.profiling.semantic import detect_semantic, is_id_name, luhn_ok


def test_luhn_check():
    assert luhn_ok("4111111111111111")        # well-known Visa test number
    assert not luhn_ok("4111111111111112")    # last digit changed -> invalid


def test_credit_card_detected():
    cards = pd.Series(["4111 1111 1111 1111", "5500-0000-0000-0004", "340000000000009"])
    assert detect_semantic("payment", cards, "text")["semantic_type"] == "credit_card"


def test_random_16_digit_numbers_are_not_cards():
    # same length as cards but fail the Luhn check -> must not be flagged
    s = pd.Series(["1234567812345678", "1111111111111112", "9999999999999990"])
    assert detect_semantic("account_ref", s, "numeric")["semantic_type"] != "credit_card"


def test_ip_address_detected_not_phone():
    ips = pd.Series(["192.168.0.1", "10.0.0.255", "8.8.8.8"])
    assert detect_semantic("client", ips, "text")["semantic_type"] == "ip_address"
    bad = pd.Series(["999.1.1.1", "1.2.3", "abc"])   # not valid IPv4
    assert detect_semantic("client", bad, "text")["semantic_type"] != "ip_address"


def test_date_of_birth_by_column_name():
    dates = pd.Series(["1990-01-05", "1985-12-31"])
    assert detect_semantic("dob", dates, "datetime")["semantic_type"] == "date_of_birth"
    assert detect_semantic("DateOfBirth", dates, "datetime")["semantic_type"] == "date_of_birth"
    # a normal date column is still just a date, not PII
    assert detect_semantic("signup_date", dates, "datetime")["semantic_type"] == "date"


def test_new_pii_types_in_report():
    df = pd.DataFrame({
        "card": ["4111111111111111", "5500000000000004", "340000000000009"],
        "ip": ["192.168.0.1", "10.0.0.2", "8.8.8.8"],
        "birth_date": ["1990-01-05", "1985-12-31", "2001-07-20"],
        "amount": ["10.5", "20", "30"],
    })
    pii = profile_dataset(df)["summary"]["pii_columns"]
    assert set(pii) == {"card", "ip", "birth_date"}


def test_id_name_uses_whole_words():
    assert is_id_name("customer_id") and is_id_name("ID") and is_id_name("customerId")
    # "idx" and "paid" contain the letters "id" but are not ids
    assert not is_id_name("cons.price.idx")
    assert not is_id_name("paid")
