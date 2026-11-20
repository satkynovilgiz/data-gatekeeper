"""Fuzzy duplicate detection — the same record typed twice with small differences.

Example: "Jon Smith" and "John Smith" with the same phone, department and hire date.
Exact duplicate removal (step 9) cannot see this: one letter differs.

How it works:
  1. key = the name column(s), normalised: lowercase, no punctuation, single spaces
  2. blocking: only names whose SORTED words start with the same letter are compared
     ("Maria Garcia" and "Garcia, Maria" both sort to "garcia maria"). Typos rarely hit
     the first letter, and it avoids comparing every row with every other row.
  3. name similarity with rapidfuzz token_sort_ratio (0-100): the words are sorted
     first, so "Smith John" = "John Smith" (100); "Jon Smith" vs "John Smith" = 94.7
  4. a pair counts only when the name is similar (>= threshold) AND most of the other
     fields agree (>= min_agreement). Two different people can share a name; they will
     not share age, phone, hire date and salary.
"""
from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

FUZZY_THRESHOLD = 90          # name similarity 0-100
MIN_AGREEMENT = 0.8           # share of the other fields that must be equal


def name_key(value: object) -> str:
    """Normalise a name for comparing: 'Smith,  JOHN ' -> 'smith john'."""
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    return " ".join(text.split())


def _agreement(a: pd.Series, b: pd.Series) -> tuple[int, int, list[str]]:
    """(fields equal, fields compared, names of the fields that differ). Empty cells are skipped."""
    both = a.notna() & b.notna()
    equal = both & (a.astype(str) == b.astype(str))
    return int(equal.sum()), int(both.sum()), [c for c in a.index if both[c] and not equal[c]]


def find_fuzzy_duplicates(df: pd.DataFrame, key_columns: list[str], compare_columns: list[str],
                          threshold: float = FUZZY_THRESHOLD,
                          min_agreement: float = MIN_AGREEMENT) -> list[dict]:
    """Pairs of rows that are probably the same record.

    key_columns:     name columns joined into one key (e.g. ["full_name"])
    compare_columns: the other fields that must mostly agree (not IDs, not the key)
    Returns [{row_a, row_b, name_a, name_b, name_similarity, fields_equal, fields_compared,
              agreement, differs}], best matches first. row_a is always the earlier row.
    """
    keys = df[key_columns].astype(str).agg(" ".join, axis=1).where(
        df[key_columns].notna().all(axis=1)).dropna().map(name_key)
    keys = keys[keys != ""]
    pairs = []
    # block = first letter of the words in alphabetical order (same order token_sort uses)
    block_of = keys.map(lambda k: " ".join(sorted(k.split()))[0])
    for _, block in keys.groupby(block_of):
        if len(block) < 2:
            continue
        # similarity of every name with every other name in the block (fast, in C++);
        # scores below the threshold come back as 0
        scores = process.cdist(block.tolist(), block.tolist(), scorer=fuzz.token_sort_ratio,
                               score_cutoff=threshold)
        rows = block.index.tolist()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if scores[i, j] == 0:
                    continue
                a, b = rows[i], rows[j]
                equal, compared, differs = _agreement(df.loc[a, compare_columns],
                                                      df.loc[b, compare_columns])
                if compared == 0 or equal / compared < min_agreement:
                    continue
                pairs.append({
                    "row_a": int(a), "row_b": int(b),
                    "name_a": " ".join(str(df.at[a, k]) for k in key_columns),
                    "name_b": " ".join(str(df.at[b, k]) for k in key_columns),
                    "name_similarity": round(float(scores[i, j]), 1),
                    "fields_equal": equal, "fields_compared": compared,
                    "agreement": round(equal / compared, 3), "differs": differs,
                })
    return sorted(pairs, key=lambda p: (-p["name_similarity"], -p["agreement"], p["row_a"]))
