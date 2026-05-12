"""Tests for the data loading module."""

import pandas as pd
import pytest

from uplift.data import (
    EXPECTED_COLUMNS,
    EXPECTED_ROW_COUNT,
    load_raw,
)


def test_load_raw_shape():
    df = load_raw()
    assert df.shape == (EXPECTED_ROW_COUNT, len(EXPECTED_COLUMNS))


def test_load_raw_columns():
    df = load_raw()
    assert list(df.columns) == EXPECTED_COLUMNS


def test_treatment_arms_balanced():
    """Hillstrom is a 3-arm randomized experiment with ~1/3 in each arm."""
    df = load_raw()
    arm_counts = df["segment"].value_counts(normalize=True)
    for arm in ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]:
        assert 0.32 < arm_counts[arm] < 0.34, f"Arm {arm!r} has unexpected proportion"


def test_no_missing_values():
    """The original dataset has no missing values; if we see any, the load broke."""
    df = load_raw()
    assert df.isna().sum().sum() == 0
