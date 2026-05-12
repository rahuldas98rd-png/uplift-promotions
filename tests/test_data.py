"""Tests for the data loading module."""

import pandas as pd
import pytest

from uplift.data import (
    EXPECTED_COLUMNS,
    EXPECTED_ROW_COUNT,
    load_raw,
)
from uplift.treatment import make_binary_treatment, make_xty, naive_ate


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


def test_binary_treatment_shape_and_dtype():
    df = load_raw()
    t = make_binary_treatment(df)
    assert len(t) == len(df)
    assert t.dtype == "int8"
    assert set(t.unique()) == {0, 1}


def test_binary_treatment_rate():
    """Two of the three Hillstrom arms are treated, so T ~ 2/3."""
    df = load_raw()
    t = make_binary_treatment(df)
    assert 0.66 < t.mean() < 0.68


def test_make_xty_no_leakage():
    """X must not contain the treatment column or any outcome."""
    df = load_raw()
    X, _, _ = make_xty(df)
    forbidden = {"segment", "visit", "conversion", "spend"}
    assert forbidden.isdisjoint(X.columns), (
        f"X contains leaked columns: {forbidden & set(X.columns)}"
    )


def test_naive_ate_visit_is_positive_and_significant():
    """The email campaign should measurably lift visit rate."""
    df = load_raw()
    result = naive_ate(df, "visit")
    assert result["ate"] > 0
    # t-stat > 2 means clearly significant
    assert result["ate"] / result["ate_se"] > 2
