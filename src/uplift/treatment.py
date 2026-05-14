"""Treatment and outcome definitions for the uplift pipeline.

This module turns the raw Hillstrom DataFrame into the (X, T, Y) triples
that estimators consume. The mapping is deliberately explicit and lives
in one place so the estimand is unambiguous and easy to audit.

Estimand
--------
We estimate the Conditional Average Treatment Effect (CATE) of receiving
any promotional email on the probability of visiting the site:

    tau(x) = E[Y(T=1) - Y(T=0) | X = x]

where T = 1 if the customer was sent any email (men's or women's) and
T = 0 if they were in the no-email control arm. Y is the binary `visit`
outcome during the 2-week observation window. Policies are evaluated on
the `spend` outcome (dollars).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import re

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

# The 'segment' column in the raw data names the arm of the experiment.
TREATMENT_COL_RAW = "segment"

# Possible outcomes. `visit` is the training signal, `spend` is what the
# policy is ultimately evaluated against. `conversion` is sparse but
# available for sanity checks.
OutcomeName = Literal["visit", "conversion", "spend"]

# The features the model is allowed to use. `segment`, `visit`, `conversion`,
# `spend` are excluded — they're either the treatment or the outcome.
FEATURE_COLS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_binary_treatment(df: pd.DataFrame) -> pd.Series:
    """Binarize the 3-arm `segment` column into T ∈ {0, 1}.

    T = 1 if the customer received any email, T = 0 for the no-email arm.

    Returns
    -------
    pd.Series
        int8 series, same index as `df`, named 'treatment'.
    """
    if TREATMENT_COL_RAW not in df.columns:
        raise KeyError(f"Expected column {TREATMENT_COL_RAW!r} not in DataFrame.")

    # Be explicit about which strings count as treated. Comparing to
    # "No E-Mail" rather than 'in {treated_arms}' makes the failure mode
    # loud if the source data ever introduces a new arm — we'd get a
    # silent miscategorization otherwise.
    treatment = (df[TREATMENT_COL_RAW] != "No E-Mail").astype("int8")
    treatment.name = "treatment"
    return treatment


def get_outcome(df: pd.DataFrame, outcome: OutcomeName) -> pd.Series:
    """Extract a single outcome column with a documented dtype."""
    if outcome not in ("visit", "conversion", "spend"):
        raise ValueError(f"Unknown outcome {outcome!r}.")
    return df[outcome].rename("outcome")


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix X — pre-treatment covariates only.

    Excludes the treatment column and all outcome columns, leaving only
    customer attributes known *before* the email was sent. This is the
    'no leakage' contract: anything in here must be a pre-treatment
    variable.
    """
    return df[FEATURE_COLS].copy()


def make_xty(
    df: pd.DataFrame,
    outcome: OutcomeName = "visit",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """One-shot helper to extract (X, T, Y) from a raw DataFrame.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, only pre-treatment covariates.
    T : pd.Series
        Binary treatment indicator (0/1).
    Y : pd.Series
        The selected outcome.
    """
    X = get_features(df)
    T = make_binary_treatment(df)
    Y = get_outcome(df, outcome)
    return X, T, Y


# ---------------------------------------------------------------------------
# Quick descriptive statistics — useful for the EDA notebook
# ---------------------------------------------------------------------------


def naive_ate(df: pd.DataFrame, outcome: OutcomeName = "visit") -> dict[str, float]:
    """Compute the unadjusted difference-in-means estimate of the ATE.

    Under randomization this is an unbiased estimator of the population
    Average Treatment Effect (ATE). Use it as a sanity floor: if your
    fancy estimator's average predicted CATE wildly disagrees with this
    number, something is wrong.

    Returns
    -------
    dict with keys: y_treated, y_control, ate, ate_se, n_treated, n_control
    """
    _, T, Y = make_xty(df, outcome)
    y1 = Y[T == 1]
    y0 = Y[T == 0]
    ate = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    return {
        "y_treated": float(y1.mean()),
        "y_control": float(y0.mean()),
        "ate": ate,
        "ate_se": se,
        "n_treated": int(len(y1)),
        "n_control": int(len(y0)),
    }


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals with LightGBM-safe column names.

    pd.get_dummies produces names like 'history_segment_2) $100 - $200'.
    LightGBM rejects feature names containing JSON-special characters
    (parentheses, dollar signs, spaces, etc.), so we sanitize to keep
    only alphanumerics and underscores. The mapping is one-way but
    consistent: same input always yields same output column names, which
    is what model alignment needs.
    """
    X_enc = pd.get_dummies(X, drop_first=True)
    X_enc.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_") for c in X_enc.columns]
    return X_enc
