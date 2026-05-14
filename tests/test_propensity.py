"""Tests for propensity estimation."""

import numpy as np

from uplift.data import load_raw
from uplift.propensity import (
    PropensityModel,
    estimate_propensity_cv,
    overlap_diagnostics,
)
from uplift.treatment import make_binary_treatment, get_features


def test_propensity_under_randomization_is_near_constant():
    """Under randomization, propensity predictions should cluster around 2/3.

    We check the things that actually matter for downstream causal use:
    - Mean is at the true rate (no systematic bias)
    - No extreme predictions (overlap holds everywhere)
    - Spread is bounded (model isn't finding 'signal' where there is none)

    A small amount of spread is expected: with finite data, LightGBM will
    fit some noise. The threshold below is loose enough to pass on
    subsamples but tight enough to fail if randomization actually broke.
    """
    df = load_raw().sample(10000, random_state=0)
    X = get_features(df)
    T = make_binary_treatment(df)

    p = estimate_propensity_cv(X, T)

    # Mean at the prior
    assert 0.65 < p.mean() < 0.68

    # No extreme propensities — overlap holds
    assert p.min() > 0.20, f"min propensity {p.min()} suggests overlap problem"
    assert p.max() < 0.95, f"max propensity {p.max()} suggests overlap problem"

    # Spread is bounded — model isn't seeing signal that doesn't exist.
    # 0.15 is generous for a 10K sample; on the full 38K training set
    # the std will typically be in the 0.05-0.07 range.
    assert p.std() < 0.15


def test_propensity_in_unit_interval():
    df = load_raw().sample(5000, random_state=0)
    X = get_features(df)
    T = make_binary_treatment(df)

    pm = PropensityModel().fit(X, T)
    p = pm.predict(X)
    assert (p >= 0).all() and (p <= 1).all()


def test_overlap_diagnostics_keys():
    p = np.array([0.5, 0.6, 0.7, 0.65, 0.62])
    diag = overlap_diagnostics(p)
    assert {"min", "max", "mean", "std", "frac_extreme", "frac_in_unit_interval"} <= diag.keys()
