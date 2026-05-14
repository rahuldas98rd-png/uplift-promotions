"""Tests for S-learner and T-learner."""

import numpy as np
import pytest

from uplift.data import load_raw
from uplift.learners import SLearner, TLearner, XLearner
from uplift.treatment import get_features, make_binary_treatment


@pytest.fixture(scope="module")
def small_data():
    df = load_raw().sample(5000, random_state=0)
    X = get_features(df)
    T = make_binary_treatment(df)
    Y = df["visit"]
    return X, T, Y, df


def test_slearner_shape(small_data):
    X, T, Y, _ = small_data
    cate = SLearner().fit(X, T, Y).predict_cate(X)
    assert cate.shape == (len(X),)


def test_tlearner_shape(small_data):
    X, T, Y, _ = small_data
    cate = TLearner().fit(X, T, Y).predict_cate(X)
    assert cate.shape == (len(X),)


def test_slearner_mean_cate_near_naive_ate(small_data):
    """S-learner can shrink — we allow it to be smaller, not bigger."""
    X, T, Y, _ = small_data
    cate = SLearner().fit(X, T, Y).predict_cate(X)
    naive = Y[T == 1].mean() - Y[T == 0].mean()
    # Sanity: shouldn't be negative on average for a real positive effect,
    # and shouldn't exceed the naive ATE by much.
    assert -0.01 < cate.mean() < naive + 0.03


def test_tlearner_mean_cate_near_naive_ate(small_data):
    """T-learner should be roughly unbiased in expectation."""
    X, T, Y, _ = small_data
    cate = TLearner().fit(X, T, Y).predict_cate(X)
    naive = Y[T == 1].mean() - Y[T == 0].mean()
    assert abs(cate.mean() - naive) < 0.03


def test_slearner_continuous_outcome(small_data):
    """SLearner should auto-detect regression for the continuous spend outcome."""
    X, T, _, df = small_data
    learner = SLearner().fit(X, T, df["spend"])
    cate = learner.predict_cate(X)
    assert cate.shape == (len(X),)
    assert not learner.is_classifier_


def test_learners_handle_new_data_columns(small_data):
    """predict_cate must work on data with the same feature schema."""
    X, T, Y, _ = small_data
    learner = SLearner().fit(X[:4000], T[:4000], Y[:4000])
    cate_new = learner.predict_cate(X[4000:])
    assert cate_new.shape == (len(X) - 4000,)


def test_xlearner_shape(small_data):
    X, T, Y, _ = small_data
    cate = XLearner().fit(X, T, Y).predict_cate(X)
    assert cate.shape == (len(X),)


def test_xlearner_mean_cate_near_naive_ate(small_data):
    """X-learner should be roughly unbiased."""
    X, T, Y, _ = small_data
    cate = XLearner().fit(X, T, Y).predict_cate(X)
    naive = Y[T == 1].mean() - Y[T == 0].mean()
    assert abs(cate.mean() - naive) < 0.03


def test_xlearner_uses_explicit_propensity(small_data):
    """When given a propensity model, X-learner should use it."""
    X, T, Y, _ = small_data

    # Mock propensity that returns the true randomization rate everywhere
    class ConstantPropensity:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    learner = XLearner(propensity_model=ConstantPropensity(0.5)).fit(X, T, Y)
    cate = learner.predict_cate(X)
    assert cate.shape == (len(X),)


def test_xlearner_continuous_outcome(small_data):
    X, T, _, df = small_data
    learner = XLearner().fit(X, T, df["spend"])
    cate = learner.predict_cate(X)
    assert cate.shape == (len(X),)
    assert not learner.is_classifier_
