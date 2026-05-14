"""Tests for the causal forest wrapper."""

import pytest

from uplift.data import load_raw
from uplift.forest import CausalForest
from uplift.treatment import get_features, make_binary_treatment


@pytest.fixture(scope="module")
def small_data():
    df = load_raw().sample(3000, random_state=0)
    X = get_features(df)
    T = make_binary_treatment(df)
    Y = df["visit"]
    return X, T, Y


def test_causal_forest_shape(small_data):
    X, T, Y = small_data
    # Smaller forest for fast tests
    forest = CausalForest(n_estimators=48, nuisance_n_estimators=50).fit(X, T, Y)
    cate = forest.predict_cate(X)
    assert cate.shape == (len(X),)


def test_causal_forest_mean_cate_near_naive(small_data):
    X, T, Y = small_data
    forest = CausalForest(n_estimators=48, nuisance_n_estimators=50).fit(X, T, Y)
    cate = forest.predict_cate(X)
    naive = Y[T == 1].mean() - Y[T == 0].mean()
    # Small samples + small forest → lenient tolerance
    assert abs(cate.mean() - naive) < 0.05


def test_causal_forest_feature_importances(small_data):
    X, T, Y = small_data
    forest = CausalForest(n_estimators=48, nuisance_n_estimators=50).fit(X, T, Y)
    importances = forest.feature_importances()
    assert len(importances) == len(forest.feature_cols_)
    assert importances.sum() > 0  # at least some signal
