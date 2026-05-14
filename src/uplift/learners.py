"""Meta-learners for CATE estimation.

A meta-learner is a CATE estimator built by wrapping a base regression
or classification model. The 'meta' part refers to the fact that the
base learner is interchangeable — you can plug in LightGBM, sklearn's
random forest, a linear model, etc.

This module implements:
- SLearner: single model on the augmented feature space [X, T].
- TLearner: two independent models, one per treatment arm.

X-learner and DR-learner come in Phase 3.

The base learner used here is LightGBM, but each class accepts arbitrary
parameter overrides via `base_params`. Classification vs regression is
auto-detected from the outcome (binary -> classifier, otherwise regressor).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from uplift.treatment import encode_features


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_params() -> dict[str, Any]:
    """Reasonable LightGBM defaults for ~10K-100K row tabular data."""
    return dict(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        random_state=42,
        verbose=-1,
    )


def _is_binary(y) -> bool:
    return pd.Series(y).nunique() <= 2


def _make_model(is_classifier: bool, params: dict[str, Any]):
    cls = LGBMClassifier if is_classifier else LGBMRegressor
    return cls(**params)


def _predict_outcome(model, X: pd.DataFrame, is_classifier: bool) -> np.ndarray:
    if is_classifier:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


# ---------------------------------------------------------------------------
# S-learner
# ---------------------------------------------------------------------------


class SLearner:
    """Single-model meta-learner: μ(x, t), CATE = μ(x,1) - μ(x,0)."""

    def __init__(self, base_params: dict | None = None):
        self.base_params = {**_default_params(), **(base_params or {})}

    def fit(self, X: pd.DataFrame, T, Y) -> "SLearner":
        X_enc = encode_features(X)
        self.feature_cols_ = list(X_enc.columns)

        # Augment X with treatment as just another feature
        X_aug = X_enc.copy()
        X_aug["_treatment"] = np.asarray(T).astype(np.int8)

        self.is_classifier_ = _is_binary(Y)
        self.model_ = _make_model(self.is_classifier_, self.base_params)
        self.model_.fit(X_aug, Y)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = encode_features(X).reindex(columns=self.feature_cols_, fill_value=0)
        X1, X0 = X_enc.copy(), X_enc.copy()
        X1["_treatment"] = 1
        X0["_treatment"] = 0
        mu1 = _predict_outcome(self.model_, X1, self.is_classifier_)
        mu0 = _predict_outcome(self.model_, X0, self.is_classifier_)
        return mu1 - mu0


# ---------------------------------------------------------------------------
# T-learner
# ---------------------------------------------------------------------------


class TLearner:
    """Two-model meta-learner: μ_t fit on T==t subset, CATE = μ_1 - μ_0."""

    def __init__(self, base_params: dict | None = None):
        self.base_params = {**_default_params(), **(base_params or {})}

    def fit(self, X: pd.DataFrame, T, Y) -> "TLearner":
        X_enc = encode_features(X)
        self.feature_cols_ = list(X_enc.columns)

        T_arr = np.asarray(T)
        Y_arr = np.asarray(Y)

        self.is_classifier_ = _is_binary(Y_arr)
        treated = T_arr == 1

        self.model_1_ = _make_model(self.is_classifier_, self.base_params)
        self.model_0_ = _make_model(self.is_classifier_, self.base_params)

        self.model_1_.fit(X_enc[treated], Y_arr[treated])
        self.model_0_.fit(X_enc[~treated], Y_arr[~treated])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = encode_features(X).reindex(columns=self.feature_cols_, fill_value=0)
        mu1 = _predict_outcome(self.model_1_, X_enc, self.is_classifier_)
        mu0 = _predict_outcome(self.model_0_, X_enc, self.is_classifier_)
        return mu1 - mu0
