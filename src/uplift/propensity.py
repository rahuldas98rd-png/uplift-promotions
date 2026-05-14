"""Propensity score estimation: P(T = 1 | X = x).

Used downstream by:
- Inverse Propensity Score (IPS) and SNIPS policy value estimators.
- DR-learner (doubly robust CATE) — fit later in Phase 3.
- Overlap diagnostics — distributions of e(x) far from {0, 1} are required
  for any CATE estimator to be valid.

Under randomization the true propensity is constant (~0.667 here). A
well-calibrated model should predict near that constant everywhere; any
spread in predictions is the model fitting noise. We check this explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict
from uplift.treatment import encode_features


class PropensityModel:
    """Wrapper around an LGBM classifier that estimates P(T = 1 | X).

    Holds the trained model plus the column ordering it was fit on, so
    that calls to .predict() on new data with possibly-different
    categorical levels still align correctly.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 50,
        random_state: int = 42,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state,
            verbose=-1,
        )
        self.model: LGBMClassifier | None = None
        self.feature_cols_: list[str] | None = None

    def fit(self, X: pd.DataFrame, T: pd.Series) -> "PropensityModel":
        X_enc = encode_features(X)
        self.feature_cols_ = list(X_enc.columns)
        self.model = LGBMClassifier(**self.params)
        self.model.fit(X_enc, T)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted propensities P(T = 1 | X), shape (n,)."""
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        X_enc = encode_features(X).reindex(columns=self.feature_cols_, fill_value=0)
        return self.model.predict_proba(X_enc)[:, 1]


def estimate_propensity_cv(
    X: pd.DataFrame,
    T: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Cross-fitted propensity estimates.

    Use these when you need out-of-fold predictions to avoid the bias from
    using in-sample propensities in DR-style estimators. For pure
    randomization-check purposes, a single-fit model is fine.
    """
    X_enc = encode_features(X)
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        random_state=random_state,
        verbose=-1,
    )
    proba = cross_val_predict(model, X_enc, T, cv=n_splits, method="predict_proba")
    return proba[:, 1]


def overlap_diagnostics(propensities: np.ndarray) -> dict[str, float]:
    """Summary statistics for assessing overlap / positivity.

    Healthy propensities for this project should cluster near 0.667. We
    report min, max, fraction extreme (< 0.05 or > 0.95), and the spread.
    """
    p = np.asarray(propensities)
    return {
        "min": float(p.min()),
        "max": float(p.max()),
        "mean": float(p.mean()),
        "std": float(p.std()),
        "frac_extreme": float(((p < 0.05) | (p > 0.95)).mean()),
        "frac_in_unit_interval": float(((p >= 0.0) & (p <= 1.0)).mean()),
    }
