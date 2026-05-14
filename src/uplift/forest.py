"""Causal Forest via EconML's CausalForestDML.

A causal forest partitions feature space to maximize heterogeneity in
treatment effects, rather than minimize variance in the outcome. Each
tree finds splits that produce the most-different CATEs in its children.

EconML's CausalForestDML wraps this with Double Machine Learning: nuisance
models for E[Y|X] and E[T|X] are fit first, and the forest is trained on
residuals. This makes the estimator Neyman-orthogonal — first-order
robust to errors in nuisance estimation.

For us this is mostly a thin interface: we want a class with the same
.fit / .predict_cate API as the meta-learners so downstream code is
uniform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor

from uplift.treatment import encode_features
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")


class CausalForest:
    """Wrapper around EconML's CausalForestDML for our pipeline.

    Parameters
    ----------
    n_estimators
            Number of trees. Must be a multiple of 4 because EconML's
            default subforest inference groups trees in subforests of
            size 4 for confidence interval estimation. 500 is a good
            balance between accuracy and training time on ~40K rows.
    min_samples_leaf
        Minimum samples per leaf. Larger values produce smoother CATEs
        and reduce overfitting. 50 is a sensible default for this size.
    max_depth
        Tree depth cap. None lets trees grow until min_samples_leaf or
        purity stops them.
    nuisance_n_estimators
        LightGBM trees in the nuisance models for E[Y|X] and E[T|X].
    random_state
        Reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf: int = 50,
        max_depth: int | None = None,
        nuisance_n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.nuisance_n_estimators = nuisance_n_estimators
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, T, Y) -> "CausalForest":
        X_enc = encode_features(X)
        self.feature_cols_ = list(X_enc.columns)

        # LightGBM nuisance models (faster than sklearn defaults)
        nuisance_kwargs = dict(
            n_estimators=self.nuisance_n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            random_state=self.random_state,
            verbose=-1,
        )
        # model_y predicts E[Y|X] — regression even for binary Y (predicts probabilities)
        # model_t predicts E[T|X] — always classification
        self.forest_ = CausalForestDML(
            model_y=LGBMRegressor(**nuisance_kwargs),
            model_t=LGBMClassifier(**nuisance_kwargs),
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        # EconML expects numpy arrays for Y and T; X can be a DataFrame
        self.forest_.fit(Y=np.asarray(Y), T=np.asarray(T), X=X_enc)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = encode_features(X).reindex(columns=self.feature_cols_, fill_value=0)
        return self.forest_.effect(X_enc)

    def feature_importances(self) -> pd.Series:
        """Heterogeneity-driving feature importances.

        Note: these are different from predictive importances. They
        measure which features the forest uses to split on for CATE
        heterogeneity, not for predicting Y.
        """
        importances = self.forest_.feature_importances_
        return pd.Series(importances, index=self.feature_cols_).sort_values(ascending=False)
