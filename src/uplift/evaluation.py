"""Evaluation primitives for uplift models.

This module computes:
- Uplift curves and Qini curves
- Qini coefficient (single-number summary)
- Top-decile realized lift and related summary statistics

Convention: all functions take three aligned arrays — predicted CATEs,
observed binary treatment, and observed outcomes — and return either
arrays for plotting or scalar summaries.

The IPS/SNIPS policy value estimators are in this module too but are
introduced separately in Step 4.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Uplift and Qini curves
# ---------------------------------------------------------------------------


def _align(
    cate: np.ndarray | pd.Series,
    T: np.ndarray | pd.Series,
    Y: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert inputs to numpy arrays, validate shape and dtype."""
    cate = np.asarray(cate).ravel()
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(float)
    if not (len(cate) == len(T) == len(Y)):
        raise ValueError(f"Length mismatch: cate={len(cate)}, T={len(T)}, Y={len(Y)}.")
    if not set(np.unique(T)).issubset({0, 1}):
        raise ValueError("T must contain only 0/1 values.")
    return cate, T, Y


def uplift_curve(cate: np.ndarray, T: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the uplift curve.

    For each ranking position k (0 to n), compute the cumulative uplift
    from targeting the top-k customers by predicted CATE:

        uplift(k) = sum_{i in top k, T_i=1} Y_i
                   - sum_{i in top k, T_i=0} Y_i * (N_T(k) / N_C(k))

    The scaling factor on the control sum is what makes the uplift curve
    comparable across treatment ratios. Without it, uplift would just
    track the marginal treatment effect, not the model's targeting skill.

    Returns
    -------
    fractions : ndarray, shape (n+1,)
        Targeting fractions from 0 to 1.
    uplift : ndarray, shape (n+1,)
        Cumulative uplift at each fraction.
    """
    cate, T, Y = _align(cate, T, Y)
    n = len(cate)

    # Sort by CATE descending; tie-break by Y for stability
    order = np.argsort(-cate, kind="stable")
    T_sorted = T[order]
    Y_sorted = Y[order]

    # Cumulative counts and sums of treated and control
    n_treated_cum = np.cumsum(T_sorted)
    n_control_cum = np.cumsum(1 - T_sorted)
    y_treated_cum = np.cumsum(Y_sorted * T_sorted)
    y_control_cum = np.cumsum(Y_sorted * (1 - T_sorted))

    # Avoid divide-by-zero when no control units yet
    safe_n_control = np.where(n_control_cum == 0, 1, n_control_cum)
    uplift = y_treated_cum - y_control_cum * (n_treated_cum / safe_n_control)

    # Prepend a zero so the curve starts at (0, 0)
    fractions = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    uplift = np.concatenate([[0.0], uplift])

    return fractions, uplift


def qini_curve(cate: np.ndarray, T: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Qini curve (Radcliffe formulation).

    Identical to uplift_curve under balanced arms. Under imbalanced
    arms, the control sum is rescaled differently:

        qini(k) = sum_{i in top k, T_i=1} Y_i
                 - sum_{i in top k, T_i=0} Y_i * (N_T_total / N_C_total)

    Note the rescaling factor uses *total* arm sizes, not per-k counts.
    This is the original Qini definition; the uplift curve uses per-k.
    """
    cate, T, Y = _align(cate, T, Y)

    n_treated_total = int(T.sum())
    n_control_total = int((1 - T).sum())
    if n_treated_total == 0 or n_control_total == 0:
        raise ValueError("Need both treated and control units to compute Qini.")

    scale = n_treated_total / n_control_total

    order = np.argsort(-cate, kind="stable")
    T_sorted = T[order]
    Y_sorted = Y[order]

    y_treated_cum = np.cumsum(Y_sorted * T_sorted)
    y_control_cum = np.cumsum(Y_sorted * (1 - T_sorted))
    qini = y_treated_cum - y_control_cum * scale

    n = len(cate)
    fractions = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    qini = np.concatenate([[0.0], qini])

    return fractions, qini


def qini_coefficient(
    cate: np.ndarray, T: np.ndarray, Y: np.ndarray, normalize: bool = True
) -> float:
    """Single-number summary of a Qini curve: area between curve and diagonal.

    The diagonal here is the line from (0, 0) to (1, total_qini) — the
    Qini value at full targeting, which equals the naive total lift.

    Parameters
    ----------
    normalize
        If True, divide by the Qini of perfect targeting (an oracle that
        ranks customers by their true treatment effect). The result is in
        [-some_negative, 1] with 1 meaning oracle-quality ranking.

    Returns
    -------
    float
        The Qini coefficient. Positive means better-than-random; negative
        means worse-than-random; ~0 means random.
    """
    fractions, qini = qini_curve(cate, T, Y)

    # Diagonal: linear interpolation from (0, 0) to (1, qini[-1])
    diagonal = fractions * qini[-1]

    # Area between curve and diagonal, by trapezoidal integration
    area = np.trapezoid(qini - diagonal, fractions)

    if not normalize:
        return float(area)

    # Oracle area: rank by realized outcome difference per arm — not
    # perfect but a defensible upper bound. Treated units with Y=1 get
    # ranked highest, then control with Y=0 (they "would have responded
    # if treated"), then control with Y=1 and treated with Y=0.
    # For binary outcomes this approximates the oracle Qini.
    cate, T, Y = _align(cate, T, Y)
    oracle_score = np.where(T == 1, Y, -Y)  # crude but stable
    _, oracle_qini = qini_curve(oracle_score, T, Y)
    oracle_area = np.trapezoid(oracle_qini - fractions * oracle_qini[-1], fractions)

    if oracle_area == 0:
        return float(area)
    return float(area / oracle_area)


def top_k_lift(cate: np.ndarray, T: np.ndarray, Y: np.ndarray, k: float) -> dict[str, float]:
    """Realized lift in the top-k fraction by predicted CATE.

    Returns dict with ATE in the top-k group plus standard error and
    sample sizes per arm — useful for sanity checks alongside curves.
    """
    cate, T, Y = _align(cate, T, Y)
    n_top = int(len(cate) * k)
    if n_top < 2:
        raise ValueError(f"Top-k fraction {k} gives fewer than 2 units.")

    top_idx = np.argsort(-cate, kind="stable")[:n_top]
    T_top = T[top_idx]
    Y_top = Y[top_idx]
    y1 = Y_top[T_top == 1]
    y0 = Y_top[T_top == 0]
    if len(y1) < 2 or len(y0) < 2:
        return {"ate": np.nan, "se": np.nan, "n_treated": len(y1), "n_control": len(y0)}

    ate = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    return {"ate": ate, "se": se, "n_treated": len(y1), "n_control": len(y0)}


# ---------------------------------------------------------------------------
# IPS / SNIPS policy value estimators
# ---------------------------------------------------------------------------


def _validate_policy_inputs(
    policy: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    propensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert inputs and validate. Policies must be 0/1 indicators."""
    policy = np.asarray(policy).ravel().astype(int)
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(float)
    propensity = np.asarray(propensity).ravel().astype(float)

    if not (len(policy) == len(T) == len(Y) == len(propensity)):
        raise ValueError(
            f"Length mismatch: policy={len(policy)}, T={len(T)}, "
            f"Y={len(Y)}, propensity={len(propensity)}."
        )
    if not set(np.unique(policy)).issubset({0, 1}):
        raise ValueError("policy must be 0/1 indicators.")
    if (propensity <= 0).any() or (propensity >= 1).any():
        raise ValueError("propensity must be strictly in (0, 1).")
    return policy, T, Y, propensity


def ips_value(
    policy: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    propensity: np.ndarray,
) -> float:
    """Inverse Propensity Score estimate of policy value.

    Estimates E[Y(π(X))] — the expected outcome if the policy π were
    deployed. Unbiased under unconfoundedness and positivity.

        V_IPS = (1/n) * sum_i  1{T_i = π(X_i)} / P(T_i = π(X_i) | X_i)  * Y_i

    The denominator is e(X) if π(X)=1, else 1-e(X). Observations where
    T disagrees with π contribute zero — they're not informative about
    the policy's value under unconfoundedness.

    Parameters
    ----------
    policy : array of {0, 1}
        Policy recommendation per unit.
    T, Y : arrays
        Observed treatment and outcome.
    propensity : array in (0, 1)
        Probability of treatment given covariates, P(T=1|X). For
        randomized data, this is constant; in observational data, comes
        from a propensity model.

    Returns
    -------
    float : the IPS estimate of policy value.
    """
    policy, T, Y, propensity = _validate_policy_inputs(policy, T, Y, propensity)
    n = len(policy)

    # P(T = policy | X) — depends on what the policy recommends
    p_action = np.where(policy == 1, propensity, 1 - propensity)

    # Indicator: did the observed T happen to match the policy?
    match = (T == policy).astype(float)

    return float(np.sum(match * Y / p_action) / n)


def snips_value(
    policy: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    propensity: np.ndarray,
) -> float:
    """Self-Normalized IPS (SNIPS): IPS divided by sum of weights, not n.

    Slightly biased but lower variance than IPS, especially when
    propensities are small or extreme weights dominate. Generally
    preferred in practice.

        V_SNIPS = sum_i [1{T_i=π} / p_action(X_i)] * Y_i
                / sum_i [1{T_i=π} / p_action(X_i)]
    """
    policy, T, Y, propensity = _validate_policy_inputs(policy, T, Y, propensity)
    p_action = np.where(policy == 1, propensity, 1 - propensity)
    match = (T == policy).astype(float)

    weights = match / p_action
    total = weights.sum()
    if total == 0:
        return 0.0
    return float(np.sum(weights * Y) / total)


def policy_from_cate(
    cate: np.ndarray,
    cost: float = 0.0,
    margin: float = 1.0,
) -> np.ndarray:
    """Cost-aware decision rule: treat if predicted CATE × margin > cost.

    For binary 'treat or not' decisions with a known per-unit cost and
    a margin that converts outcome units (e.g., spend dollars) to value.
    """
    return (np.asarray(cate) * margin > cost).astype(int)


def topk_policy(cate: np.ndarray, k: float) -> np.ndarray:
    """Policy: treat the top-k fraction by predicted CATE.

    Useful for value-vs-coverage curves (the marginal cost analysis).
    """
    cate = np.asarray(cate).ravel()
    n_top = int(len(cate) * k)
    if n_top == 0:
        return np.zeros(len(cate), dtype=int)
    threshold = np.partition(-cate, n_top - 1)[n_top - 1]
    policy = (-cate <= threshold).astype(int)
    # Tie-breaking: if duplicates push us over, trim
    if policy.sum() > n_top:
        # Take the lowest n_top indices among the policy-1 group
        cand = np.where(policy == 1)[0]
        keep = cand[np.argsort(-cate[cand], kind="stable")[:n_top]]
        policy = np.zeros(len(cate), dtype=int)
        policy[keep] = 1
    return policy


# ---------------------------------------------------------------------------
# Customer segmentation by (baseline response, predicted CATE)
# ---------------------------------------------------------------------------


def assign_segments(
    cate: np.ndarray,
    baseline: np.ndarray,
    cate_threshold: float = 0.0,
    baseline_threshold: float = 0.5,
) -> np.ndarray:
    """Classify customers into the four uplift segments.

    Uses two thresholds:
    - cate_threshold: CATE > threshold means "treatment helps this customer"
    - baseline_threshold: μ_0(x) > threshold means "would have responded
      without treatment"

    For binary outcomes, baseline_threshold = 0.5 is the natural choice
    (more likely than not to visit without the email). For continuous
    outcomes (spend), use the median observed baseline as the threshold.

    Returns
    -------
    np.ndarray of strings, one per customer:
      "persuadable", "sure_thing", "lost_cause", "do_not_disturb"
    """
    cate = np.asarray(cate).ravel()
    baseline = np.asarray(baseline).ravel()
    if len(cate) != len(baseline):
        raise ValueError(f"Length mismatch: cate={len(cate)}, baseline={len(baseline)}")

    high_cate = cate > cate_threshold
    high_baseline = baseline > baseline_threshold

    segments = np.empty(len(cate), dtype=object)
    segments[high_cate & ~high_baseline] = "persuadable"
    segments[high_cate & high_baseline] = "sure_thing"
    segments[~high_cate & ~high_baseline] = "lost_cause"
    segments[~high_cate & high_baseline] = "do_not_disturb"
    return segments


def segment_summary(
    segments: np.ndarray,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Mean feature values per segment, for narrative interpretation.

    Produces a comparison table showing how segments differ on the
    interpretable features. This is the input to the 'who are the
    persuadables in human terms?' analysis.

    For numeric columns: returns the mean. For categorical/boolean:
    returns the proportion of each level via numeric coercion of
    indicator columns. The caller typically prepares feature data
    accordingly.
    """
    df = features.copy()
    df["_segment"] = segments
    counts = df["_segment"].value_counts().rename("count")
    proportions = (counts / counts.sum()).rename("proportion")

    # Compute mean per segment for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    means = df.groupby("_segment", observed=True)[numeric_cols].mean()

    summary = means.copy()
    summary.insert(0, "count", counts)
    summary.insert(1, "proportion", proportions)
    return summary.sort_values("count", ascending=False)
