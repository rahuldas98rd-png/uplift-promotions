"""Tests for evaluation primitives."""

import numpy as np
import pytest

from uplift.evaluation import (
    qini_coefficient,
    qini_curve,
    top_k_lift,
    uplift_curve,
)

from uplift.evaluation import (
    ips_value,
    policy_from_cate,
    snips_value,
    topk_policy,
)


@pytest.fixture
def perfect_targeting():
    """1000 units. True CATE = 0.5 for first half, 0 for second half.
    Predicted CATE perfectly identifies the persuadables.
    """
    rng = np.random.default_rng(0)
    n = 1000
    T = (rng.random(n) < 0.5).astype(int)
    # True effect of 0.5 for first half, 0 for second half
    is_persuadable = np.arange(n) < 500
    Y = (rng.random(n) < (0.2 + 0.5 * is_persuadable * T)).astype(float)
    # Predicted CATE: high for persuadables, low for others
    cate_predicted = is_persuadable.astype(float) + rng.random(n) * 0.01
    return cate_predicted, T, Y


@pytest.fixture
def random_targeting(perfect_targeting):
    """Same data as perfect_targeting, but CATE is random — no signal."""
    cate, T, Y = perfect_targeting
    rng = np.random.default_rng(99)
    return rng.random(len(cate)), T, Y


def test_qini_starts_at_zero(perfect_targeting):
    cate, T, Y = perfect_targeting
    fractions, q = qini_curve(cate, T, Y)
    assert fractions[0] == 0.0
    assert q[0] == 0.0


def test_qini_ends_at_total_lift(perfect_targeting):
    """At k=1 (target everyone), Qini equals the total realized lift."""
    cate, T, Y = perfect_targeting
    _, q = qini_curve(cate, T, Y)
    n_T = T.sum()
    n_C = (1 - T).sum()
    expected = Y[T == 1].sum() - Y[T == 0].sum() * (n_T / n_C)
    assert abs(q[-1] - expected) < 1e-6


def test_uplift_and_qini_agree_at_endpoints(perfect_targeting):
    """Both curves start at 0 and end at the same total realized lift.

    They generally disagree in the middle because uplift_curve uses per-k
    scaling N_T(k)/N_C(k) while qini_curve uses total-arm scaling N_T/N_C.
    The two conventions coincide only at k=0 and k=n.
    """
    cate, T, Y = perfect_targeting
    _, q = qini_curve(cate, T, Y)
    _, u = uplift_curve(cate, T, Y)
    assert q[0] == 0.0 and u[0] == 0.0
    assert abs(q[-1] - u[-1]) < 1e-6


def test_qini_positive_for_good_targeting(perfect_targeting):
    cate, T, Y = perfect_targeting
    qc = qini_coefficient(cate, T, Y, normalize=False)
    assert qc > 0, "Qini should be positive for good targeting"


def test_qini_near_zero_for_random(random_targeting):
    cate, T, Y = random_targeting
    qc = qini_coefficient(cate, T, Y, normalize=False)
    # Random ranking → Qini ≈ 0. Allow small finite-sample wobble.
    assert abs(qc) < 5, f"Random Qini should be near zero, got {qc}"


def test_qini_anti_targeting_is_negative(perfect_targeting):
    """Reversing the ranking should produce negative Qini."""
    cate, T, Y = perfect_targeting
    qc_good = qini_coefficient(cate, T, Y, normalize=False)
    qc_bad = qini_coefficient(-cate, T, Y, normalize=False)
    assert qc_bad < 0
    assert abs(qc_bad + qc_good) < 1e-6  # symmetric about zero


def test_top_k_lift_shape(perfect_targeting):
    cate, T, Y = perfect_targeting
    result = top_k_lift(cate, T, Y, 0.2)
    assert set(result.keys()) == {"ate", "se", "n_treated", "n_control"}


def test_top_k_lift_picks_up_persuadables(perfect_targeting):
    """Top-decile lift should be much larger than overall ATE."""
    cate, T, Y = perfect_targeting
    overall = Y[T == 1].mean() - Y[T == 0].mean()
    top = top_k_lift(cate, T, Y, 0.1)["ate"]
    assert top > overall * 1.3, f"Top-decile lift {top} should exceed overall {overall}"


def test_ips_treat_all_equals_treated_mean(perfect_targeting):
    """Under constant propensity, IPS for 'treat everyone' should equal
    the observed mean Y in the treated arm.
    """
    cate, T, Y = perfect_targeting
    policy = np.ones(len(T), dtype=int)
    propensity = np.full(len(T), 0.5)
    v = ips_value(policy, T, Y, propensity)
    # With propensity=0.5 and policy=all-1: 1/n * sum(T_i==1) * 2 * Y_i
    #   = 2/n * sum_{T=1} Y = (n_treated/n) * 2 * Y_treated_mean
    # Under T~Bernoulli(0.5), n_treated/n ≈ 0.5, so v ≈ Y_treated_mean
    expected = Y[T == 1].mean()
    assert abs(v - expected) < 0.05


def test_ips_treat_none_equals_control_mean(perfect_targeting):
    cate, T, Y = perfect_targeting
    policy = np.zeros(len(T), dtype=int)
    propensity = np.full(len(T), 0.5)
    v = ips_value(policy, T, Y, propensity)
    expected = Y[T == 0].mean()
    assert abs(v - expected) < 0.05


def test_snips_treat_all_unbiased_at_constant_propensity(perfect_targeting):
    """SNIPS with constant propensity should exactly recover the
    weighted mean — and under propensity=0.5 + balanced arms, that's
    the treated-arm mean."""
    cate, T, Y = perfect_targeting
    policy = np.ones(len(T), dtype=int)
    propensity = np.full(len(T), 0.5)
    v = snips_value(policy, T, Y, propensity)
    expected = Y[T == 1].mean()
    assert abs(v - expected) < 0.02


def test_smart_policy_beats_random(perfect_targeting):
    """A policy that targets high-CATE customers should outperform
    a random-fraction policy of the same coverage."""
    cate, T, Y = perfect_targeting
    propensity = np.full(len(T), 0.5)

    smart_policy = topk_policy(cate, 0.3)
    rng = np.random.default_rng(7)
    random_policy = (rng.random(len(cate)) < 0.3).astype(int)

    v_smart = snips_value(smart_policy, T, Y, propensity)
    v_random = snips_value(random_policy, T, Y, propensity)
    assert v_smart > v_random


def test_topk_policy_sizes_correctly():
    cate = np.array([1.0, 0.5, 0.2, 0.1, 0.9, 0.7, 0.3, 0.4, 0.6, 0.8])
    policy = topk_policy(cate, 0.3)
    assert policy.sum() == 3, f"Expected 3 selected, got {policy.sum()}"


def test_policy_from_cate_threshold():
    cate = np.array([-0.5, 0.0, 0.1, 0.5, 1.0])
    # Cost=0.1, margin=1.0 → treat iff cate > 0.1
    policy = policy_from_cate(cate, cost=0.1, margin=1.0)
    assert policy.tolist() == [0, 0, 0, 1, 1]


def test_ips_validates_propensity_range():
    """Propensity at exactly 0 or 1 should raise — division blow-up."""
    policy = np.array([1, 0])
    T = np.array([1, 0])
    Y = np.array([1.0, 0.0])
    bad_propensity = np.array([0.5, 1.0])  # 1.0 is bad
    try:
        ips_value(policy, T, Y, bad_propensity)
    except ValueError:
        return  # expected
    raise AssertionError("Should have raised on propensity=1.0")
