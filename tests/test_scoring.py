"""Scoring math tests.

These are property tests for the metrics. If any of them fail, downstream
backtests are unreliable, so we test them tightly.
"""

import math

import pytest

from clio.frozen.scoring import (
    brier_score,
    calibration_ece,
    kelly_pnl,
    resolution_decomposition,
    score_vector,
)


def test_brier_perfect_prediction_is_zero():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)


def test_brier_worst_prediction_is_one():
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_brier_uninformed_at_climatology():
    # Always predicting the base rate gives Brier = base_rate * (1 - base_rate)
    outcomes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    base = sum(outcomes) / len(outcomes)
    probs = [base] * len(outcomes)
    assert brier_score(probs, outcomes) == pytest.approx(base * (1 - base))


def test_brier_length_mismatch_raises():
    with pytest.raises(ValueError):
        brier_score([0.5], [1, 0])


def test_ece_perfect_calibration_is_zero():
    # Predict 0.7 ten times, exactly 7 are YES.
    probs = [0.7] * 10
    outcomes = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert calibration_ece(probs, outcomes) == pytest.approx(0.0, abs=1e-9)


def test_ece_worst_calibration_is_one():
    # Predict 1.0 always, but all resolve NO.
    probs = [1.0] * 5
    outcomes = [0] * 5
    assert calibration_ece(probs, outcomes) == pytest.approx(1.0, abs=1e-6)


def test_ece_empty_inputs_safe():
    assert calibration_ece([], []) == 0.0


def test_ece_length_mismatch_raises():
    with pytest.raises(ValueError):
        calibration_ece([0.5], [1, 0])


def test_resolution_decomposition_identity():
    # Murphy decomposition: BS = REL - RES + UNC, exactly.
    probs = [0.1, 0.4, 0.6, 0.9, 0.5, 0.7, 0.2]
    outcomes = [0, 1, 1, 1, 0, 1, 0]
    rel, res, unc = resolution_decomposition(probs, outcomes)
    bs = brier_score(probs, outcomes)
    assert (rel - res + unc) == pytest.approx(bs, abs=1e-9)


def test_kelly_pnl_no_edge_no_trades():
    # If our prob equals market price exactly, edge is zero, so no trades.
    probs = [0.5, 0.3, 0.7]
    outcomes = [1, 0, 1]
    market = [0.5, 0.3, 0.7]
    out = kelly_pnl(probs, outcomes, market)
    assert out["n_trades"] == 0
    assert out["bankroll_final"] == out["bankroll_initial"]


def test_kelly_pnl_positive_when_we_have_real_edge():
    # We're systematically right and the market is wrong.
    n = 50
    probs = [0.7] * n
    outcomes = [1 if i % 10 < 7 else 0 for i in range(n)]  # ~70% YES
    market = [0.5] * n
    out = kelly_pnl(probs, outcomes, market)
    assert out["n_trades"] > 0
    assert out["bankroll_final"] > out["bankroll_initial"]


def test_kelly_pnl_max_dd_in_unit_interval():
    probs = [0.6, 0.6, 0.6, 0.6, 0.6]
    outcomes = [0, 0, 0, 0, 0]  # we lose everything we bet
    market = [0.4, 0.4, 0.4, 0.4, 0.4]
    out = kelly_pnl(probs, outcomes, market)
    assert 0.0 <= out["max_dd"] <= 1.0


def test_score_vector_dominance_is_asymmetric_and_strict():
    """Dominance is a strict partial order: never reflexive, asymmetric."""
    probs = [0.9, 0.1, 0.8, 0.2]
    outcomes = [1, 0, 1, 0]
    market = [0.5, 0.5, 0.5, 0.5]
    sv = score_vector(probs, outcomes, market)
    assert not sv.dominates(sv)


def test_score_vector_uniformly_better_dominates():
    from clio.frozen.scoring import ScoreVector

    a = ScoreVector(
        neg_brier=-0.1, neg_ece=-0.05, resolution=0.2,
        total_return=0.15, sharpe=1.0, neg_max_dd=-0.05,
        coverage=1.0, regime_breadth=3,
    )
    b = ScoreVector(
        neg_brier=-0.2, neg_ece=-0.10, resolution=0.1,
        total_return=0.05, sharpe=0.5, neg_max_dd=-0.10,
        coverage=0.9, regime_breadth=2,
    )
    assert a.dominates(b)
    assert not b.dominates(a)


def test_score_vector_no_dominance_when_tradeoff():
    from clio.frozen.scoring import ScoreVector

    # a has better Brier, b has better Sharpe. Neither dominates.
    a = ScoreVector(
        neg_brier=-0.10, neg_ece=0.0, resolution=0.1,
        total_return=0.05, sharpe=0.5, neg_max_dd=0.0,
        coverage=1.0, regime_breadth=1,
    )
    b = ScoreVector(
        neg_brier=-0.15, neg_ece=0.0, resolution=0.1,
        total_return=0.10, sharpe=1.5, neg_max_dd=0.0,
        coverage=1.0, regime_breadth=1,
    )
    assert not a.dominates(b)
    assert not b.dominates(a)
