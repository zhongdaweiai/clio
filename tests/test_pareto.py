"""Pareto frontier tests."""

from clio.frozen.scoring import ScoreVector
from clio.pareto import pareto_frontier


def _sv(brier: float, sharpe: float, ret: float = 0.0) -> ScoreVector:
    return ScoreVector(
        neg_brier=-brier,
        neg_ece=0.0,
        resolution=0.0,
        total_return=ret,
        sharpe=sharpe,
        neg_max_dd=0.0,
        coverage=1.0,
        regime_breadth=1,
    )


def test_strictly_dominated_strategy_is_excluded():
    a = _sv(brier=0.20, sharpe=1.5)  # better Brier, better Sharpe
    b = _sv(brier=0.25, sharpe=1.0)  # dominated
    front = pareto_frontier([a, b])
    assert front == [0]


def test_no_strict_dominance_keeps_both():
    # A trades off accuracy for risk-adjusted return.
    a = _sv(brier=0.18, sharpe=0.8)  # better Brier
    b = _sv(brier=0.22, sharpe=1.5)  # better Sharpe
    front = pareto_frontier([a, b])
    assert set(front) == {0, 1}


def test_identical_scores_are_both_kept():
    a = _sv(brier=0.20, sharpe=1.0)
    b = _sv(brier=0.20, sharpe=1.0)
    # neither strictly dominates → both stay
    front = pareto_frontier([a, b])
    assert set(front) == {0, 1}


def test_three_way_with_one_dominated():
    a = _sv(brier=0.18, sharpe=0.8)
    b = _sv(brier=0.22, sharpe=1.5)
    c = _sv(brier=0.30, sharpe=0.5)  # dominated by both
    front = pareto_frontier([a, b, c])
    assert set(front) == {0, 1}


def test_empty_input():
    assert pareto_frontier([]) == []
