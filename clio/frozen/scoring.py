"""Scoring primitives.

These are the metrics that go into the Pareto frontier. None of them are
sufficient on their own; together they characterize a strategy.

Conventions:
- All probabilities are in [0, 1].
- Outcomes are 0 or 1 (binary markets only in MVP; multi-outcome is future).
- Higher is better for `score_vector` outputs (Brier and ECE are negated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import math


_EPS = 1e-9


def _clip(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of probability vs binary outcome. Lower is better."""
    if len(probs) != len(outcomes):
        raise ValueError("length mismatch")
    if not probs:
        return 0.0
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def calibration_ece(
    probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected calibration error. Lower is better. Returns NaN-safe 0 on empty."""
    if not probs:
        return 0.0
    if len(probs) != len(outcomes):
        raise ValueError("length mismatch")

    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in zip(probs, outcomes):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, o))

    total = len(probs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = sum(p for p, _ in b) / len(b)
        avg_o = sum(o for _, o in b) / len(b)
        ece += (len(b) / total) * abs(avg_p - avg_o)
    return ece


def resolution_decomposition(
    probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> tuple[float, float, float]:
    """Murphy decomposition: BS = REL - RES + UNC.

    REL (reliability): calibration error squared, lower is better.
    RES (resolution): how much information our forecasts carry vs the climatology.
                      Higher is better.
    UNC (uncertainty): variance of the marginal outcome — independent of forecaster.
    """
    if not probs:
        return 0.0, 0.0, 0.0
    n = len(probs)
    base_rate = sum(outcomes) / n
    unc = base_rate * (1 - base_rate)

    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in zip(probs, outcomes):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, o))

    rel = 0.0
    res = 0.0
    for b in bins:
        if not b:
            continue
        nk = len(b)
        avg_p = sum(p for p, _ in b) / nk
        avg_o = sum(o for _, o in b) / nk
        rel += (nk / n) * (avg_p - avg_o) ** 2
        res += (nk / n) * (avg_o - base_rate) ** 2

    return rel, res, unc


def kelly_pnl(
    probs: Sequence[float],
    outcomes: Sequence[int],
    market_prices: Sequence[float],
    kelly_fraction: float = 0.25,
    initial_bankroll: float = 10_000.0,
    max_position_frac: float = 0.05,
    cost_per_trade: float = 0.0,
) -> dict[str, float]:
    """Simulate fractional-Kelly betting and report bankroll trajectory.

    For each market: if our prob > market_price, we buy YES at market_price.
    Stake = max(0, kelly_fraction * (edge / odds)) capped at max_position_frac.
    Pure long only in MVP — no shorts.

    Returns dict with bankroll_final, max_dd, sharpe, hit_rate, n_trades.
    """
    n = len(probs)
    assert len(outcomes) == len(market_prices) == n, "length mismatch"

    bankroll = initial_bankroll
    history = [bankroll]
    rets: list[float] = []
    n_trades = 0
    hits = 0

    for p_hat, outcome, mkt in zip(probs, outcomes, market_prices):
        edge = p_hat - mkt
        if edge <= 0 or mkt <= 0 or mkt >= 1:
            history.append(bankroll)
            continue

        b = (1 - mkt) / mkt
        f_kelly = (p_hat * b - (1 - p_hat)) / b
        f = max(0.0, min(max_position_frac, kelly_fraction * f_kelly))
        if f == 0:
            history.append(bankroll)
            continue

        stake = bankroll * f
        if outcome == 1:
            payoff = stake * (1 - mkt) / mkt - cost_per_trade
            hits += 1
        else:
            payoff = -stake - cost_per_trade

        ret = payoff / bankroll
        rets.append(ret)
        bankroll += payoff
        history.append(bankroll)
        n_trades += 1

    peak = initial_bankroll
    max_dd = 0.0
    for x in history:
        peak = max(peak, x)
        dd = (peak - x) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    if rets:
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(1, len(rets) - 1)
        sd = math.sqrt(var)
        sharpe = (mu / sd) * math.sqrt(252) if sd > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "bankroll_final": bankroll,
        "bankroll_initial": initial_bankroll,
        "total_return": (bankroll - initial_bankroll) / initial_bankroll,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_trades": n_trades,
        "hit_rate": hits / n_trades if n_trades else 0.0,
    }


@dataclass(frozen=True)
class ScoreVector:
    """Multi-axis score for a strategy. Higher is better on every axis.

    Brier and ECE are negated so the dominance check is uniform direction.
    """

    neg_brier: float
    neg_ece: float
    resolution: float
    total_return: float
    sharpe: float
    neg_max_dd: float
    coverage: float
    regime_breadth: int = 1
    extra: dict[str, float] = field(default_factory=dict)

    def axes(self) -> tuple[float, ...]:
        return (
            self.neg_brier,
            self.neg_ece,
            self.resolution,
            self.total_return,
            self.sharpe,
            self.neg_max_dd,
            self.coverage,
            float(self.regime_breadth),
        )

    def dominates(self, other: "ScoreVector") -> bool:
        a = self.axes()
        b = other.axes()
        all_ge = all(x >= y for x, y in zip(a, b))
        any_gt = any(x > y for x, y in zip(a, b))
        return all_ge and any_gt


def score_vector(
    probs: Sequence[float],
    outcomes: Sequence[int],
    market_prices: Sequence[float],
    coverage: float = 1.0,
    regime_breadth: int = 1,
    cost_per_trade: float = 0.0,
) -> ScoreVector:
    """Compute the full score vector for a strategy on a holdout."""
    bs = brier_score(probs, outcomes)
    ece = calibration_ece(probs, outcomes)
    _, res, _ = resolution_decomposition(probs, outcomes)
    pnl = kelly_pnl(probs, outcomes, market_prices, cost_per_trade=cost_per_trade)
    return ScoreVector(
        neg_brier=-bs,
        neg_ece=-ece,
        resolution=res,
        total_return=pnl["total_return"],
        sharpe=pnl["sharpe"],
        neg_max_dd=-pnl["max_dd"],
        coverage=coverage,
        regime_breadth=regime_breadth,
        extra={
            "brier": bs,
            "ece": ece,
            "resolution": res,
            "n_trades": pnl["n_trades"],
            "hit_rate": pnl["hit_rate"],
            "bankroll_final": pnl["bankroll_final"],
        },
    )
