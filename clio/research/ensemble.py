"""Pareto-aware strategy ensembles.

The lesson from autoresearch's design doc: don't pick "the best" strategy —
pick the non-dominated set and use them together. Two ensembles supported:

- `MeanEnsemble`: arithmetic mean of member predictions.
- `BrierWeightedEnsemble`: members weighted inverse-proportional to their
  train Brier (better members count more).

The ensemble can be more conservative (lower variance) than its members,
which is exactly what we want for risk-controlled live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from clio.frozen.harness import Market
from clio.research.strategies import StrategyParams, predict, simulate


@dataclass
class EnsembleResult:
    member_params: list[StrategyParams]
    member_weights: list[float]
    description: str


def _ensemble_predict(
    members: list[StrategyParams],
    weights: list[float],
    market: Market,
    step: int,
    qtype: str,
    base_rates: dict[str, float],
) -> float:
    total = 0.0
    w_sum = 0.0
    for p, w in zip(members, weights):
        total += w * predict(p, market, step, qtype, base_rates)
        w_sum += w
    return total / w_sum if w_sum > 0 else 0.5


def simulate_ensemble(
    members: list[StrategyParams],
    weights: list[float],
    edge_threshold: float,
    notional: float,
    markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    base_rates: dict[str, float],
) -> dict:
    """Run the ensemble end-to-end. Edge/threshold/notional are at the
    ensemble level, not per-member."""
    sq = []
    pnls = []
    trades = []
    by_type_sq: dict[str, list[float]] = {}

    for m in markets:
        outcome = resolutions[m.market_id]
        qtype = qtypes[m.market_id]
        for step in range(len(m.timeline)):
            pred = _ensemble_predict(members, weights, m, step, qtype, base_rates)
            err = (pred - outcome) ** 2
            sq.append(err)
            by_type_sq.setdefault(qtype, []).append(err)

            mkt = m.market_prices[m.timeline[step]]
            edge = pred - mkt
            if not (0.02 < mkt < 0.98) or abs(edge) < edge_threshold:
                continue

            if edge > 0:
                b = (1 - mkt) / mkt
                payoff = b * notional if outcome == 1 else -notional
            else:
                b = mkt / (1 - mkt)
                payoff = b * notional if outcome == 0 else -notional
            pnls.append(payoff)
            trades.append({
                "market_id": m.market_id, "qtype": qtype, "step": step,
                "pred": pred, "mkt": mkt, "edge": edge,
                "outcome": outcome, "payoff": payoff,
                "question": m.question,
            })
    return {
        "brier_overall": sum(sq) / len(sq) if sq else 0,
        "brier_by_qtype": {t: sum(s) / len(s) for t, s in by_type_sq.items()},
        "pnl": sum(pnls),
        "n_trades": len(trades),
        "n_wins": sum(1 for p in pnls if p > 0),
        "trades": trades,
    }


def pareto_filter_strategies(
    candidates: list[tuple[StrategyParams, dict]],
    axes: tuple[str, ...] = ("brier_overall", "pnl"),
    minimize: tuple[bool, ...] = (True, False),
    max_keep: int = 10,
) -> list[tuple[StrategyParams, dict]]:
    """Return the Pareto-non-dominated set on the given axes."""
    def axis_value(res: dict, axis: str) -> float:
        return res.get(axis, 0.0)

    keep = []
    for i, (p_i, r_i) in enumerate(candidates):
        dominated = False
        for j, (p_j, r_j) in enumerate(candidates):
            if i == j:
                continue
            all_ge = True
            any_gt = False
            for axis, mini in zip(axes, minimize):
                vi = axis_value(r_i, axis)
                vj = axis_value(r_j, axis)
                if mini:
                    vi, vj = -vi, -vj  # convert to "higher = better"
                if vj < vi:
                    all_ge = False
                    break
                if vj > vi:
                    any_gt = True
            if all_ge and any_gt:
                dominated = True
                break
        if not dominated:
            keep.append((p_i, r_i))
    keep.sort(key=lambda x: -x[1].get("pnl", 0))
    return keep[:max_keep]


def make_brier_weighted_ensemble(
    members: list[tuple[StrategyParams, dict]],
) -> tuple[list[StrategyParams], list[float]]:
    """Build Brier-weighted ensemble: better train Brier → larger weight."""
    if not members:
        return [], []
    briers = [max(1e-6, r["brier_overall"]) for _, r in members]
    inv = [1 / b for b in briers]
    total = sum(inv)
    weights = [w / total for w in inv]
    params = [p for p, _ in members]
    return params, weights
