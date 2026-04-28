"""Generalized parametric strategies for prediction-market backtesting.

A `ParametricStrategy` is a vector of knobs:
- shrink_lambda_by_qtype: per-question-type shrinkage toward base rate
- trend_alpha: momentum coefficient
- time_decay: exponentially less aggressive shrink as we approach close
- vol_threshold: skip markets with |price - prev_price| > threshold (regime change)
- edge_threshold: minimum |strategy - market| to take a trade
- kelly_fraction: confidence-weighted sizing parameter
- symmetric: also long under-priced certainties (P(YES) base_rate >> market)

These strategies subsume v2, v3, v6 from earlier iterations and are designed
to be searched over by `tuner.py` or evolved by `evolve.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from clio.frozen.harness import Market


@dataclass(frozen=True)
class StrategyParams:
    """Fully specified strategy, hashable and pickleable."""
    shrink_lambda: tuple[tuple[str, float], ...] = (
        ("event", 0.20), ("field", 0.30), ("deadline", 0.0), ("durative", 0.0),
    )
    trend_alpha: float = 0.0
    time_decay: float = 0.0  # 0=no decay, 1=full decay
    edge_threshold: float = 0.05
    kelly_fraction: float = 0.0  # 0=fixed notional, 1=full Kelly
    notional: float = 0.05  # base notional fraction of bankroll
    symmetric: bool = False  # also long under-priced events

    @property
    def lambda_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.shrink_lambda}

    def to_dict(self) -> dict:
        return {
            "shrink_lambda": dict(self.shrink_lambda),
            "trend_alpha": self.trend_alpha,
            "time_decay": self.time_decay,
            "edge_threshold": self.edge_threshold,
            "kelly_fraction": self.kelly_fraction,
            "notional": self.notional,
            "symmetric": self.symmetric,
        }


def predict(
    params: StrategyParams,
    market: Market,
    step: int,
    qtype: str,
    base_rates: dict[str, float],
) -> float:
    """Compute the strategy's predicted probability at this as_of."""
    cur = market.market_prices[market.timeline[step]]
    if step > 0:
        prev = market.market_prices[market.timeline[step - 1]]
        trended = cur + params.trend_alpha * (cur - prev)
    else:
        trended = cur
    trended = min(0.99, max(0.01, trended))

    lam_base = params.lambda_dict.get(qtype, 0.0)
    if lam_base == 0:
        return trended

    # Time decay: scale lambda by (1 - step / total_steps) ** decay_exponent
    n_steps = len(market.timeline)
    progress = step / max(1, n_steps - 1) if n_steps > 1 else 0
    decay = (1 - progress) ** params.time_decay if params.time_decay > 0 else 1.0
    lam = lam_base * decay
    base = base_rates.get(qtype, 0.0)
    pred = (1 - lam) * trended + lam * base
    return min(0.99, max(0.01, pred))


@dataclass
class TradeRecord:
    market_id: str
    qtype: str
    step: int
    pred: float
    mkt: float
    edge: float
    notional: float
    outcome: int
    payoff: float
    question: str = ""


def simulate(
    params: StrategyParams,
    markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    base_rates: dict[str, float],
) -> dict:
    """Run a full backtest, return Brier + PnL + trade list."""
    sq: list[float] = []
    by_type_sq: dict[str, list[float]] = {}
    pnls: list[float] = []
    trades: list[TradeRecord] = []

    for m in markets:
        outcome = resolutions[m.market_id]
        qtype = qtypes[m.market_id]
        for step in range(len(m.timeline)):
            pred = predict(params, m, step, qtype, base_rates)
            err = (pred - outcome) ** 2
            sq.append(err)
            by_type_sq.setdefault(qtype, []).append(err)

            mkt = m.market_prices[m.timeline[step]]
            edge = pred - mkt
            if not (0.02 < mkt < 0.98) or abs(edge) < params.edge_threshold:
                continue

            # Confidence-weighted sizing. When edge is large, scale up.
            # Kelly for binary YES bet at price p with our prob p_hat:
            #   f* = (p_hat * b - (1 - p_hat)) / b   where b = (1 - p) / p
            # We use kelly_fraction × f* as the bet size, capped at notional max.
            if edge > 0:
                # Buy YES at mkt
                if mkt <= 0:
                    continue
                b = (1 - mkt) / mkt
                f_star = max(0.0, (pred * b - (1 - pred)) / b)
                size = params.notional + params.kelly_fraction * f_star * params.notional
                size = min(size, 0.10)
                if outcome == 1:
                    payoff = b * size
                else:
                    payoff = -size
            else:
                # Skip the YES side, but consider NO if either symmetric or
                # the NO bet has positive edge.
                if not params.symmetric and edge > -params.edge_threshold:
                    continue
                if mkt >= 1:
                    continue
                b = mkt / (1 - mkt)
                f_star = max(0.0, ((1 - pred) * b - pred) / b)
                size = params.notional + params.kelly_fraction * f_star * params.notional
                size = min(size, 0.10)
                if outcome == 0:
                    payoff = b * size
                else:
                    payoff = -size

            pnls.append(payoff)
            trades.append(
                TradeRecord(
                    market_id=m.market_id, qtype=qtype, step=step,
                    pred=pred, mkt=mkt, edge=edge, notional=size,
                    outcome=outcome, payoff=payoff, question=m.question,
                )
            )

    return {
        "brier_overall": sum(sq) / len(sq) if sq else 0.0,
        "brier_by_qtype": {t: sum(s) / len(s) for t, s in by_type_sq.items()},
        "pnl": sum(pnls),
        "n_trades": len(trades),
        "n_wins": sum(1 for p in pnls if p > 0),
        "trades": trades,
    }
