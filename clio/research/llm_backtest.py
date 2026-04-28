"""Backtest using LLM forecasts as the strategy's predictions.

Replaces the rule-based predict() with LLM calls. Caches forecasts to disk
so re-runs are free.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from clio.agents.llm_anthropic import AnthropicLLMClient, LLMForecaster
from clio.frozen.corpus import Corpus
from clio.frozen.harness import Market
from clio.research.proper_backtest import (
    BacktestResult, CostModel, TradeFill,
)


log = logging.getLogger(__name__)


class ForecastCache:
    """Disk cache for LLM forecasts, keyed by (market_id, as_of, model)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    def _key(self, market_id: str, as_of: date, model: str) -> str:
        return f"{model}::{market_id}::{as_of.isoformat()}"

    def get(self, market_id: str, as_of: date, model: str) -> dict | None:
        return self.data.get(self._key(market_id, as_of, model))

    def put(self, market_id: str, as_of: date, model: str, forecast: dict) -> None:
        self.data[self._key(market_id, as_of, model)] = forecast

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self.data)


def precompute_llm_forecasts(
    markets: list[Market],
    qtypes: dict[str, str],
    base_rates: dict[str, float],
    corpus: Corpus,
    forecaster: LLMForecaster,
    cache: ForecastCache,
    *,
    save_every: int = 50,
    only_first_step: bool = True,
    short_horizon_days: int | None = None,
) -> int:
    """Precompute LLM forecasts for every (market, as_of). Caches each result.

    `only_first_step=True` makes one forecast per market (at the earliest as_of),
    matching one-bet-per-market backtest logic and minimizing API cost.
    """
    n_calls = 0
    n_cached_hits = 0
    for i, m in enumerate(markets, 1):
        if short_horizon_days is not None:
            lifetime_days = (m.closes_at - m.observed_at).days
            if lifetime_days > short_horizon_days:
                continue

        steps = [m.timeline[0]] if only_first_step else list(m.timeline)
        qtype = qtypes[m.market_id]
        base_rate = base_rates.get(qtype, 0.30)
        for as_of in steps:
            if as_of >= m.closes_at:
                continue
            cached = cache.get(m.market_id, as_of, forecaster.llm.model)
            if cached is not None:
                n_cached_hits += 1
                continue
            res = forecaster.forecast(m, as_of, corpus, base_rate_hint=base_rate)
            cache.put(m.market_id, as_of, forecaster.llm.model, {
                "p_yes": res.p_yes, "confidence": res.confidence,
                "reasoning": res.reasoning,
            })
            n_calls += 1
            if n_calls % save_every == 0:
                cache.save()
                u = forecaster.llm.usage_summary()
                log.info(
                    f"[{i}/{len(markets)}] {n_calls} new calls, {n_cached_hits} cache hits, "
                    f"~${u['estimated_cost_usd']:.2f} spent"
                )
    cache.save()
    return n_calls


def llm_backtest(
    markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    cache: ForecastCache,
    model: str,
    *,
    edge_threshold: float = 0.10,
    initial_bankroll: float = 100_000.0,
    max_position_pct: float = 0.25,
    max_gross_exposure_pct: float = 1.5,
    kelly_fraction: float = 1.2,
    notional_floor: float = 0.10,
    cost_model: CostModel | None = None,
    market_volumes: dict[str, float] | None = None,
    liquidity_cap_pct: float = 0.005,
    short_horizon_days: int | None = None,
    skip_low_confidence: bool = False,
) -> BacktestResult:
    """One-bet-per-market backtest using cached LLM forecasts."""
    cost_model = cost_model or CostModel()
    market_volumes = market_volumes or {}

    # Build event stream.
    events: list[tuple[date, str, Market]] = []
    used_markets = []
    for m in markets:
        if short_horizon_days is not None:
            lifetime = (m.closes_at - m.observed_at).days
            if lifetime > short_horizon_days:
                continue
        used_markets.append(m)
        for d in m.timeline:
            if d < m.closes_at:
                events.append((d, "snapshot", m))
        events.append((m.closes_at, "resolve", m))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "snapshot" else 1))

    bankroll = initial_bankroll
    open_positions: dict[str, dict] = {}
    trades: list[TradeFill] = []
    equity_curve: list[tuple[date, float]] = [(events[0][0], bankroll)] if events else []

    def gross_exposure() -> float:
        return sum(p["notional"] for p in open_positions.values())

    for d, kind, m in events:
        if kind == "snapshot":
            if m.market_id in open_positions:
                continue
            qtype = qtypes[m.market_id]
            cached = cache.get(m.market_id, d, model)
            if cached is None:
                continue
            if skip_low_confidence and cached.get("confidence") == "low":
                continue
            pred = float(cached["p_yes"])
            mkt = m.market_prices[d]
            edge = pred - mkt
            if not (0.02 < mkt < 0.98) or abs(edge) < edge_threshold:
                continue

            if edge > 0:
                b = (1 - mkt) / mkt
                f_star = max(0.0, (pred * b - (1 - pred)) / b)
                side = "YES"
            else:
                b = mkt / (1 - mkt)
                f_star = max(0.0, ((1 - pred) * b - pred) / b)
                side = "NO"
            size_frac = notional_floor + kelly_fraction * f_star * notional_floor
            size_frac = min(size_frac, max_position_pct)

            vol = market_volumes.get(m.market_id, 0.0)
            if vol > 0 and bankroll > 0:
                vol_cap_pct = (vol * liquidity_cap_pct) / bankroll
                size_frac = min(size_frac, vol_cap_pct)

            if size_frac < 0.001:
                continue

            notional = size_frac * bankroll
            available = max_gross_exposure_pct * bankroll - gross_exposure()
            if available <= 0:
                continue
            notional = min(notional, available)
            size_frac = notional / bankroll
            if size_frac < 0.001:
                continue

            entry_cost_pct = cost_model.round_trip_cost_pct(0.0) / 2
            entry_cost = notional * entry_cost_pct
            if side == "YES":
                fill = min(0.99, mkt + entry_cost_pct)
            else:
                fill = max(0.01, mkt - entry_cost_pct)

            cash_before = bankroll
            bankroll -= notional + entry_cost
            open_positions[m.market_id] = {
                "open_date": d, "side": side, "open_price": mkt,
                "fill_price": fill, "pred": pred, "edge": edge,
                "size_frac": size_frac, "notional": notional, "qtype": qtype,
                "bankroll_at_open": cash_before, "entry_cost": entry_cost,
            }
            equity_curve.append((d, bankroll + sum(p["notional"] for p in open_positions.values())))

        else:  # resolution
            if m.market_id not in open_positions:
                equity_curve.append((d, bankroll + sum(p["notional"] for p in open_positions.values())))
                continue
            pos = open_positions.pop(m.market_id)
            outcome = resolutions.get(m.market_id)
            if outcome is None:
                bankroll += pos["notional"]
                equity_curve.append((d, bankroll + sum(p["notional"] for p in open_positions.values())))
                continue

            f = pos["fill_price"]
            if pos["side"] == "YES":
                gross = pos["notional"] / f if outcome == 1 else 0.0
            else:
                gross = pos["notional"] / (1 - f) if outcome == 0 else 0.0
            exit_cost = pos["notional"] * (cost_model.fee_bps / 10_000)
            net_pnl = gross - pos["notional"] - exit_cost
            bankroll += pos["notional"] + net_pnl

            trades.append(TradeFill(
                market_id=m.market_id, qtype=pos["qtype"],
                open_date=pos["open_date"], close_date=d,
                side=pos["side"], open_price=pos["open_price"],
                fill_price=pos["fill_price"], pred=pos["pred"],
                edge=pos["edge"], size_pct=pos["size_frac"],
                notional_dollars=pos["notional"],
                outcome=outcome, payoff_dollars=net_pnl,
                return_pct=net_pnl / pos["notional"] if pos["notional"] > 0 else 0.0,
                bankroll_at_open=pos["bankroll_at_open"],
                bankroll_at_close=bankroll,
                days_held=(d - pos["open_date"]).days,
                question=m.question,
            ))
            equity_curve.append((d, bankroll + sum(p["notional"] for p in open_positions.values())))

    # Metrics — copy the proper_backtest logic.
    import statistics
    n_wins = sum(1 for t in trades if t.payoff_dollars > 0)
    wins_pnl = [t.payoff_dollars for t in trades if t.payoff_dollars > 0]
    losses_pnl = [t.payoff_dollars for t in trades if t.payoff_dollars <= 0]
    profit_factor = (
        sum(wins_pnl) / abs(sum(losses_pnl)) if losses_pnl and abs(sum(losses_pnl)) > 0 else float("inf")
    )

    daily_eq: dict[date, float] = {}
    for d, e in equity_curve:
        daily_eq[d] = e
    daily_dates = sorted(daily_eq)
    daily_returns: list[float] = []
    for i in range(1, len(daily_dates)):
        prev = daily_eq[daily_dates[i - 1]]
        cur = daily_eq[daily_dates[i]]
        if prev > 0:
            daily_returns.append((cur - prev) / prev)

    if daily_returns:
        mu = statistics.mean(daily_returns)
        sd = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0
        sharpe = (mu / sd * math.sqrt(252)) if sd > 0 else 0.0
    else:
        sharpe = 0.0

    peak = initial_bankroll
    max_dd_pct = 0.0
    for _, e in equity_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd)
    total_return = (bankroll - initial_bankroll) / initial_bankroll
    calmar = total_return / max_dd_pct if max_dd_pct > 0 else (float("inf") if total_return > 0 else 0.0)

    return BacktestResult(
        initial_bankroll=initial_bankroll, final_bankroll=bankroll,
        total_return=total_return,
        n_markets=len(used_markets), n_trades=len(trades), n_wins=n_wins,
        hit_rate=n_wins / len(trades) if trades else 0.0,
        avg_win_dollars=statistics.mean(wins_pnl) if wins_pnl else 0.0,
        avg_loss_dollars=statistics.mean(losses_pnl) if losses_pnl else 0.0,
        profit_factor=profit_factor,
        sharpe_annual=sharpe,
        max_drawdown_pct=max_dd_pct,
        calmar=calmar,
        avg_days_held=statistics.mean([t.days_held for t in trades]) if trades else 0.0,
        capital_deployed_pct=statistics.mean([t.size_pct for t in trades]) if trades else 0.0,
        equity_curve=equity_curve, trades=trades, cost_model=cost_model,
    )
