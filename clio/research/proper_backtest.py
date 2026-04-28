"""Proper backtest: one bet per market, compounded bankroll, real costs.

The previous `simulate()` counts every (market, as_of) snapshot as an
independent trade. Real trading isn't like that — you open a position
once and hold to resolution. Re-betting at every snapshot pyramids the
same view.

This module fixes that. Each market gets at most one trade:
  - Walk through timeline snapshots in order
  - At the first snapshot where |edge| > threshold, OPEN a position
  - HOLD to resolution
  - Realize the actual outcome at close

Adds:
  - Compounded bankroll (size = % of CURRENT, not initial)
  - Round-trip costs (spread + fees)
  - Liquidity cap (size limited to fraction of market volume)
  - Equity curve, Sharpe, Max DD, Profit Factor, Calmar
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

from clio.frozen.harness import Market
from clio.research.strategies import StrategyParams, predict


@dataclass
class CostModel:
    spread_bps: float = 150.0     # 1.5% half-spread on entry alone (Polymarket-typical)
    fee_bps: float = 0.0          # Polymarket waives most retail fees
    slippage_bps: float = 30.0    # additional adverse fill, scales with size

    def round_trip_cost_pct(self, size_frac_of_book: float = 0.0) -> float:
        return (self.spread_bps + self.fee_bps + self.slippage_bps * size_frac_of_book) / 10_000


@dataclass
class TradeFill:
    market_id: str
    qtype: str
    open_date: date
    close_date: date
    side: str  # "YES" or "NO"
    open_price: float          # market mid at open
    fill_price: float          # actual price after spread/slippage
    pred: float                # strategy's prediction at open
    edge: float                # strategy_pred - market_mid
    size_pct: float            # size as % of bankroll at open time
    notional_dollars: float    # size in dollars
    outcome: int
    payoff_dollars: float      # net of costs
    return_pct: float          # payoff / notional
    bankroll_at_open: float
    bankroll_at_close: float
    days_held: int
    question: str = ""


@dataclass
class BacktestResult:
    initial_bankroll: float
    final_bankroll: float
    total_return: float
    n_markets: int
    n_trades: int
    n_wins: int
    hit_rate: float
    avg_win_dollars: float
    avg_loss_dollars: float
    profit_factor: float
    sharpe_annual: float
    max_drawdown_pct: float
    calmar: float
    avg_days_held: float
    capital_deployed_pct: float  # avg of position notional / bankroll across trades
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
    trades: list[TradeFill] = field(default_factory=list)
    cost_model: CostModel | None = None


def proper_backtest(
    params: StrategyParams,
    markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    base_rates: dict[str, float],
    *,
    initial_bankroll: float = 10_000.0,
    max_position_pct: float = 0.02,    # 2% max per single position
    max_gross_exposure_pct: float = 0.30,  # 30% max total deployed at once
    cost_model: CostModel | None = None,
    liquidity_cap_pct: float = 0.003,  # max bet = 0.3% of market lifetime volume
    market_volumes: dict[str, float] | None = None,
) -> BacktestResult:
    """One-bet-per-market chronological backtest with real costs.

    The bankroll starts at `initial_bankroll`. At each (market, as_of) in
    chronological order:
      - If we have NO open position in this market, check for entry signal.
      - If signal fires, open a position. Subtract entry cost from bankroll
        immediately so it's reflected in equity curve.
      - When the market resolves, realize PnL.
    """
    cost_model = cost_model or CostModel()
    market_volumes = market_volumes or {}

    # Build a chronological event stream of (date, kind, market) where
    # kind ∈ {"snapshot", "resolution"}.
    events: list[tuple[date, str, Market]] = []
    for m in markets:
        for d in m.timeline:
            if d < m.closes_at:
                events.append((d, "snapshot", m))
        events.append((m.closes_at, "resolve", m))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "snapshot" else 1))

    bankroll = initial_bankroll
    open_positions: dict[str, dict] = {}  # market_id -> position info
    trades: list[TradeFill] = []
    equity_curve: list[tuple[date, float]] = [(events[0][0], bankroll)] if events else []

    def gross_exposure() -> float:
        return sum(p["notional"] for p in open_positions.values())

    for d, kind, m in events:
        if kind == "snapshot":
            if m.market_id in open_positions:
                continue
            qtype = qtypes[m.market_id]
            step = m.timeline.index(d)
            pred = predict(params, m, step, qtype, base_rates)
            mkt = m.market_prices[d]
            edge = pred - mkt

            if not (0.02 < mkt < 0.98) or abs(edge) < params.edge_threshold:
                continue

            # Position sizing.
            if edge > 0:
                b = (1 - mkt) / mkt
                f_star = max(0.0, (pred * b - (1 - pred)) / b)
                side = "YES"
            else:
                b = mkt / (1 - mkt)
                f_star = max(0.0, ((1 - pred) * b - pred) / b)
                side = "NO"
            size_frac = params.notional + params.kelly_fraction * f_star * params.notional
            size_frac = min(size_frac, max_position_pct)

            # Liquidity cap.
            vol = market_volumes.get(m.market_id, 0.0)
            if vol > 0 and bankroll > 0:
                vol_cap_dollars = vol * liquidity_cap_pct
                vol_cap_pct = vol_cap_dollars / bankroll
                size_frac = min(size_frac, vol_cap_pct)

            if size_frac < 0.001:  # not worth trading dust
                continue

            notional = size_frac * bankroll

            # Global gross-exposure cap: don't over-leverage across concurrent positions.
            available = max_gross_exposure_pct * bankroll - gross_exposure()
            if available <= 0:
                continue
            notional = min(notional, available)
            size_frac = notional / bankroll
            if size_frac < 0.001:
                continue
            # Apply spread + slippage on entry. Estimate book fraction as
            # notional / (vol * 1) ≈ tiny for our sizes; we use a fixed
            # half-spread cost applied as immediate equity hit on entry.
            entry_cost_pct = cost_model.round_trip_cost_pct(0.0) / 2  # half on entry
            entry_cost = notional * entry_cost_pct

            if side == "YES":
                fill_price = min(0.99, mkt + entry_cost_pct)
            else:
                fill_price = max(0.01, mkt - entry_cost_pct)

            # Reserve cash for the position. Cash decreases by notional;
            # equity stays flat (we hold a position worth ~notional at fill).
            cash_before = bankroll
            bankroll -= notional + entry_cost
            open_positions[m.market_id] = {
                "open_date": d,
                "side": side,
                "open_price": mkt,
                "fill_price": fill_price,
                "pred": pred,
                "edge": edge,
                "size_frac": size_frac,
                "notional": notional,
                "qtype": qtype,
                "bankroll_at_open": cash_before,
                "entry_cost": entry_cost,
            }
            # Equity = cash + sum(open notional) — mark open positions at fill.
            equity_curve.append((d, bankroll + sum(p["notional"] for p in open_positions.values())))

        else:  # resolution
            if m.market_id not in open_positions:
                # No open position — equity curve carried forward.
                equity_curve.append((d, bankroll + sum(p["notional"] for p in open_positions.values())))
                continue
            pos = open_positions.pop(m.market_id)
            outcome = resolutions.get(m.market_id)
            if outcome is None:
                bankroll += pos["notional"]
                equity_curve.append((d, bankroll + sum(p["notional"] for p in open_positions.values())))
                continue

            # Payoff. For YES @ fill f, notional buys notional/f shares;
            # each pays $1 if outcome=1.
            f = pos["fill_price"]
            if pos["side"] == "YES":
                gross = pos["notional"] / f if outcome == 1 else 0.0
            else:
                gross = pos["notional"] / (1 - f) if outcome == 0 else 0.0

            exit_cost = pos["notional"] * (cost_model.fee_bps / 10_000)
            net_pnl = gross - pos["notional"] - exit_cost
            # Cash returns: original notional reserved at open + realized P&L.
            bankroll += pos["notional"] + net_pnl
            trades.append(TradeFill(
                market_id=m.market_id, qtype=pos["qtype"],
                open_date=pos["open_date"], close_date=d,
                side=pos["side"], open_price=pos["open_price"], fill_price=pos["fill_price"],
                pred=pos["pred"], edge=pos["edge"],
                size_pct=pos["size_frac"], notional_dollars=pos["notional"],
                outcome=outcome, payoff_dollars=net_pnl,
                return_pct=net_pnl / pos["notional"] if pos["notional"] > 0 else 0.0,
                bankroll_at_open=pos["bankroll_at_open"],
                bankroll_at_close=bankroll,
                days_held=(d - pos["open_date"]).days,
                question=m.question,
            ))
            equity_curve.append((d, bankroll + sum(p["notional"] for p in open_positions.values())))

    # ---- metrics ----
    n_wins = sum(1 for t in trades if t.payoff_dollars > 0)
    wins_pnl = [t.payoff_dollars for t in trades if t.payoff_dollars > 0]
    losses_pnl = [t.payoff_dollars for t in trades if t.payoff_dollars <= 0]

    profit_factor = (
        sum(wins_pnl) / abs(sum(losses_pnl)) if losses_pnl and abs(sum(losses_pnl)) > 0 else float("inf")
    )

    # Daily returns from equity curve (compounded).
    daily_eq: dict[date, float] = {}
    for d, e in equity_curve:
        daily_eq[d] = e  # last value of the day wins
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

    # Max drawdown
    peak = initial_bankroll
    max_dd_pct = 0.0
    for _, e in equity_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd)

    total_return = (bankroll - initial_bankroll) / initial_bankroll
    if max_dd_pct > 0:
        calmar = total_return / max_dd_pct
    else:
        calmar = float("inf") if total_return > 0 else 0.0

    return BacktestResult(
        initial_bankroll=initial_bankroll,
        final_bankroll=bankroll,
        total_return=total_return,
        n_markets=len(markets),
        n_trades=len(trades),
        n_wins=n_wins,
        hit_rate=n_wins / len(trades) if trades else 0.0,
        avg_win_dollars=statistics.mean(wins_pnl) if wins_pnl else 0.0,
        avg_loss_dollars=statistics.mean(losses_pnl) if losses_pnl else 0.0,
        profit_factor=profit_factor,
        sharpe_annual=sharpe,
        max_drawdown_pct=max_dd_pct,
        calmar=calmar,
        avg_days_held=statistics.mean([t.days_held for t in trades]) if trades else 0.0,
        capital_deployed_pct=statistics.mean([t.size_pct for t in trades]) if trades else 0.0,
        equity_curve=equity_curve,
        trades=trades,
        cost_model=cost_model,
    )
