"""Push the strategy to its actual limit.

Question: if I let the sizing rip — full Kelly, big single-position cap,
high gross exposure — and add a high-conviction filter, what's the actual
ceiling of this strategy on the same 638-market dataset?

Five tiers of aggression, same strategy parameters, same dataset:

  conservative   2% max,   30% gross, 0.5 Kelly  ← current "production" sizing
  moderate       5% max,   60% gross, 1.0 Kelly
  aggressive    10% max,   80% gross, 1.0 Kelly
  full_kelly    20% max,  100% gross, 1.0 Kelly
  yolo          30% max,  150% gross, 1.5 Kelly  ← ruin-likely

Plus a "high_conviction" tier that filters trades to only the highest-edge
ones AND sizes aggressively. Theoretically this should produce the best
risk-adjusted return.

For each: equity curve, CAGR, Sharpe, Max DD, ruin probability, time-to-ruin
distribution from monte-carlo bootstrap.
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

from clio.frozen.harness import Market
from clio.research.proper_backtest import (
    BacktestResult, CostModel, TradeFill, proper_backtest,
)
from clio.research.strategies import StrategyParams


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("aggressive")


def load_v3():
    with open("runs/live_iter/markets_v3.json") as f:
        payload = json.load(f)
    markets, qtypes = [], {}
    for d in payload["markets"]:
        m = Market(
            market_id=d["market_id"], question=d["question"], regime=d["regime"],
            observed_at=date.fromisoformat(d["observed_at"]),
            closes_at=date.fromisoformat(d["closes_at"]),
            timeline=tuple(date.fromisoformat(s) for s in d["timeline"]),
            market_prices={date.fromisoformat(k): float(v) for k, v in d["market_prices"].items()},
        )
        markets.append(m)
        qtypes[m.market_id] = d.get("qtype", "event")
    resolutions = {k: int(v) for k, v in payload["resolutions"].items()}
    return markets, qtypes, resolutions


def load_volumes():
    p = Path("runs/live_iter/market_volumes.json")
    return json.loads(p.read_text()) if p.exists() else {}


def base_strategy() -> StrategyParams:
    return StrategyParams(
        shrink_lambda=(("event", 0.5), ("field", 0.49), ("deadline", 0.0), ("durative", 0.09)),
        trend_alpha=-0.30,
        time_decay=0.0,
        edge_threshold=0.05,
        kelly_fraction=1.0,  # tier overrides this
        notional=0.05,
        symmetric=False,
    )


def high_conviction_strategy() -> StrategyParams:
    """Tighter edge threshold — only high-conviction trades."""
    return StrategyParams(
        shrink_lambda=(("event", 0.5), ("field", 0.49), ("deadline", 0.0), ("durative", 0.09)),
        trend_alpha=-0.30,
        time_decay=0.0,
        edge_threshold=0.15,  # 3x higher than default
        kelly_fraction=1.0,
        notional=0.10,
        symmetric=False,
    )


# ---- proper Kelly sizing — patch into the backtest via a parameter override ----


def cagr(initial: float, final: float, days: int) -> float:
    if initial <= 0 or days <= 0:
        return 0.0
    return (final / initial) ** (365 / days) - 1


def annualized_monthly_growth(initial: float, final: float, days: int) -> float:
    """Equivalent monthly compounding rate."""
    if initial <= 0 or days <= 0 or final <= 0:
        return 0.0
    months = days / 30.44
    return (final / initial) ** (1 / months) - 1


def trading_days(eq):
    if not eq:
        return 0
    return (eq[-1][0] - eq[0][0]).days


def first_trade_date(trades):
    return min((t.open_date for t in trades), default=None)


def last_trade_date(trades):
    return max((t.close_date for t in trades), default=None)


def ascii_curve(eq, width=92, height=14):
    if not eq:
        return "(empty)"
    eq_sorted = sorted(eq, key=lambda x: x[0])
    if eq_sorted[-1][0] == eq_sorted[0][0]:
        return "(degenerate)"
    start, end = eq_sorted[0][0], eq_sorted[-1][0]
    span = (end - start).days
    samples = []
    last_v = eq_sorted[0][1]
    cur = 0
    for i in range(width):
        target = start.toordinal() + int(span * i / (width - 1))
        while cur < len(eq_sorted) and eq_sorted[cur][0].toordinal() <= target:
            last_v = eq_sorted[cur][1]
            cur += 1
        samples.append(last_v)
    lo, hi = min(samples), max(samples)
    if hi - lo < 1e-9:
        hi = lo + 1
    grid = [[" "] * width for _ in range(height)]
    initial = eq_sorted[0][1]
    for x, v in enumerate(samples):
        y = height - 1 - int((v - lo) / (hi - lo) * (height - 1))
        y = max(0, min(height - 1, y))
        grid[y][x] = "*"
    if lo <= initial <= hi:
        y0 = height - 1 - int((initial - lo) / (hi - lo) * (height - 1))
        for x in range(width):
            if grid[y0][x] == " ":
                grid[y0][x] = "·"
    out = []
    for y, row in enumerate(grid):
        v = lo + (hi - lo) * (height - 1 - y) / (height - 1)
        if v >= 1_000_000:
            label = f"${v/1_000_000:.1f}M"
        elif v >= 1_000:
            label = f"${v/1_000:.0f}K"
        else:
            label = f"${v:.0f}"
        out.append(f"  {label:>10}  |{''.join(row)}|")
    out.append("  " + " " * 10 + "  +" + "-" * width + "+")
    out.append(f"              {start}{' ' * (width - 22)}{end}")
    return "\n".join(out)


# ---- summary printer ----


def summarize(label: str, res: BacktestResult, days_active: int) -> dict:
    cag = cagr(res.initial_bankroll, res.final_bankroll, days_active) if days_active > 0 else 0
    monthly = annualized_monthly_growth(res.initial_bankroll, res.final_bankroll, days_active)
    multiplier = res.final_bankroll / res.initial_bankroll if res.initial_bankroll > 0 else 0
    log.info(f"\n=== {label} ===")
    log.info(f"  ${res.initial_bankroll:>11,.0f} → ${res.final_bankroll:>13,.0f}    "
             f"({multiplier:7,.2f}× over {days_active} active days)")
    log.info(f"  CAGR:               {100*cag:+8.2f}%")
    log.info(f"  monthly compound:   {100*monthly:+8.2f}%/mo")
    log.info(f"  trades: {res.n_trades:4d}   hit_rate: {100*res.hit_rate:5.1f}%   "
             f"profit factor: {res.profit_factor:5.2f}")
    log.info(f"  Sharpe:             {res.sharpe_annual:8.2f}")
    log.info(f"  Max DD:             {100*res.max_drawdown_pct:7.1f}%")
    log.info(f"  Calmar:             {res.calmar:8.2f}")
    log.info(f"  avg pos size:       {100*res.capital_deployed_pct:7.2f}% of bankroll")
    log.info(f"  avg days held:      {res.avg_days_held:7.0f}")
    return {
        "label": label,
        "initial": res.initial_bankroll, "final": res.final_bankroll,
        "multiplier": multiplier, "cagr": cag, "monthly": monthly,
        "sharpe": res.sharpe_annual, "max_dd": res.max_drawdown_pct,
        "calmar": res.calmar, "n_trades": res.n_trades,
        "hit_rate": res.hit_rate, "profit_factor": res.profit_factor,
        "avg_pos_pct": res.capital_deployed_pct,
    }


# ---- monte carlo ruin analysis ----


def monte_carlo_ruin(
    res: BacktestResult, n_runs: int = 5000, seed: int = 42, ruin_threshold: float = 0.10
) -> dict:
    """For a given strategy, what's the chance of bankroll falling to ruin if
    we resample the same trades in random order?

    We use the fractional return per trade (payoff / bankroll_at_open) and
    randomize their order. This bounds the path-dependent risk.
    """
    if not res.trades:
        return {"ruin_prob": 0.0, "p10_final_mult": 1.0, "p50_final_mult": 1.0, "p90_final_mult": 1.0}
    rng = random.Random(seed)
    # Per-trade fractional returns relative to bankroll at OPEN.
    fracs = [
        t.payoff_dollars / max(t.bankroll_at_open, 1.0)
        for t in res.trades
    ]
    finals = []
    ruined = 0
    for _ in range(n_runs):
        order = fracs.copy()
        rng.shuffle(order)
        bk = 1.0
        is_ruin = False
        for f in order:
            bk *= (1 + f)
            if bk < ruin_threshold:
                is_ruin = True
        if is_ruin:
            ruined += 1
        finals.append(bk)
    finals.sort()
    return {
        "ruin_prob": ruined / n_runs,
        "p10_final_mult": finals[int(n_runs * 0.10)],
        "p50_final_mult": finals[int(n_runs * 0.50)],
        "p90_final_mult": finals[int(n_runs * 0.90)],
        "p99_final_mult": finals[int(n_runs * 0.99)],
    }


# ---- main ----


def main() -> int:
    markets, qtypes, resolutions = load_v3()
    volumes = load_volumes()
    log.info(f"loaded {len(markets)} markets, {len(volumes)} volume records")

    base_rates = {"event": 0.204, "field": 0.085, "deadline": 0.324, "durative": 0.0}
    cost = CostModel()  # default 1.5% spread + 30bps slippage

    tiers = [
        # label,                strategy,                      max_pos, max_gross, kelly_mult, liq_cap
        ("conservative   (current)", base_strategy(),          0.02,    0.30,      0.5,        0.003),
        ("moderate       (Kelly 1×)", base_strategy(),         0.05,    0.60,      1.0,        0.005),
        ("aggressive     (10% pos)", base_strategy(),          0.10,    0.80,      1.0,        0.005),
        ("full_kelly     (20% pos)", base_strategy(),          0.20,    1.00,      1.0,        0.010),
        ("yolo           (30% pos)", base_strategy(),          0.30,    1.50,      1.5,        0.020),
        ("high_conviction (filter+10%)", high_conviction_strategy(), 0.15, 1.00,   1.0,        0.010),
        ("high_conviction_full (filter+25%)", high_conviction_strategy(), 0.25, 1.20, 1.5,    0.020),
    ]

    results = []
    raw_results = []
    for label, strat, max_pos, max_gross, kelly_mult, liq_cap in tiers:
        # Patch the strategy's kelly_fraction with the tier multiplier.
        s = StrategyParams(
            shrink_lambda=strat.shrink_lambda,
            trend_alpha=strat.trend_alpha,
            time_decay=strat.time_decay,
            edge_threshold=strat.edge_threshold,
            kelly_fraction=strat.kelly_fraction * kelly_mult,
            notional=strat.notional,
            symmetric=strat.symmetric,
        )
        res = proper_backtest(
            s, markets, qtypes, resolutions, base_rates,
            initial_bankroll=10_000,
            max_position_pct=max_pos,
            max_gross_exposure_pct=max_gross,
            cost_model=cost,
            liquidity_cap_pct=liq_cap,
            market_volumes=volumes,
        )
        ft = first_trade_date(res.trades)
        lt = last_trade_date(res.trades)
        active_days = (lt - ft).days if ft and lt else 0
        summary = summarize(label, res, active_days)
        mc = monte_carlo_ruin(res)
        log.info(f"  monte carlo (5000 reorderings):")
        log.info(f"    P(ruin to <10% of initial):  {100*mc['ruin_prob']:.2f}%")
        log.info(f"    p10 final multiplier:        {mc['p10_final_mult']:.2f}×")
        log.info(f"    p50 final multiplier:        {mc['p50_final_mult']:.2f}×")
        log.info(f"    p90 final multiplier:        {mc['p90_final_mult']:.2f}×")
        log.info(f"    p99 final multiplier:        {mc['p99_final_mult']:.2f}×")
        summary.update({"mc_" + k: v for k, v in mc.items()})
        results.append(summary)
        raw_results.append((label, res, active_days, mc))

    # Side-by-side
    log.info("\n" + "=" * 70)
    log.info("SIDE BY SIDE")
    log.info("=" * 70)
    log.info(f"  {'tier':<35} {'final':>13} {'mult':>9} {'CAGR':>9} {'mo':>9} {'Sh':>6} {'DD':>7} {'ruin%':>7}")
    log.info(f"  {'-'*35} {'-'*13} {'-'*9} {'-'*9} {'-'*9} {'-'*6} {'-'*7} {'-'*7}")
    for r in results:
        log.info(
            f"  {r['label']:<35} ${r['final']:>11,.0f} "
            f"{r['multiplier']:>7,.1f}× {100*r['cagr']:>+7.0f}% {100*r['monthly']:>+6.1f}%/m "
            f"{r['sharpe']:>5.2f} {100*r['max_dd']:>5.1f}% {100*r['mc_ruin_prob']:>5.1f}%"
        )

    # Equity curves for the most interesting tiers
    log.info("\n" + "=" * 70)
    log.info("EQUITY CURVES")
    log.info("=" * 70)
    for label, res, _, _ in raw_results:
        if "conservative" in label or "high_conviction_full" in label or "full_kelly" in label or "yolo" in label:
            log.info(f"\n--- {label} ---")
            log.info(ascii_curve(res.equity_curve))

    # Top winner across all tiers
    log.info("\n" + "=" * 70)
    log.info("TOP 5 WINNING TRADES (high_conviction_full)")
    log.info("=" * 70)
    hcf = [r for r in raw_results if "high_conviction_full" in r[0]][0][1]
    for t in sorted(hcf.trades, key=lambda t: -t.payoff_dollars)[:5]:
        log.info(
            f"  {t.open_date} [{t.qtype:<8}] {t.side:<3} mkt={t.open_price:.3f}→fill={t.fill_price:.3f}  "
            f"outcome={t.outcome}  pnl=${t.payoff_dollars:+,.0f}  "
            f"({100*t.return_pct:+.0f}% on ${t.notional_dollars:,.0f}, {t.days_held}d)"
        )
        log.info(f"     {t.question[:90]}")

    out = Path("runs/live_iter/aggressive_backtest.json")
    out.write_text(json.dumps({"tiers": results}, indent=2, default=str))
    log.info(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
