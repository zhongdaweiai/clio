"""Find the actual ceiling at the optimal edge filter (>=0.10).

The first extreme run identified edge>=0.10 as the sweet spot for filtering.
Going tighter (>=0.20) had too few trades; going looser (>=0.05) included
mediocre trades. Now sweep position sizing at the optimal edge filter.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from datetime import date
from pathlib import Path

from clio.frozen.harness import Market
from clio.research.proper_backtest import CostModel, proper_backtest
from clio.research.strategies import StrategyParams


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("extreme_v2")


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


def make_strat(edge_thr: float = 0.10) -> StrategyParams:
    return StrategyParams(
        shrink_lambda=(("event", 0.5), ("field", 0.49), ("deadline", 0.0), ("durative", 0.09)),
        trend_alpha=-0.30,
        time_decay=0.0,
        edge_threshold=edge_thr,
        kelly_fraction=1.0,
        notional=0.10,
        symmetric=False,
    )


def cagr(initial, final, days):
    if initial <= 0 or days <= 0 or final <= 0:
        return 0.0
    return (final / initial) ** (365 / days) - 1


def monthly(initial, final, days):
    if initial <= 0 or days <= 0 or final <= 0:
        return 0.0
    return (final / initial) ** (1 / (days / 30.44)) - 1


def first_last(trades):
    if not trades:
        return None, None
    return min(t.open_date for t in trades), max(t.close_date for t in trades)


def bootstrap(res, n_runs=5000, seed=13, ruin_thr=0.05):
    if not res.trades:
        return {}
    fracs = [t.payoff_dollars / max(t.bankroll_at_open, 1.0) for t in res.trades]
    n = len(fracs)
    rng = random.Random(seed)
    finals = []
    ruined = 0
    for _ in range(n_runs):
        bk = 1.0
        is_ruin = False
        for _ in range(n):
            f = rng.choice(fracs)
            bk *= (1 + f)
            if bk < ruin_thr:
                is_ruin = True
        if is_ruin:
            ruined += 1
        finals.append(bk)
    finals.sort()
    pct = lambda p: finals[max(0, min(n_runs - 1, int(n_runs * p)))]
    return {
        "ruin": ruined / n_runs, "p01": pct(0.01), "p10": pct(0.10),
        "p25": pct(0.25), "p50": pct(0.50), "p75": pct(0.75),
        "p90": pct(0.90), "p99": pct(0.99),
    }


def run_one(label, max_pos, max_gross, kelly_mult, liq_cap, edge_thr,
            markets, qtypes, resolutions, base_rates, volumes):
    s = make_strat(edge_thr)
    s = StrategyParams(
        shrink_lambda=s.shrink_lambda, trend_alpha=s.trend_alpha,
        time_decay=s.time_decay, edge_threshold=s.edge_threshold,
        kelly_fraction=s.kelly_fraction * kelly_mult,
        notional=s.notional, symmetric=s.symmetric,
    )
    res = proper_backtest(
        s, markets, qtypes, resolutions, base_rates,
        initial_bankroll=10_000,
        max_position_pct=max_pos, max_gross_exposure_pct=max_gross,
        cost_model=CostModel(),
        liquidity_cap_pct=liq_cap, market_volumes={},  # ignore vol cap to find true ceiling
    )
    ft, lt = first_last(res.trades)
    days = (lt - ft).days if ft and lt else 0
    cag = cagr(res.initial_bankroll, res.final_bankroll, days)
    mo = monthly(res.initial_bankroll, res.final_bankroll, days)
    mult = res.final_bankroll / res.initial_bankroll
    bs = bootstrap(res)

    log.info(f"\n=== {label} ===")
    log.info(f"  ${res.initial_bankroll:>10,.0f} → ${res.final_bankroll:>13,.0f}  "
             f"({mult:>9,.2f}× over {days}d active = {days/30.44:.1f}mo)")
    log.info(f"  CAGR: {100*cag:+6.1f}%   monthly: {100*mo:+6.2f}%/mo")
    log.info(f"  trades: {res.n_trades}  hit: {100*res.hit_rate:.1f}%  PF: {res.profit_factor:.2f}  "
             f"avg_pos: {100*res.capital_deployed_pct:.1f}%")
    log.info(f"  Sharpe: {res.sharpe_annual:.2f}  Max DD: {100*res.max_drawdown_pct:.1f}%")
    log.info(f"  bootstrap (5000 runs, n={res.n_trades} trades each):")
    log.info(f"    p01={bs.get('p01', 0):.2f}× p10={bs.get('p10', 0):.2f}× "
             f"p25={bs.get('p25', 0):.2f}× p50={bs.get('p50', 0):.2f}× "
             f"p75={bs.get('p75', 0):.2f}× p90={bs.get('p90', 0):.2f}× "
             f"p99={bs.get('p99', 0):.2f}×")
    log.info(f"    ruin (<5% of initial): {100*bs.get('ruin', 0):.2f}%")
    return {
        "label": label, "final": res.final_bankroll, "mult": mult,
        "cagr": cag, "monthly": mo, "sharpe": res.sharpe_annual,
        "max_dd": res.max_drawdown_pct, "n_trades": res.n_trades,
        "hit_rate": res.hit_rate, "profit_factor": res.profit_factor,
        "days": days, "bootstrap": bs,
    }


def main() -> int:
    markets, qtypes, resolutions = load_v3()
    base_rates = {"event": 0.204, "field": 0.085, "deadline": 0.324, "durative": 0.0}
    log.info(f"loaded {len(markets)} markets")

    log.info("\n" + "=" * 80)
    log.info("FIXED edge>=0.10 — sweep position sizing & gross exposure")
    log.info("=" * 80)

    summaries = []
    grid = [
        ("max_pos=0.05  gross=0.50  K=1.0", 0.05, 0.50, 1.0, 0.005, 0.10),
        ("max_pos=0.10  gross=0.80  K=1.0", 0.10, 0.80, 1.0, 0.005, 0.10),
        ("max_pos=0.15  gross=1.00  K=1.0", 0.15, 1.00, 1.0, 0.010, 0.10),
        ("max_pos=0.20  gross=1.20  K=1.0", 0.20, 1.20, 1.0, 0.010, 0.10),
        ("max_pos=0.25  gross=1.50  K=1.0", 0.25, 1.50, 1.0, 0.020, 0.10),
        ("max_pos=0.30  gross=2.00  K=1.0", 0.30, 2.00, 1.0, 0.030, 0.10),
        ("max_pos=0.40  gross=2.50  K=1.0", 0.40, 2.50, 1.0, 0.030, 0.10),
        ("max_pos=0.50  gross=3.00  K=1.0", 0.50, 3.00, 1.0, 0.030, 0.10),
        # Sweep Kelly multiplier at fixed 25% pos
        ("max_pos=0.25  gross=1.50  K=0.5", 0.25, 1.50, 0.5, 0.020, 0.10),
        ("max_pos=0.25  gross=1.50  K=0.7", 0.25, 1.50, 0.7, 0.020, 0.10),
        ("max_pos=0.25  gross=1.50  K=1.2", 0.25, 1.50, 1.2, 0.020, 0.10),
    ]
    for args in grid:
        s = run_one(*args, markets, qtypes, resolutions, base_rates, {})
        summaries.append(s)

    # Side by side
    log.info("\n" + "=" * 80)
    log.info("SIDE BY SIDE")
    log.info("=" * 80)
    log.info(f"  {'tier':<35} {'final':>13} {'mult':>9} {'CAGR':>9} {'mo':>9} "
             f"{'DD':>6} {'PF':>5} {'p10':>6} {'p50':>7} {'p90':>8} {'ruin':>6}")
    log.info("  " + "-" * 130)
    for s in summaries:
        bs = s["bootstrap"]
        log.info(
            f"  {s['label']:<35} ${s['final']:>11,.0f} "
            f"{s['mult']:>7,.1f}× {100*s['cagr']:>+7.0f}% {100*s['monthly']:>+6.1f}%/m "
            f"{100*s['max_dd']:>4.0f}% {s['profit_factor']:>4.2f} "
            f"{bs.get('p10',0):>4.2f}× {bs.get('p50',0):>5.2f}× {bs.get('p90',0):>6.2f}× "
            f"{100*bs.get('ruin',0):>4.1f}%"
        )

    best_cagr = max(summaries, key=lambda s: s["cagr"])
    best_p10 = max(summaries, key=lambda s: s["bootstrap"].get("p10", 0))
    log.info(f"\nBEST CAGR:  {best_cagr['label']} → {100*best_cagr['cagr']:+.1f}% CAGR / "
             f"{100*best_cagr['monthly']:+.2f}%/mo")
    log.info(f"BEST p10:   {best_p10['label']} → bootstrap p10 = {best_p10['bootstrap']['p10']:.2f}×")

    out = Path("runs/live_iter/extreme_backtest_v2.json")
    out.write_text(json.dumps({"tiers": summaries}, indent=2, default=str))
    log.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
