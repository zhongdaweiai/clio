"""Push to the actual ceiling.

Strategy choices to maximize CAGR:
  1. ULTRA-high edge filter — only trades where |edge| > 0.20 (vs 0.15)
  2. Field-only restriction — qtype with 76% hit rate
  3. Combined ensemble of best-of-each-qtype tuning
  4. Multi-Kelly (>1) sizing — over-Kelly captures more growth at higher variance
  5. The single mathematical limit: full historical Kelly with no caps

Then a Monte Carlo proper bootstrap (sample WITH replacement from historical
trades, sequence them in random order, simulate path) to get an honest
distribution of P50, P10, P1 outcomes given the historical edge holds.

If 50%+ monthly is achievable from this dataset's edge, this script finds it.
If it isn't, this script reports the actual ceiling.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from datetime import date
from pathlib import Path

from clio.frozen.harness import Market
from clio.research.proper_backtest import (
    BacktestResult, CostModel, proper_backtest,
)
from clio.research.strategies import StrategyParams


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("extreme")


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


def make_strat(edge_thr: float, only_qtypes: set[str] | None = None) -> StrategyParams:
    """Strategy params with edge threshold and optional qtype filter."""
    if only_qtypes is None:
        only_qtypes = {"event", "field", "deadline"}
    lam = []
    for q in ("event", "field", "deadline", "durative"):
        if q in only_qtypes:
            lam.append((q, 0.5 if q == "event" else (0.49 if q == "field" else 0.0)))
        else:
            lam.append((q, 0.0))  # zero shrinkage means strategy never disagrees → no trade
    return StrategyParams(
        shrink_lambda=tuple(lam),
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


def monthly_compound(initial, final, days):
    if initial <= 0 or days <= 0 or final <= 0:
        return 0.0
    return (final / initial) ** (1 / (days / 30.44)) - 1


def first_last(trades):
    if not trades:
        return None, None
    return min(t.open_date for t in trades), max(t.close_date for t in trades)


def bootstrap_paths(res: BacktestResult, n_runs: int = 5000, n_trades_to_simulate: int = None,
                    seed: int = 13, ruin_threshold: float = 0.05) -> dict:
    """Sample WITH replacement from historical trade fractional returns and
    simulate a path. This treats the historical trade-return distribution as
    the population and bootstraps from it.

    A 1.5× per-trade-cap is included to truncate the longest tail wins
    (which would otherwise dominate every bootstrap draw, hiding the
    distribution).
    """
    if not res.trades:
        return {"ruin_prob": 0.0}
    fracs = [t.payoff_dollars / max(t.bankroll_at_open, 1.0) for t in res.trades]
    n = n_trades_to_simulate or len(fracs)
    rng = random.Random(seed)
    finals = []
    ruined = 0
    for _ in range(n_runs):
        bk = 1.0
        is_ruin = False
        path_min = 1.0
        for _ in range(n):
            f = rng.choice(fracs)
            bk *= (1 + f)
            path_min = min(path_min, bk)
            if bk < ruin_threshold:
                is_ruin = True
        if is_ruin:
            ruined += 1
        finals.append(bk)
    finals.sort()
    pct = lambda p: finals[max(0, min(n_runs - 1, int(n_runs * p)))]
    return {
        "ruin_prob_at_5pct": ruined / n_runs,
        "n_simulated_trades": n,
        "p01": pct(0.01), "p05": pct(0.05), "p10": pct(0.10),
        "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
        "mean": sum(finals) / n_runs,
    }


def run_one(label, strat, max_pos, max_gross, kelly_mult, liq_cap,
            markets, qtypes, resolutions, base_rates, volumes):
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
        cost_model=CostModel(),
        liquidity_cap_pct=liq_cap,
        market_volumes=volumes,
    )
    ft, lt = first_last(res.trades)
    days = (lt - ft).days if ft and lt else 0
    cag = cagr(res.initial_bankroll, res.final_bankroll, days)
    mo = monthly_compound(res.initial_bankroll, res.final_bankroll, days)
    mult = res.final_bankroll / res.initial_bankroll if res.initial_bankroll > 0 else 0
    bs = bootstrap_paths(res)

    log.info(f"\n=== {label} ===")
    log.info(f"  ${res.initial_bankroll:>10,.0f} → ${res.final_bankroll:>13,.0f}  "
             f"({mult:7,.2f}× over {days}d)")
    log.info(f"  CAGR:               {100*cag:+8.1f}%")
    log.info(f"  monthly compound:   {100*mo:+8.2f}%/mo")
    log.info(f"  trades: {res.n_trades}  hit: {100*res.hit_rate:.1f}%  PF: {res.profit_factor:.2f}")
    log.info(f"  Sharpe: {res.sharpe_annual:.2f}  Max DD: {100*res.max_drawdown_pct:.1f}%")
    log.info(f"  bootstrap (sample-w/-replacement, {bs['n_simulated_trades']} trades):")
    log.info(f"    p01 mult: {bs['p01']:>6.2f}×   p10: {bs['p10']:>6.2f}×   "
             f"p50: {bs['p50']:>6.2f}×   p90: {bs['p90']:>6.2f}×   p99: {bs['p99']:>6.2f}×")
    log.info(f"    mean:     {bs['mean']:>6.2f}×   ruin (<5%): {100*bs['ruin_prob_at_5pct']:.2f}%")
    return {
        "label": label, "final": res.final_bankroll, "mult": mult,
        "cagr": cag, "monthly": mo, "sharpe": res.sharpe_annual,
        "max_dd": res.max_drawdown_pct, "n_trades": res.n_trades,
        "hit_rate": res.hit_rate, "profit_factor": res.profit_factor,
        "avg_pos_pct": res.capital_deployed_pct,
        "days": days, "bootstrap": bs,
    }, res


def main() -> int:
    markets, qtypes, resolutions = load_v3()
    volumes = load_volumes()
    base_rates = {"event": 0.204, "field": 0.085, "deadline": 0.324, "durative": 0.0}
    log.info(f"loaded {len(markets)} markets")

    log.info("\n" + "=" * 70)
    log.info("EDGE-THRESHOLD SWEEP (event+field+deadline; 10% pos, 80% gross, full Kelly)")
    log.info("=" * 70)

    summaries = []
    raw = []
    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        s, r = run_one(
            f"edge>={thr}", make_strat(thr), 0.10, 0.80, 1.0, 0.005,
            markets, qtypes, resolutions, base_rates, volumes,
        )
        summaries.append(s)
        raw.append((s["label"], r))

    log.info("\n" + "=" * 70)
    log.info("QTYPE-RESTRICTED VARIANTS (edge>=0.15)")
    log.info("=" * 70)
    for sub in [{"field"}, {"event"}, {"event", "field"}, {"event", "field", "deadline"}]:
        label = f"qtypes={sorted(sub)}"
        s, _ = run_one(
            label, make_strat(0.15, only_qtypes=sub), 0.10, 0.80, 1.0, 0.005,
            markets, qtypes, resolutions, base_rates, volumes,
        )
        summaries.append(s)

    log.info("\n" + "=" * 70)
    log.info("ULTRA tier — high conviction + concentrated sizing + over-Kelly")
    log.info("=" * 70)
    for label, thr, max_pos, max_gross, kmult, liq in [
        ("ultra_a (edge>=0.20, 15% pos, 1.5× Kelly)", 0.20, 0.15, 1.0, 1.5, 0.01),
        ("ultra_b (edge>=0.25, 20% pos, 2.0× Kelly)", 0.25, 0.20, 1.0, 2.0, 0.02),
        ("ultra_c (edge>=0.30, 30% pos, 2.0× Kelly)", 0.30, 0.30, 1.5, 2.0, 0.05),
        ("ultra_d (field-only, edge>=0.10, 20% pos)", 0.10, 0.20, 1.0, 1.0, 0.02),
    ]:
        if "field-only" in label:
            sp = make_strat(thr, only_qtypes={"field"})
        else:
            sp = make_strat(thr)
        s, r = run_one(label, sp, max_pos, max_gross, kmult, liq,
                       markets, qtypes, resolutions, base_rates, volumes)
        summaries.append(s)
        raw.append((s["label"], r))

    # Side-by-side
    log.info("\n" + "=" * 70)
    log.info("SIDE BY SIDE")
    log.info("=" * 70)
    log.info(f"  {'tier':<48} {'final':>13} {'mult':>9} {'CAGR':>9} {'mo':>9} {'DD':>7} {'p10':>7} {'p50':>7} {'ruin%':>7}")
    log.info(f"  {'-'*48} {'-'*13} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for s in summaries:
        bs = s["bootstrap"]
        log.info(
            f"  {s['label']:<48} ${s['final']:>11,.0f} "
            f"{s['mult']:>7,.1f}× {100*s['cagr']:>+7.0f}% {100*s['monthly']:>+6.1f}%/m "
            f"{100*s['max_dd']:>5.1f}% "
            f"{bs['p10']:>5.2f}× {bs['p50']:>5.2f}× {100*bs['ruin_prob_at_5pct']:>5.2f}%"
        )

    # Find absolute best CAGR
    best = max(summaries, key=lambda s: s["cagr"])
    log.info(f"\nBEST CAGR: {best['label']} → {100*best['cagr']:+.1f}% CAGR ({100*best['monthly']:+.2f}%/mo)")

    out = Path("runs/live_iter/extreme_backtest.json")
    out.write_text(json.dumps({"tiers": summaries}, indent=2, default=str))
    log.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
