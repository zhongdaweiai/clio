"""Run the proper compounded backtest on the v3 dataset.

Outputs:
- The strategy's actual realized equity curve, Sharpe, Max DD, profit factor
- Comparison vs alternatives: market_echo, always_no, base_rate
- Stress tests at higher cost levels
- Top 10 winners and losers in dollars
- ASCII equity curve so you can SEE the path
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

from clio.frozen.harness import Market
from clio.research.proper_backtest import BacktestResult, CostModel, proper_backtest
from clio.research.strategies import StrategyParams
from clio.research.tuner import grid_search
from clio.research.walk_forward import compute_base_rates


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backtest")


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


def fetch_volumes(market_ids: list[str], cache_path: Path) -> dict[str, float]:
    """Pull volume for each market (one-time, cached)."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    out: dict[str, float] = {}
    for i, mid in enumerate(market_ids):
        url = f"https://gamma-api.polymarket.com/markets/{mid}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
            vol = float(data.get("volumeNum") or 0)
            out[mid] = vol
        except Exception:
            out[mid] = 0
        time.sleep(0.05)
        if (i + 1) % 50 == 0:
            log.info("fetched %d/%d volumes", i + 1, len(market_ids))
    cache_path.write_text(json.dumps(out))
    return out


# ---- alternative strategies for comparison ----


def _market_echo() -> StrategyParams:
    return StrategyParams(
        shrink_lambda=(("event", 0.0), ("field", 0.0), ("deadline", 0.0), ("durative", 0.0)),
        edge_threshold=10.0,  # never trades
    )


def _always_no_constant() -> StrategyParams:
    """Predict 0.0 for every market; will trade NO whenever market price > threshold."""
    # Use shrink_lambda=1.0 with base_rates patched to 0 in the runner.
    return StrategyParams(
        shrink_lambda=(("event", 1.0), ("field", 1.0), ("deadline", 1.0), ("durative", 1.0)),
        edge_threshold=0.05,
    )


def _base_rate_only() -> StrategyParams:
    """Pure shrink to base rate (no market info)."""
    return StrategyParams(
        shrink_lambda=(("event", 1.0), ("field", 1.0), ("deadline", 1.0), ("durative", 1.0)),
        edge_threshold=0.05,
    )


def _strategy_v3_best() -> StrategyParams:
    """The strategy from iterate_v5_master grid top-1."""
    return StrategyParams(
        shrink_lambda=(("event", 0.3), ("field", 0.4), ("deadline", 0.0), ("durative", 0.0)),
        trend_alpha=0.25,
        time_decay=0.0,
        edge_threshold=0.05,
        kelly_fraction=0.5,
        symmetric=False,
    )


def _strategy_v3_evolved() -> StrategyParams:
    """The strategy genetic evolution found (mean-reversion)."""
    return StrategyParams(
        shrink_lambda=(("event", 0.5), ("field", 0.49), ("deadline", 0.0), ("durative", 0.09)),
        trend_alpha=-0.30,
        time_decay=0.0,
        edge_threshold=0.09,
        kelly_fraction=0.80,
        symmetric=False,
    )


# ---- ASCII equity curve ----


def ascii_equity_curve(eq: list[tuple[date, float]], width: int = 90, height: int = 14) -> str:
    if not eq:
        return "(empty)"
    # Reduce to evenly-spaced sample points across calendar time.
    eq_sorted = sorted(eq, key=lambda x: x[0])
    if eq_sorted[-1][0] == eq_sorted[0][0]:
        return "(equity curve degenerate, all on same day)"
    start, end = eq_sorted[0][0], eq_sorted[-1][0]
    span_days = (end - start).days

    samples: list[tuple[date, float]] = []
    last_eq = eq_sorted[0][1]
    cur_idx = 0
    for i in range(width):
        target_day = start.toordinal() + int(span_days * i / (width - 1))
        while cur_idx < len(eq_sorted) and eq_sorted[cur_idx][0].toordinal() <= target_day:
            last_eq = eq_sorted[cur_idx][1]
            cur_idx += 1
        samples.append((date.fromordinal(target_day), last_eq))

    values = [v for _, v in samples]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        hi = lo + 1
    initial = eq_sorted[0][1]

    grid = [[" "] * width for _ in range(height)]
    for x, (_, v) in enumerate(samples):
        y = height - 1 - int((v - lo) / (hi - lo) * (height - 1))
        y = max(0, min(height - 1, y))
        grid[y][x] = "*"
    # Mark initial bankroll line if visible.
    if lo <= initial <= hi:
        y0 = height - 1 - int((initial - lo) / (hi - lo) * (height - 1))
        for x in range(width):
            if grid[y0][x] == " ":
                grid[y0][x] = "·"

    lines = []
    for y, row in enumerate(grid):
        v = lo + (hi - lo) * (height - 1 - y) / (height - 1)
        lines.append(f"  ${v:>10,.0f}  |{''.join(row)}|")
    lines.append("              " + "+" + "-" * width + "+")
    lines.append(f"              {start}{'  '*(width // 8 - 4)}                                  {end}")
    return "\n".join(lines)


# ---- pretty print result ----


def print_result(name: str, res: BacktestResult, base: BacktestResult | None = None) -> None:
    log.info(f"\n=== {name} ===")
    log.info(f"  starting bankroll:  ${res.initial_bankroll:>12,.0f}")
    log.info(f"  ending bankroll:    ${res.final_bankroll:>12,.0f}  (return {100 * res.total_return:+.2f}%)")
    if base is not None:
        outperformance = res.total_return - base.total_return
        log.info(f"  vs market echo:     {100 * outperformance:+.2f} pp outperformance")
    log.info(f"  trades:             {res.n_trades}  (hit rate {100 * res.hit_rate:.1f}%)")
    log.info(f"  avg win:            ${res.avg_win_dollars:+,.2f}")
    log.info(f"  avg loss:           ${res.avg_loss_dollars:+,.2f}")
    log.info(f"  profit factor:      {res.profit_factor:.2f}")
    log.info(f"  Sharpe (annual):    {res.sharpe_annual:.2f}")
    log.info(f"  max drawdown:       {100 * res.max_drawdown_pct:.1f}%")
    log.info(f"  Calmar:             {res.calmar:.2f}")
    log.info(f"  avg days held:      {res.avg_days_held:.0f}")
    log.info(f"  avg position size:  {100 * res.capital_deployed_pct:.2f}% of bankroll")


def main() -> int:
    markets, qtypes, resolutions = load_v3()
    log.info("loaded %d markets", len(markets))

    # Train base rates on full corpus (for fair comparison across strategies).
    base_rates = compute_base_rates(markets, qtypes, resolutions)
    log.info("base rates: %s", {k: round(v, 3) for k, v in base_rates.items()})

    # Volume cache.
    vol_cache = Path("runs/live_iter/market_volumes.json")
    log.info("loading volumes (cached: %s)...", vol_cache.exists())
    volumes = fetch_volumes([m.market_id for m in markets], vol_cache)

    cost_low = CostModel(spread_bps=100, slippage_bps=20)
    cost_default = CostModel()
    cost_high = CostModel(spread_bps=300, slippage_bps=80)

    log.info("\n" + "=" * 70)
    log.info("MAIN BACKTEST: strategy v3_best, default cost model")
    log.info("=" * 70)
    main_res = proper_backtest(
        _strategy_v3_best(), markets, qtypes, resolutions, base_rates,
        cost_model=cost_default, market_volumes=volumes,
    )
    print_result("v3_best (default costs 1.5% spread + 30bps slippage)", main_res)

    market_res = proper_backtest(
        _market_echo(), markets, qtypes, resolutions, base_rates,
        cost_model=cost_default, market_volumes=volumes,
    )
    print_result("market_echo (no trades)", market_res)
    print_result("v3_best vs market_echo", main_res, market_res)

    log.info("\n" + "=" * 70)
    log.info("ALTERNATIVES")
    log.info("=" * 70)
    alts = [
        ("v3_evolved (mean-reversion)", _strategy_v3_evolved()),
    ]
    alt_results = []
    for name, sp in alts:
        res = proper_backtest(sp, markets, qtypes, resolutions, base_rates,
                              cost_model=cost_default, market_volumes=volumes)
        print_result(name, res, market_res)
        alt_results.append((name, res))

    log.info("\n" + "=" * 70)
    log.info("STRESS TEST: cost sensitivity")
    log.info("=" * 70)
    for label, cm in [
        ("low_cost (1.0% + 20bps)", cost_low),
        ("mid_cost (1.5% + 30bps)", cost_default),
        ("high_cost (3.0% + 80bps)", cost_high),
    ]:
        res = proper_backtest(_strategy_v3_best(), markets, qtypes, resolutions, base_rates,
                              cost_model=cm, market_volumes=volumes)
        log.info(f"  {label:<25} ret={100 * res.total_return:+.2f}%  Sharpe={res.sharpe_annual:.2f}  "
                 f"MaxDD={100 * res.max_drawdown_pct:.1f}%  trades={res.n_trades}")

    log.info("\n" + "=" * 70)
    log.info("STRESS TEST: tail-risk — what if we miss the 5 best trades?")
    log.info("=" * 70)
    sorted_trades = sorted(main_res.trades, key=lambda t: -t.payoff_dollars)
    for k in [0, 5, 10, 20]:
        excluded = set(t.market_id for t in sorted_trades[:k])
        kept = [t for t in main_res.trades if t.market_id not in excluded]
        bk = main_res.initial_bankroll
        for t in kept:
            bk += t.payoff_dollars
        ret_pct = (bk - main_res.initial_bankroll) / main_res.initial_bankroll
        log.info(f"  exclude top-{k:2d} wins → return {100 * ret_pct:+.2f}%  (n={len(kept)} trades)")

    log.info("\n" + "=" * 70)
    log.info("TOP 10 WINNERS BY DOLLARS")
    log.info("=" * 70)
    for t in sorted(main_res.trades, key=lambda t: -t.payoff_dollars)[:10]:
        log.info(
            f"  {t.open_date} [{t.qtype:<8}] {t.side:<3} mkt={t.open_price:.3f}→fill={t.fill_price:.3f}  "
            f"outcome={t.outcome}  pnl=${t.payoff_dollars:+,.2f}  ({100*t.return_pct:.0f}% on "
            f"${t.notional_dollars:,.0f}, {t.days_held}d)  {t.question[:60]}"
        )

    log.info("\n" + "=" * 70)
    log.info("TOP 10 LOSERS BY DOLLARS")
    log.info("=" * 70)
    for t in sorted(main_res.trades, key=lambda t: t.payoff_dollars)[:10]:
        log.info(
            f"  {t.open_date} [{t.qtype:<8}] {t.side:<3} mkt={t.open_price:.3f}→fill={t.fill_price:.3f}  "
            f"outcome={t.outcome}  pnl=${t.payoff_dollars:+,.2f}  ({100*t.return_pct:.0f}% on "
            f"${t.notional_dollars:,.0f}, {t.days_held}d)  {t.question[:60]}"
        )

    # Per-qtype breakdown.
    log.info("\n=== PER-QTYPE BREAKDOWN (v3_best) ===")
    by_qtype: dict[str, list] = defaultdict(list)
    for t in main_res.trades:
        by_qtype[t.qtype].append(t)
    for qt, ts in sorted(by_qtype.items()):
        wins = sum(1 for t in ts if t.payoff_dollars > 0)
        pnl = sum(t.payoff_dollars for t in ts)
        notional = sum(t.notional_dollars for t in ts)
        avg_ret = (pnl / notional) if notional > 0 else 0
        log.info(
            f"  {qt:<10} n={len(ts):<3} wins={wins:<3} hit={100*wins/len(ts):.1f}%  "
            f"pnl=${pnl:+,.2f}  avg_per_trade_ret={100*avg_ret:.2f}%"
        )

    log.info("\n" + "=" * 70)
    log.info("EQUITY CURVE")
    log.info("=" * 70)
    log.info("\n%s\n", ascii_equity_curve(main_res.equity_curve))

    out = Path("runs/live_iter/proper_backtest.json")
    out.write_text(json.dumps({
        "main": {
            "initial": main_res.initial_bankroll, "final": main_res.final_bankroll,
            "total_return": main_res.total_return,
            "n_trades": main_res.n_trades, "hit_rate": main_res.hit_rate,
            "profit_factor": main_res.profit_factor,
            "sharpe_annual": main_res.sharpe_annual,
            "max_drawdown_pct": main_res.max_drawdown_pct,
            "calmar": main_res.calmar,
            "avg_days_held": main_res.avg_days_held,
            "trades": [
                {
                    "open_date": str(t.open_date), "close_date": str(t.close_date),
                    "qtype": t.qtype, "side": t.side,
                    "open_price": t.open_price, "fill_price": t.fill_price,
                    "pred": t.pred, "edge": t.edge,
                    "size_pct": t.size_pct, "notional": t.notional_dollars,
                    "outcome": t.outcome, "payoff": t.payoff_dollars,
                    "return_pct": t.return_pct, "days_held": t.days_held,
                    "question": t.question, "market_id": t.market_id,
                }
                for t in main_res.trades
            ],
            "equity_curve": [(str(d), v) for d, v in main_res.equity_curve],
        },
        "alternatives": {
            name: {
                "total_return": res.total_return, "sharpe": res.sharpe_annual,
                "max_dd": res.max_drawdown_pct, "n_trades": res.n_trades,
            }
            for name, res in alt_results + [("market_echo", market_res)]
        },
    }, indent=2, default=str))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
