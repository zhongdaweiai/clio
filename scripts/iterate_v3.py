"""Iteration v3 — verify robustness of v2's edge and refine.

Findings from v2:
- v2_shrink_10 (light shrinkage toward type base rate) produced +47.82% PnL
  on 19 trades vs market_echo's 0, on a single seed=7 holdout.
- Brier was slightly *worse* than market echo (0.1094 vs 0.1082) — the
  edge is in *trade selection*, not Brier. The strategy bets only when its
  prediction differs from market by > 0.05 absolute, and those bets win
  more often than they lose.
- Per-qtype: shrink helped on event+field (where YES is always 0%);
  hurt on deadline (where YES happens 14% of the time).

This script:
1. Verifies v2 across 30 random splits (seeds 0..29) — is the edge
   stable, or seed-7 luck?
2. Tests a refined strategy: shrink ONLY on event+field, market_echo on
   deadline. This should keep the Brier-better-on-the-cases-where-it-works
   property without taking the deadline hit.
3. Reports the edge with confidence intervals.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from clio.frozen.harness import Market


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("iter3")


# -------------------- data --------------------


def load_v2():
    with open("runs/live_iter/markets_v2.json") as f:
        payload = json.load(f)
    markets: list[Market] = []
    qtypes: dict[str, str] = {}
    for d in payload["markets"]:
        m = Market(
            market_id=d["market_id"],
            question=d["question"],
            regime=d["regime"],
            observed_at=date.fromisoformat(d["observed_at"]),
            closes_at=date.fromisoformat(d["closes_at"]),
            timeline=tuple(date.fromisoformat(s) for s in d["timeline"]),
            market_prices={date.fromisoformat(k): float(v) for k, v in d["market_prices"].items()},
        )
        markets.append(m)
        qtypes[m.market_id] = d.get("qtype", "event")
    resolutions = {k: int(v) for k, v in payload["resolutions"].items()}
    return markets, qtypes, resolutions


def stratified_split_by_type(markets, qtypes, seed: int = 7, train_frac: float = 0.5):
    rng = random.Random(seed)
    by: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        by[qtypes[m.market_id]].append(m)
    train, test = [], []
    for t, ms in by.items():
        ms = list(ms)
        rng.shuffle(ms)
        cut = max(1, int(len(ms) * train_frac))
        train.extend(ms[:cut])
        test.extend(ms[cut:])
    return train, test


def compute_type_base_rates(train_markets, qtypes, resolutions) -> dict[str, float]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for m in train_markets:
        if m.market_id in resolutions:
            by_type[qtypes[m.market_id]].append(resolutions[m.market_id])
    return {t: sum(o) / len(o) for t, o in by_type.items() if o}


# -------------------- strategies --------------------


PredictFn = Callable[[Market, int, str, dict[str, float]], float]


def predict_market_echo(market, step, qtype, base_rates):
    return market.market_prices[market.timeline[step]]


def make_shrink(lam: float) -> PredictFn:
    def fn(market, step, qtype, base_rates):
        m = market.market_prices[market.timeline[step]]
        b = base_rates.get(qtype, 0.25)
        return (1 - lam) * m + lam * b
    return fn


def make_shrink_typed(lam_by_type: dict[str, float]) -> PredictFn:
    """Shrink only on the types where lam > 0."""
    def fn(market, step, qtype, base_rates):
        m = market.market_prices[market.timeline[step]]
        lam = lam_by_type.get(qtype, 0.0)
        if lam == 0:
            return m
        b = base_rates.get(qtype, 0.25)
        return (1 - lam) * m + lam * b
    return fn


# -------------------- evaluation --------------------


@dataclass
class StrategyEval:
    name: str
    brier_overall: float
    pnl: float
    n_trades: int
    hits: int
    by_type_brier: dict[str, float] = field(default_factory=dict)
    by_type_pnl: dict[str, float] = field(default_factory=dict)
    by_type_trades: dict[str, int] = field(default_factory=dict)
    by_type_hits: dict[str, int] = field(default_factory=dict)


def evaluate(
    name: str,
    pred_fn: PredictFn,
    markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    base_rates: dict[str, float],
    edge_threshold: float = 0.05,
    notional: float = 0.05,
) -> StrategyEval:
    sq_total: list[float] = []
    pnls: list[float] = []
    n_trades = 0
    hits = 0
    by_type_brier: dict[str, list[float]] = defaultdict(list)
    by_type_pnl: dict[str, list[float]] = defaultdict(list)
    by_type_trades: dict[str, int] = defaultdict(int)
    by_type_hits: dict[str, int] = defaultdict(int)

    for m in markets:
        outcome = resolutions[m.market_id]
        qtype = qtypes[m.market_id]
        for step in range(len(m.timeline)):
            pred = pred_fn(m, step, qtype, base_rates)
            sq = (pred - outcome) ** 2
            sq_total.append(sq)
            by_type_brier[qtype].append(sq)

            mkt = m.market_prices[m.timeline[step]]
            edge = pred - mkt
            if 0.02 < mkt < 0.98 and abs(edge) > edge_threshold:
                if edge > 0:  # buy YES
                    payoff = ((1 - mkt) / mkt) * notional if outcome == 1 else -notional
                else:  # buy NO
                    payoff = (mkt / (1 - mkt)) * notional if outcome == 0 else -notional
                pnls.append(payoff)
                by_type_pnl[qtype].append(payoff)
                n_trades += 1
                by_type_trades[qtype] += 1
                if payoff > 0:
                    hits += 1
                    by_type_hits[qtype] += 1

    return StrategyEval(
        name=name,
        brier_overall=sum(sq_total) / len(sq_total) if sq_total else 0,
        pnl=sum(pnls),
        n_trades=n_trades,
        hits=hits,
        by_type_brier={t: sum(s) / len(s) for t, s in by_type_brier.items()},
        by_type_pnl={t: sum(s) for t, s in by_type_pnl.items()},
        by_type_trades=dict(by_type_trades),
        by_type_hits=dict(by_type_hits),
    )


# -------------------- main loop --------------------


def main() -> int:
    markets, qtypes, resolutions = load_v2()
    log.info("loaded %d markets", len(markets))

    SEEDS = list(range(30))
    strategies = {
        "v0_market_echo": predict_market_echo,
        "v2_shrink_10": make_shrink(0.10),
        "v2_shrink_15": make_shrink(0.15),
        "v6_shrink_typed": make_shrink_typed({
            "event": 0.20, "field": 0.30, "deadline": 0.0, "durative": 0.0,
        }),
        "v7_shrink_typed_v2": make_shrink_typed({
            "event": 0.15, "field": 0.20, "deadline": 0.0, "durative": 0.0,
        }),
        "v8_shrink_typed_strong": make_shrink_typed({
            "event": 0.30, "field": 0.40, "deadline": 0.0, "durative": 0.0,
        }),
    }

    # Run each strategy on each seed; collect per-seed numbers.
    log.info("\nRunning %d strategies x %d seeds = %d evals on holdout...",
             len(strategies), len(SEEDS), len(strategies) * len(SEEDS))
    results: dict[str, list[StrategyEval]] = {name: [] for name in strategies}
    for seed in SEEDS:
        train, holdout = stratified_split_by_type(markets, qtypes, seed=seed)
        base_rates = compute_type_base_rates(train, qtypes, resolutions)
        for name, fn in strategies.items():
            ev = evaluate(name, fn, holdout, qtypes, resolutions, base_rates)
            results[name].append(ev)

    log.info("\n%-25s %-13s %-13s %-13s %-13s %-13s",
             "strategy",
             "Brier  μ±σ",
             "PnL %  μ±σ",
             "n_trades μ",
             "hit_rate μ",
             "PnL>0 seeds",
    )
    log.info("-" * 95)
    summaries = {}
    for name, evals in results.items():
        briers = [e.brier_overall for e in evals]
        pnls = [e.pnl for e in evals]
        trades = [e.n_trades for e in evals]
        hit_rates = [e.hits / e.n_trades if e.n_trades > 0 else 0 for e in evals]
        n_pnl_pos = sum(1 for p in pnls if p > 0)
        s = {
            "brier_mu": statistics.mean(briers),
            "brier_sd": statistics.stdev(briers) if len(briers) > 1 else 0,
            "pnl_mu": statistics.mean(pnls),
            "pnl_sd": statistics.stdev(pnls) if len(pnls) > 1 else 0,
            "trades_mu": statistics.mean(trades),
            "hit_rate_mu": statistics.mean(hit_rates),
            "pnl_pos_seeds": n_pnl_pos,
        }
        summaries[name] = s
        log.info(
            "%-25s %.4f±%.4f %+0.3f±%.3f   %5.1f         %.3f         %2d/%d",
            name,
            s["brier_mu"], s["brier_sd"],
            s["pnl_mu"], s["pnl_sd"],
            s["trades_mu"],
            s["hit_rate_mu"],
            n_pnl_pos, len(SEEDS),
        )

    # Edge over baseline: Brier delta and PnL delta with bootstrap-style CI
    # (just min/max/mean across seeds).
    log.info("\n=== edge over v0_market_echo (mean across %d seeds) ===", len(SEEDS))
    log.info("%-25s %-15s %-15s %-15s %-15s",
             "strategy", "Δ Brier μ", "Δ Brier sign", "Δ PnL μ", "Δ PnL > 0 in")
    baseline = results["v0_market_echo"]
    for name, evals in results.items():
        if name == "v0_market_echo":
            continue
        brier_deltas = [e.brier_overall - b.brier_overall for e, b in zip(evals, baseline)]
        pnl_deltas = [e.pnl - b.pnl for e, b in zip(evals, baseline)]
        n_brier_better = sum(1 for d in brier_deltas if d < 0)
        n_pnl_better = sum(1 for d in pnl_deltas if d > 0)
        log.info(
            "%-25s %+.4f         %2d/%d         %+0.4f         %2d/%d",
            name,
            statistics.mean(brier_deltas),
            n_brier_better, len(SEEDS),
            statistics.mean(pnl_deltas),
            n_pnl_better, len(SEEDS),
        )

    # Per-qtype performance for the typed strategies
    log.info("\n=== per-qtype PnL (mean per seed) for typed strategies ===")
    for name in ["v0_market_echo", "v6_shrink_typed", "v7_shrink_typed_v2", "v8_shrink_typed_strong"]:
        evals = results[name]
        types = sorted({t for e in evals for t in e.by_type_pnl})
        agg_pnl: dict[str, float] = defaultdict(float)
        agg_trades: dict[str, int] = defaultdict(int)
        agg_hits: dict[str, int] = defaultdict(int)
        for ev in evals:
            for t in types:
                agg_pnl[t] += ev.by_type_pnl.get(t, 0.0)
                agg_trades[t] += ev.by_type_trades.get(t, 0)
                agg_hits[t] += ev.by_type_hits.get(t, 0)
        line = f"{name:<25}"
        for t in types:
            avg_pnl = agg_pnl[t] / len(evals)
            avg_trades = agg_trades[t] / len(evals)
            hit_rate = agg_hits[t] / agg_trades[t] if agg_trades[t] else 0
            line += f"  {t}: pnl={avg_pnl:+.3f}/seed, n_trades={avg_trades:.1f}, hit={hit_rate:.2f}"
        log.info(line)

    # Save full results
    out = Path("runs/live_iter/iterations_v3.json")
    out.write_text(json.dumps({
        "n_markets": len(markets),
        "n_seeds": len(SEEDS),
        "summaries": summaries,
        "per_seed": {
            name: [
                {
                    "seed": SEEDS[i],
                    "brier": ev.brier_overall,
                    "pnl": ev.pnl,
                    "n_trades": ev.n_trades,
                    "hits": ev.hits,
                    "by_type_brier": ev.by_type_brier,
                    "by_type_pnl": ev.by_type_pnl,
                    "by_type_trades": ev.by_type_trades,
                    "by_type_hits": ev.by_type_hits,
                }
                for i, ev in enumerate(evals)
            ]
            for name, evals in results.items()
        },
    }, indent=2))
    log.info("\nwrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
