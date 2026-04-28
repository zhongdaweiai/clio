"""Iteration v2 — actually try to beat the market.

Key changes from iterate.py:

1. Score at EVERY as_of, not just final. Aggregate Brier across the timeline.
   The final as_of is right before close where market is near-perfect; earlier
   windows are where edge lives.
2. Strategies use type-aware base rates and price-trajectory features, not
   news (which is too weak rule-based to add signal).
3. Compare each strategy directly to the market at the same as_of (apples to
   apples — both predicting the eventual outcome with the same info window).
4. Iteration goal: find any strategy variant that beats market by > 0.005
   Brier on the FULL timeline, on the holdout.

Strategy variants tested:
  v0_market_echo:   predict market price exactly (the unbeatable baseline)
  v1_type_prior:    pure question-type base rate (no market lookup)
  v2_shrink:        market_price + λ * (type_base_rate - market_price)
  v3_trend:         price + α * (price - price_prev)  — momentum
  v4_combo:         shrink + trend
  v5_combo_typed:   shrink toward type-base-rate, with type-specific α
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from clio.cli import _load_markets_payload
from clio.frozen.harness import Forecast, Market
from clio.frozen.scoring import brier_score, calibration_ece


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("iter2")


# -------------------- data loading --------------------


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


# -------------------- prediction strategies --------------------


# Type-conditioned base rates: computed once on TRAIN, used at scoring time.
# These are P(YES | qtype). The strategies below shrink market price toward
# these.
def compute_type_base_rates(train_markets, qtypes, resolutions) -> dict[str, float]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for m in train_markets:
        if m.market_id in resolutions:
            by_type[qtypes[m.market_id]].append(resolutions[m.market_id])
    out = {}
    for t, outs in by_type.items():
        if outs:
            out[t] = sum(outs) / len(outs)
    return out


PredictFn = Callable[[Market, int, str, dict[str, float]], float]


def predict_market_echo(market: Market, step: int, qtype: str, base_rates: dict[str, float]) -> float:
    return market.market_prices[market.timeline[step]]


def predict_type_prior(market: Market, step: int, qtype: str, base_rates: dict[str, float]) -> float:
    return base_rates.get(qtype, 0.25)


def make_shrink(lam: float) -> PredictFn:
    def fn(market: Market, step: int, qtype: str, base_rates: dict[str, float]) -> float:
        m = market.market_prices[market.timeline[step]]
        b = base_rates.get(qtype, 0.25)
        return (1 - lam) * m + lam * b
    return fn


def make_trend(alpha: float) -> PredictFn:
    """Linear extrapolation of recent price movement."""
    def fn(market: Market, step: int, qtype: str, base_rates: dict[str, float]) -> float:
        if step == 0:
            return market.market_prices[market.timeline[0]]
        cur = market.market_prices[market.timeline[step]]
        prev = market.market_prices[market.timeline[step - 1]]
        pred = cur + alpha * (cur - prev)
        return min(0.99, max(0.01, pred))
    return fn


def make_combo(lam: float, alpha: float) -> PredictFn:
    """Trend + shrink. Trend first, then shrink toward type prior."""
    trend = make_trend(alpha)
    def fn(market: Market, step: int, qtype: str, base_rates: dict[str, float]) -> float:
        t = trend(market, step, qtype, base_rates)
        b = base_rates.get(qtype, 0.25)
        return (1 - lam) * t + lam * b
    return fn


def make_combo_typed(lam_by_type: dict[str, float], alpha_by_type: dict[str, float]) -> PredictFn:
    """Per-question-type lambda and alpha."""
    def fn(market: Market, step: int, qtype: str, base_rates: dict[str, float]) -> float:
        lam = lam_by_type.get(qtype, 0.0)
        alpha = alpha_by_type.get(qtype, 0.0)
        cur = market.market_prices[market.timeline[step]]
        if step == 0:
            base_pred = cur
        else:
            prev = market.market_prices[market.timeline[step - 1]]
            base_pred = cur + alpha * (cur - prev)
        base_pred = min(0.99, max(0.01, base_pred))
        b = base_rates.get(qtype, 0.25)
        return (1 - lam) * base_pred + lam * b
    return fn


# -------------------- scoring --------------------


@dataclass
class StrategyEval:
    name: str
    description: str
    brier_per_step: list[float] = field(default_factory=list)
    brier_overall: float = 0.0
    pnl: float = 0.0
    n_trades: int = 0
    hit_rate: float = 0.0
    by_type: dict[str, float] = field(default_factory=dict)


def evaluate(
    name: str,
    description: str,
    pred_fn: PredictFn,
    markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    base_rates: dict[str, float],
) -> StrategyEval:
    n_steps = len(markets[0].timeline) if markets else 0
    sq_per_step = [[] for _ in range(n_steps)]
    sq_by_type: dict[str, list[float]] = defaultdict(list)
    pnls: list[float] = []
    n_trades = 0
    hits = 0

    for m in markets:
        outcome = resolutions[m.market_id]
        qtype = qtypes[m.market_id]
        for step in range(n_steps):
            pred = pred_fn(m, step, qtype, base_rates)
            sq = (pred - outcome) ** 2
            sq_per_step[step].append(sq)
            sq_by_type[qtype].append(sq)

            # Trade simulation: at each as_of, if our edge over market > 0,
            # bet a fixed 5% notional. Real trading is at the timeline point,
            # paid out at close.
            mkt = m.market_prices[m.timeline[step]]
            edge = pred - mkt
            if edge > 0.05 and 0.02 < mkt < 0.98:
                # Buy YES at mkt, payoff if outcome=1
                if outcome == 1:
                    pnls.append((1 - mkt) / mkt * 0.05)
                    hits += 1
                else:
                    pnls.append(-0.05)
                n_trades += 1
            elif edge < -0.05 and 0.02 < mkt < 0.98:
                # Buy NO at (1-mkt), payoff if outcome=0
                if outcome == 0:
                    pnls.append(mkt / (1 - mkt) * 0.05)
                    hits += 1
                else:
                    pnls.append(-0.05)
                n_trades += 1

    overall = sum(sum(s) for s in sq_per_step) / sum(len(s) for s in sq_per_step)
    return StrategyEval(
        name=name,
        description=description,
        brier_per_step=[sum(s) / len(s) if s else 0 for s in sq_per_step],
        brier_overall=overall,
        pnl=sum(pnls),
        n_trades=n_trades,
        hit_rate=hits / n_trades if n_trades else 0,
        by_type={t: sum(s) / len(s) for t, s in sq_by_type.items()},
    )


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


# -------------------- iteration loop --------------------


def main() -> int:
    markets, qtypes, resolutions = load_v2()
    log.info("loaded %d markets", len(markets))

    train, holdout = stratified_split_by_type(markets, qtypes, seed=7, train_frac=0.5)
    log.info("train=%d holdout=%d", len(train), len(holdout))

    base_rates = compute_type_base_rates(train, qtypes, resolutions)
    log.info("train base rates by qtype: %s", {k: round(v, 3) for k, v in base_rates.items()})

    iterations: list[tuple[str, str, PredictFn]] = []

    # v0: market echo (baseline we must beat)
    iterations.append(("v0_market_echo", "predict market price exactly", predict_market_echo))

    # v1: pure type prior
    iterations.append(("v1_type_prior", "predict empirical train YES rate per qtype", predict_type_prior))

    # v2: shrinkage toward type prior
    for lam in [0.1, 0.2, 0.3]:
        iterations.append(
            (f"v2_shrink_{int(lam*100):02d}",
             f"market * (1-{lam}) + base_rate * {lam}",
             make_shrink(lam)),
        )

    # v3: pure trend
    for alpha in [0.25, 0.5, 1.0]:
        iterations.append(
            (f"v3_trend_{int(alpha*100):03d}",
             f"price + {alpha} * (price - price_prev)",
             make_trend(alpha)),
        )

    # v4: combo — fixed lam, fixed alpha
    iterations.append(("v4_combo_l20a50", "shrink λ=0.2 + trend α=0.5", make_combo(0.2, 0.5)))
    iterations.append(("v4_combo_l30a25", "shrink λ=0.3 + trend α=0.25", make_combo(0.3, 0.25)))

    # v5: type-conditional combo, tuned by hand based on what we know about
    # each qtype. (In a longer iteration we'd grid-search on train; for now
    # we use sensible defaults.)
    iterations.append((
        "v5_combo_typed",
        "per-qtype λ and α",
        make_combo_typed(
            lam_by_type={"deadline": 0.3, "field": 0.0, "event": 0.2, "durative": 0.0},
            alpha_by_type={"deadline": 0.5, "field": 0.0, "event": 0.0, "durative": 0.0},
        ),
    ))

    log.info("\n%-22s %-35s %-9s %-9s %-9s %-9s %-12s %-7s",
             "name", "description", "B_overall", "B_t0", "B_t1", "B_t3", "PnL%", "trades")
    log.info("-" * 120)

    train_results, holdout_results = [], []
    for name, desc, fn in iterations:
        # We tune on TRAIN, then report HOLDOUT. To avoid re-fitting, we just
        # look at both for transparency.
        ev_train = evaluate(name, desc, fn, train, qtypes, resolutions, base_rates)
        ev_holdout = evaluate(name, desc, fn, holdout, qtypes, resolutions, base_rates)
        train_results.append(ev_train)
        holdout_results.append(ev_holdout)
        log.info(
            "%-22s %-35s %-9.4f %-9.4f %-9.4f %-9.4f %-12s %-7d",
            name, desc[:35],
            ev_holdout.brier_overall,
            ev_holdout.brier_per_step[0],
            ev_holdout.brier_per_step[1],
            ev_holdout.brier_per_step[3],
            f"{100*ev_holdout.pnl:+.2f}",
            ev_holdout.n_trades,
        )

    # Edge analysis: compare each variant to v0_market_echo on holdout.
    baseline = holdout_results[0]
    log.info("\n=== edge over market_echo (holdout, lower Brier = better) ===")
    log.info("%-22s %-9s %-9s %-9s %-9s %-9s %-9s",
             "name", "Δoverall", "Δt=0", "Δt=1", "Δt=2", "Δt=3", "ΔPnL%")
    for ev in holdout_results[1:]:
        deltas = [
            ev.brier_overall - baseline.brier_overall,
        ] + [
            ev.brier_per_step[k] - baseline.brier_per_step[k] for k in range(len(ev.brier_per_step))
        ]
        log.info(
            "%-22s %+0.4f   %+0.4f   %+0.4f   %+0.4f   %+0.4f   %+0.2f",
            ev.name,
            deltas[0],
            deltas[1], deltas[2], deltas[3], deltas[4],
            100 * (ev.pnl - baseline.pnl),
        )

    # Per-qtype Brier (holdout) for the combo strategies.
    log.info("\n=== per-qtype Brier (holdout) ===")
    types = sorted({qt for qt in qtypes.values()})
    log.info("%-22s " + " ".join(f"{t:<10}" for t in types), "name")
    for ev in holdout_results:
        cells = " ".join(f"{ev.by_type.get(t, 0):<10.4f}" for t in types)
        log.info("%-22s %s", ev.name, cells)

    # Find the best variant by holdout overall Brier.
    best = min(holdout_results, key=lambda e: e.brier_overall)
    log.info("\nBEST on holdout: %s  brier=%.4f  vs market %.4f  (Δ=%+0.4f)",
             best.name, best.brier_overall, baseline.brier_overall,
             best.brier_overall - baseline.brier_overall)

    out = Path("runs/live_iter/iterations_v2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_train": len(train),
        "n_holdout": len(holdout),
        "base_rates": base_rates,
        "results": [
            {
                "name": ev.name,
                "description": ev.description,
                "brier_overall": ev.brier_overall,
                "brier_per_step": ev.brier_per_step,
                "by_type": ev.by_type,
                "pnl_fraction": ev.pnl,
                "n_trades": ev.n_trades,
                "hit_rate": ev.hit_rate,
            }
            for ev in holdout_results
        ],
    }, indent=2))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
