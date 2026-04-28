"""Iteration v4 — Red Team the best variant.

Take the strongest finding from v3 (v6_shrink_typed) and ask:
  1. Does the edge hold up under temporal split (train on earlier markets,
     test on later)? Random splits can hide regime change.
  2. Where exactly does the edge come from? Per-qtype, per-price-band.
  3. What's the failure mode? When does the strategy LOSE?
  4. Bootstrap CI on the PnL.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from clio.frozen.harness import Market

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("iter4")


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


# v6_shrink_typed
LAM_BY_TYPE = {"event": 0.20, "field": 0.30, "deadline": 0.0, "durative": 0.0}


def predict_v6(market, step, qtype, base_rates):
    m = market.market_prices[market.timeline[step]]
    lam = LAM_BY_TYPE.get(qtype, 0.0)
    if lam == 0:
        return m
    b = base_rates.get(qtype, 0.0)
    return (1 - lam) * m + lam * b


def predict_market(market, step, qtype, base_rates):
    return market.market_prices[market.timeline[step]]


def evaluate(markets, qtypes, resolutions, base_rates, predict_fn, edge_threshold=0.05, notional=0.05):
    sq = []
    pnls = []
    trades_records = []  # (market_id, qtype, step, pred, mkt, edge, outcome, payoff)
    for m in markets:
        outcome = resolutions[m.market_id]
        qtype = qtypes[m.market_id]
        for step in range(len(m.timeline)):
            pred = predict_fn(m, step, qtype, base_rates)
            sq.append((pred - outcome) ** 2)
            mkt = m.market_prices[m.timeline[step]]
            edge = pred - mkt
            if 0.02 < mkt < 0.98 and abs(edge) > edge_threshold:
                if edge > 0:
                    payoff = ((1 - mkt) / mkt) * notional if outcome == 1 else -notional
                else:
                    payoff = (mkt / (1 - mkt)) * notional if outcome == 0 else -notional
                pnls.append(payoff)
                trades_records.append({
                    "market_id": m.market_id,
                    "qtype": qtype,
                    "step": step,
                    "pred": pred,
                    "mkt": mkt,
                    "edge": edge,
                    "outcome": outcome,
                    "payoff": payoff,
                    "question": m.question,
                })
    return {
        "brier": sum(sq) / len(sq) if sq else 0,
        "pnl": sum(pnls),
        "n_trades": len(pnls),
        "trades": trades_records,
    }


def compute_type_base_rates(train, qtypes, resolutions):
    by: dict[str, list[int]] = defaultdict(list)
    for m in train:
        if m.market_id in resolutions:
            by[qtypes[m.market_id]].append(resolutions[m.market_id])
    return {t: sum(o) / len(o) for t, o in by.items() if o}


def main():
    markets, qtypes, resolutions = load_v2()
    log.info("loaded %d markets", len(markets))

    # ---- 1. Temporal split: train on earlier, test on later ----
    by_close = sorted(markets, key=lambda m: m.closes_at)
    cut = len(by_close) // 2
    train = by_close[:cut]
    holdout = by_close[cut:]
    log.info("\n=== TEMPORAL SPLIT (train=earlier, holdout=later) ===")
    log.info("train: %d markets, %s..%s", len(train), train[0].closes_at, train[-1].closes_at)
    log.info("holdout: %d markets, %s..%s", len(holdout), holdout[0].closes_at, holdout[-1].closes_at)

    base_rates = compute_type_base_rates(train, qtypes, resolutions)
    log.info("train base rates: %s", {k: round(v, 3) for k, v in base_rates.items()})

    res_market = evaluate(holdout, qtypes, resolutions, base_rates, predict_market)
    res_v6 = evaluate(holdout, qtypes, resolutions, base_rates, predict_v6)

    log.info("\nholdout (later markets):")
    log.info("  v0_market_echo: brier=%.4f  pnl=%+0.3f  trades=%d",
             res_market["brier"], res_market["pnl"], res_market["n_trades"])
    log.info("  v6_shrink_typed: brier=%.4f  pnl=%+0.3f  trades=%d",
             res_v6["brier"], res_v6["pnl"], res_v6["n_trades"])
    log.info("  Δ brier = %+0.4f, Δ pnl = %+0.3f",
             res_v6["brier"] - res_market["brier"],
             res_v6["pnl"] - res_market["pnl"])

    # ---- 2. Per-qtype breakdown of v6 trades ----
    log.info("\n=== v6 trade breakdown by qtype (temporal holdout) ===")
    by_type = defaultdict(list)
    for t in res_v6["trades"]:
        by_type[t["qtype"]].append(t)
    for qtype, trades in sorted(by_type.items()):
        wins = sum(1 for t in trades if t["payoff"] > 0)
        pnl = sum(t["payoff"] for t in trades)
        log.info("  %-10s n=%-3d wins=%-3d hit=%.2f  pnl=%+0.3f",
                 qtype, len(trades), wins, wins / len(trades), pnl)

    # ---- 3. Per market price band ----
    log.info("\n=== v6 trade pnl by market price band (temporal holdout) ===")
    bands = [(0.02, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.98)]
    for lo, hi in bands:
        in_band = [t for t in res_v6["trades"] if lo <= t["mkt"] < hi]
        if not in_band:
            log.info("  [%.2f, %.2f) n=0", lo, hi)
            continue
        wins = sum(1 for t in in_band if t["payoff"] > 0)
        pnl = sum(t["payoff"] for t in in_band)
        log.info("  [%.2f, %.2f) n=%-3d wins=%-3d hit=%.2f  pnl=%+0.3f",
                 lo, hi, len(in_band), wins, wins / len(in_band), pnl)

    # ---- 4. Worst losing trades ----
    log.info("\n=== v6 worst losing trades (temporal holdout) ===")
    losses = sorted(res_v6["trades"], key=lambda t: t["payoff"])[:5]
    for t in losses:
        log.info("  %s [%s] mkt=%.3f pred=%.3f edge=%+0.3f outcome=%d payoff=%+0.3f",
                 t["market_id"], t["qtype"], t["mkt"], t["pred"], t["edge"], t["outcome"], t["payoff"])
        log.info("    Q: %s", t["question"][:80])

    # ---- 5. Best winning trades ----
    log.info("\n=== v6 best winning trades (temporal holdout) ===")
    wins = sorted(res_v6["trades"], key=lambda t: -t["payoff"])[:5]
    for t in wins:
        log.info("  %s [%s] mkt=%.3f pred=%.3f edge=%+0.3f outcome=%d payoff=%+0.3f",
                 t["market_id"], t["qtype"], t["mkt"], t["pred"], t["edge"], t["outcome"], t["payoff"])
        log.info("    Q: %s", t["question"][:80])

    # ---- 6. Bootstrap CI on PnL ----
    log.info("\n=== Bootstrap 95%% CI on PnL difference (temporal holdout) ===")
    payoffs = [t["payoff"] for t in res_v6["trades"]]
    if payoffs:
        rng = random.Random(42)
        boot = []
        n = len(payoffs)
        for _ in range(2000):
            sample = [payoffs[rng.randrange(n)] for _ in range(n)]
            boot.append(sum(sample))
        boot.sort()
        lo, hi = boot[50], boot[1950]
        log.info("  v6 total PnL = %+0.3f, 95%% CI = [%+0.3f, %+0.3f] over %d trades",
                 sum(payoffs), lo, hi, n)
        if lo > 0:
            log.info("  ✓ 95%% CI is strictly positive — edge is statistically robust.")
        elif hi < 0:
            log.info("  ✗ 95%% CI is strictly negative — strategy is losing.")
        else:
            log.info("  · 95%% CI spans zero — edge not statistically distinguishable.")

    out = Path("runs/live_iter/iterations_v4_redteam.json")
    out.write_text(json.dumps({
        "temporal_split": {
            "train_count": len(train),
            "holdout_count": len(holdout),
            "train_close_min": str(train[0].closes_at),
            "train_close_max": str(train[-1].closes_at),
            "holdout_close_min": str(holdout[0].closes_at),
            "holdout_close_max": str(holdout[-1].closes_at),
        },
        "base_rates": base_rates,
        "v0_market_echo": {
            "brier": res_market["brier"],
            "pnl": res_market["pnl"],
            "n_trades": res_market["n_trades"],
        },
        "v6_shrink_typed": {
            "brier": res_v6["brier"],
            "pnl": res_v6["pnl"],
            "n_trades": res_v6["n_trades"],
            "trades": res_v6["trades"],
        },
    }, indent=2))
    log.info("\nwrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
