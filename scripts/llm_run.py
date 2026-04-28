"""Master LLM-augmented run.

1. Precompute LLM forecasts for all 638 v3 markets at FIRST timeline as_of
   (one-bet-per-market matches our backtest harness)
2. Cache to disk (~$8 first run, free thereafter)
3. Run llm_backtest at multiple sizing tiers + short-horizon filters
4. Bootstrap CI on the best variant
5. Compare to rule-based statistical-only baseline
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from datetime import date
from pathlib import Path

from clio.agents.llm_anthropic import AnthropicLLMClient, LLMForecaster
from clio.cli import _load_markets_payload
from clio.data.news_pipeline import read_corpus_jsonl
from clio.research.llm_backtest import (
    ForecastCache, llm_backtest, precompute_llm_forecasts,
)
from clio.research.proper_backtest import CostModel
from clio.research.walk_forward import compute_base_rates


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("llm_run")


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


def report_one(label, res, days, bs, log):
    mult = res.final_bankroll / res.initial_bankroll if res.initial_bankroll > 0 else 0
    cag = cagr(res.initial_bankroll, res.final_bankroll, days) if days > 0 else 0
    mo = monthly(res.initial_bankroll, res.final_bankroll, days) if days > 0 else 0
    log.info(f"\n=== {label} ===")
    log.info(f"  ${res.initial_bankroll:>10,.0f} → ${res.final_bankroll:>13,.0f}  "
             f"({mult:>9,.2f}× over {days}d active)")
    log.info(f"  CAGR: {100*cag:+6.1f}%   monthly: {100*mo:+6.2f}%/mo")
    log.info(f"  trades: {res.n_trades}  hit: {100*res.hit_rate:.1f}%  PF: {res.profit_factor:.2f}  "
             f"avg_pos: {100*res.capital_deployed_pct:.1f}%  avg_days_held: {res.avg_days_held:.0f}")
    log.info(f"  Sharpe: {res.sharpe_annual:.2f}  Max DD: {100*res.max_drawdown_pct:.1f}%")
    if bs:
        log.info(f"  bootstrap: p10={bs['p10']:.2f}× p50={bs['p50']:.2f}× p90={bs['p90']:.2f}×  "
                 f"ruin: {100*bs['ruin']:.2f}%")


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set. source .env first.")
        return 1

    markets, oracle = _load_markets_payload("runs/live_iter/markets_v3.json")
    corpus = read_corpus_jsonl("runs/live_iter/news_v3.jsonl")
    qtypes = {}
    with open("runs/live_iter/markets_v3.json") as f:
        payload = json.load(f)
    for d in payload["markets"]:
        qtypes[d["market_id"]] = d.get("qtype", "event")
    resolutions = {k: int(v) for k, v in payload["resolutions"].items()}
    base_rates = compute_base_rates(markets, qtypes, resolutions)
    log.info("loaded %d markets, %d news docs, base_rates=%s",
             len(markets), len(corpus), {k: round(v, 3) for k, v in base_rates.items()})

    # ---- Phase 1: precompute LLM forecasts ----
    cache_path = Path("runs/live_iter/llm_forecast_cache.json")
    cache = ForecastCache(cache_path)
    log.info("forecast cache loaded: %d existing entries", len(cache))

    model = "claude-sonnet-4-6"
    llm = AnthropicLLMClient(model=model, max_retries=2)
    forecaster = LLMForecaster(llm)

    log.info("=" * 60)
    log.info("PHASE 1: precompute LLM forecasts (only_first_step=True)")
    log.info("=" * 60)
    t0 = time.time()
    n_new = precompute_llm_forecasts(
        markets, qtypes, base_rates, corpus, forecaster, cache,
        save_every=20, only_first_step=True, short_horizon_days=None,
    )
    cache.save()
    elapsed = time.time() - t0
    usage = llm.usage_summary()
    log.info("phase 1 done: %d new calls in %.0fs, total cache=%d",
             n_new, elapsed, len(cache))
    log.info("  usage: %d tokens in / %d tokens out = $%.2f",
             usage["tokens_in"], usage["tokens_out"], usage["estimated_cost_usd"])

    cost_model = CostModel()

    # ---- Phase 2: run LLM-augmented backtest at multiple tiers ----
    log.info("\n" + "=" * 60)
    log.info("PHASE 2: LLM-augmented backtest, $100K starting, multiple tiers")
    log.info("=" * 60)

    summaries = []
    raw = []

    tiers = [
        # label,                            edge,  pos,  gross, kelly, horizon
        ("conservative_LLM",                 0.10, 0.05, 0.50, 0.5,  None),
        ("moderate_LLM",                     0.10, 0.10, 0.80, 1.0,  None),
        ("proper_LLM (matches rule-best)",   0.10, 0.25, 1.50, 1.2,  None),
        ("aggressive_LLM",                   0.10, 0.30, 2.00, 1.5,  None),
        # Tighter edge filters
        ("LLM_edge>=0.15",                   0.15, 0.25, 1.50, 1.2,  None),
        ("LLM_edge>=0.20",                   0.20, 0.25, 1.50, 1.2,  None),
        # Short-horizon variants
        ("LLM_short<=60d",                   0.10, 0.25, 1.50, 1.2,  60),
        ("LLM_short<=30d",                   0.10, 0.25, 1.50, 1.2,  30),
        ("LLM_high_conv_short",              0.15, 0.25, 1.50, 1.2,  60),
    ]

    for label, edge_thr, max_pos, max_gross, kelly, horizon in tiers:
        res = llm_backtest(
            markets, qtypes, resolutions, cache, model,
            edge_threshold=edge_thr,
            initial_bankroll=100_000.0,
            max_position_pct=max_pos,
            max_gross_exposure_pct=max_gross,
            kelly_fraction=kelly,
            cost_model=cost_model,
            market_volumes={},
            short_horizon_days=horizon,
            skip_low_confidence=False,
        )
        ft, lt = first_last(res.trades)
        days = (lt - ft).days if ft and lt else 0
        bs = bootstrap(res)
        report_one(label, res, days, bs, log)
        summaries.append({
            "label": label, "final": res.final_bankroll,
            "mult": res.final_bankroll / res.initial_bankroll,
            "cagr": cagr(res.initial_bankroll, res.final_bankroll, days),
            "monthly": monthly(res.initial_bankroll, res.final_bankroll, days),
            "sharpe": res.sharpe_annual, "max_dd": res.max_drawdown_pct,
            "n_trades": res.n_trades, "hit_rate": res.hit_rate,
            "profit_factor": res.profit_factor,
            "avg_days_held": res.avg_days_held,
            "days": days, "bootstrap": bs,
            "edge_thr": edge_thr, "max_pos": max_pos, "kelly": kelly,
            "horizon": horizon,
        })
        raw.append((label, res))

    # ---- Side-by-side ----
    log.info("\n" + "=" * 100)
    log.info("LLM-AUGMENTED — SIDE BY SIDE")
    log.info("=" * 100)
    log.info(f"  {'tier':<35} {'final':>13} {'mult':>9} {'CAGR':>8} {'mo':>8} {'DD':>5} "
             f"{'PF':>5} {'trades':>7} {'hit':>5} {'days':>6} {'p10':>6} {'p50':>7} {'ruin%':>6}")
    log.info("  " + "-" * 130)
    for s in summaries:
        bs = s["bootstrap"]
        log.info(
            f"  {s['label']:<35} ${s['final']:>11,.0f} "
            f"{s['mult']:>7,.1f}× {100*s['cagr']:>+6.0f}% {100*s['monthly']:>+5.1f}%/m "
            f"{100*s['max_dd']:>4.0f}% {s['profit_factor']:>4.2f} "
            f"{s['n_trades']:>6d} {100*s['hit_rate']:>4.0f}% {s['avg_days_held']:>5.0f}d "
            f"{bs.get('p10',0):>4.2f}× {bs.get('p50',0):>5.2f}× {100*bs.get('ruin',0):>5.2f}%"
        )

    # Best by CAGR, best by p10
    best_cagr = max(summaries, key=lambda s: s["cagr"])
    best_p10 = max(summaries, key=lambda s: s["bootstrap"].get("p10", 0))
    log.info(f"\nBEST CAGR: {best_cagr['label']} → CAGR {100*best_cagr['cagr']:+.1f}%, "
             f"monthly {100*best_cagr['monthly']:+.2f}%/mo")
    log.info(f"BEST p10:  {best_p10['label']} → p10 {best_p10['bootstrap']['p10']:.2f}×")

    # Final usage
    usage = llm.usage_summary()
    log.info(f"\nFINAL LLM USAGE: {usage['calls']} new calls this run, "
             f"~${usage['estimated_cost_usd']:.3f}")

    out = Path("runs/live_iter/llm_run.json")
    out.write_text(json.dumps({
        "model": model,
        "n_markets": len(markets), "n_news_docs": len(corpus),
        "n_forecasts_cached": len(cache),
        "tiers": summaries,
        "llm_usage": usage,
    }, indent=2, default=str))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
