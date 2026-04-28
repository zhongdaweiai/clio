"""Master iteration: 638 markets, auto-tuning, walk-forward, evolution, ensemble.

The full closed loop: scale → tune → validate → fuse → verify.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from clio.frozen.harness import Market
from clio.research.ensemble import (
    make_brier_weighted_ensemble,
    pareto_filter_strategies,
    simulate_ensemble,
)
from clio.research.evolve import evolve
from clio.research.strategies import StrategyParams, simulate
from clio.research.tuner import grid_search, random_search
from clio.research.walk_forward import (
    bootstrap_pnl_ci,
    compute_base_rates,
    walk_forward,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("master")


# ----- data -----


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


# ----- temporal split helpers -----


def split_temporal(markets, frac=0.7):
    by_close = sorted(markets, key=lambda m: m.closes_at)
    cut = int(len(by_close) * frac)
    return by_close[:cut], by_close[cut:]


# ----- tune fn used by walk-forward -----


def tune_grid(train_markets, qtypes, resolutions, base_rates) -> StrategyParams:
    """Per-window tuning: small grid search, pick top by train PnL."""
    sweep = grid_search(
        train_markets, qtypes, resolutions, base_rates,
        objective="pnl",
        lambda_grid={
            "event":    [0.0, 0.1, 0.2, 0.3],
            "field":    [0.0, 0.2, 0.3, 0.4],
            "deadline": [0.0, 0.1, 0.2],
            "durative": [0.0],
        },
        trend_alphas=[0.0, 0.25],
        time_decays=[0.0, 1.0],
        edge_thresholds=[0.04, 0.06],
        kelly_fractions=[0.0, 0.5],
        symmetric_options=[False, True],
    )
    return sweep[0][0]


# ----- main -----


def main() -> int:
    t0 = time.time()
    markets, qtypes, resolutions = load_v3()
    log.info("loaded %d markets, %d resolved", len(markets), len(resolutions))
    yes_by_type = defaultdict(list)
    for m in markets:
        yes_by_type[qtypes[m.market_id]].append(resolutions[m.market_id])
    log.info("by qtype:")
    for t, outs in sorted(yes_by_type.items()):
        log.info("  %-10s n=%-4d YES=%.1f%%", t, len(outs), 100 * sum(outs) / len(outs))

    train, holdout = split_temporal(markets, frac=0.7)
    log.info("\ntemporal split: train=%d (%s..%s), holdout=%d (%s..%s)",
             len(train), train[0].closes_at, train[-1].closes_at,
             len(holdout), holdout[0].closes_at, holdout[-1].closes_at)
    base_rates = compute_base_rates(train, qtypes, resolutions)
    log.info("train base rates: %s", {k: round(v, 3) for k, v in base_rates.items()})

    # --- Phase 1: grid search top candidates on train ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: grid search on train (%d markets)", len(train))
    log.info("=" * 70)
    candidates = grid_search(train, qtypes, resolutions, base_rates, objective="pnl")
    log.info("evaluated %d parameter combinations", len(candidates))
    log.info("top 8 by train PnL:")
    for i, (p, r) in enumerate(candidates[:8]):
        log.info("  #%d  brier=%.4f  pnl=%+0.3f  trades=%d  λ=%s  trend=%.2f td=%.1f et=%.2f kelly=%.1f sym=%s",
                 i + 1, r["brier_overall"], r["pnl"], r["n_trades"],
                 dict(p.shrink_lambda), p.trend_alpha, p.time_decay,
                 p.edge_threshold, p.kelly_fraction, p.symmetric)

    market_baseline = StrategyParams(
        shrink_lambda=(("event", 0.0), ("field", 0.0), ("deadline", 0.0), ("durative", 0.0)),
    )
    market_holdout = simulate(market_baseline, holdout, qtypes, resolutions, base_rates)
    log.info("\nholdout market baseline: brier=%.4f  pnl=%+0.3f", market_holdout["brier_overall"], market_holdout["pnl"])

    # --- Phase 2: evaluate top-1 on holdout ---
    top1, top1_train_res = candidates[0]
    top1_hold = simulate(top1, holdout, qtypes, resolutions, base_rates)
    log.info("\ntop-1 on holdout:")
    log.info("  brier=%.4f (Δ=%+0.4f)  pnl=%+0.3f  trades=%d  wins=%d (%.0f%%)",
             top1_hold["brier_overall"], top1_hold["brier_overall"] - market_holdout["brier_overall"],
             top1_hold["pnl"], top1_hold["n_trades"], top1_hold["n_wins"],
             100 * top1_hold["n_wins"] / max(1, top1_hold["n_trades"]))
    if top1_hold["trades"]:
        payoffs = [t.payoff for t in top1_hold["trades"]]
        ci_lo, ci_hi = bootstrap_pnl_ci(payoffs)
        log.info("  bootstrap 95%% CI on PnL: [%+0.3f, %+0.3f]", ci_lo, ci_hi)

    # --- Phase 3: random search to fill out the parameter space ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: random search (n=300)")
    log.info("=" * 70)
    rand_candidates = random_search(train, qtypes, resolutions, base_rates, n_samples=300, seed=11)
    log.info("top 5 random:")
    for i, (p, r) in enumerate(rand_candidates[:5]):
        log.info("  #%d  brier=%.4f  pnl=%+0.3f  trades=%d", i + 1, r["brier_overall"], r["pnl"], r["n_trades"])

    # --- Phase 4: Pareto filter across grid + random ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: Pareto frontier across grid+random candidates")
    log.info("=" * 70)
    all_candidates = candidates + rand_candidates
    pareto = pareto_filter_strategies(all_candidates, axes=("brier_overall", "pnl"), minimize=(True, False), max_keep=15)
    log.info("Pareto frontier (top 15 by PnL):")
    for i, (p, r) in enumerate(pareto):
        log.info("  #%d  brier=%.4f  pnl=%+0.3f  trades=%d  sym=%s",
                 i + 1, r["brier_overall"], r["pnl"], r["n_trades"], p.symmetric)

    # Take top-K Pareto members for the ensemble.
    ensemble_members = pareto[:min(8, len(pareto))]
    members_params, members_weights = make_brier_weighted_ensemble(ensemble_members)
    ensemble_hold = simulate_ensemble(
        members_params, members_weights, edge_threshold=0.05, notional=0.05,
        markets=holdout, qtypes=qtypes, resolutions=resolutions, base_rates=base_rates,
    )
    log.info("\nensemble (Brier-weighted, %d members) on holdout:", len(ensemble_members))
    log.info("  brier=%.4f (Δ=%+0.4f)  pnl=%+0.3f  trades=%d  wins=%d (%.0f%%)",
             ensemble_hold["brier_overall"], ensemble_hold["brier_overall"] - market_holdout["brier_overall"],
             ensemble_hold["pnl"], ensemble_hold["n_trades"], ensemble_hold["n_wins"],
             100 * ensemble_hold["n_wins"] / max(1, ensemble_hold["n_trades"]))
    if ensemble_hold["trades"]:
        payoffs = [t["payoff"] for t in ensemble_hold["trades"]]
        ci_lo, ci_hi = bootstrap_pnl_ci(payoffs)
        log.info("  bootstrap 95%% CI: [%+0.3f, %+0.3f]", ci_lo, ci_hi)

    # --- Phase 5: Genetic evolution ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 5: genetic evolution (8 generations, pop=60)")
    log.info("=" * 70)
    evolved, evo_log = evolve(
        train, qtypes, resolutions, base_rates,
        population_size=60, elite_size=20, n_generations=8, mutation_sigma=0.05, seed=23,
    )
    for g in evo_log.generations:
        log.info("  gen %d: best_pnl=%+0.3f  best_brier=%.4f  median_pnl=%+0.3f",
                 g["generation"], g["best_pnl"], g["best_brier"], g["median_pnl"])
    best_evolved, best_evolved_train = evolved[0]
    best_evolved_hold = simulate(best_evolved, holdout, qtypes, resolutions, base_rates)
    log.info("\nbest evolved on holdout:")
    log.info("  brier=%.4f (Δ=%+0.4f)  pnl=%+0.3f  trades=%d",
             best_evolved_hold["brier_overall"],
             best_evolved_hold["brier_overall"] - market_holdout["brier_overall"],
             best_evolved_hold["pnl"], best_evolved_hold["n_trades"])
    log.info("  params: λ=%s  trend=%.2f td=%.1f et=%.2f kelly=%.2f sym=%s",
             dict(best_evolved.shrink_lambda), best_evolved.trend_alpha, best_evolved.time_decay,
             best_evolved.edge_threshold, best_evolved.kelly_fraction, best_evolved.symmetric)

    # --- Phase 6: Walk-forward across multiple windows ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 6: walk-forward validation")
    log.info("=" * 70)
    wf = walk_forward(
        markets, qtypes, resolutions,
        train_window_days=180, test_window_days=60, step_days=45,
        tune_fn=tune_grid,
    )
    log.info("ran %d windows", len(wf.windows))
    log.info("%-3s %-12s %-12s %-7s %-7s %-9s %-9s %-9s %-9s %-7s",
             "#", "test_start", "test_end", "n_tr", "n_te",
             "mkt_brier", "str_brier", "Δ brier", "pnl", "trades")
    for i, w in enumerate(wf.windows):
        log.info("%-3d %-12s %-12s %-7d %-7d %-9.4f %-9.4f %+0.4f   %+0.3f   %-7d",
                 i + 1, w.test_start, w.test_end, w.n_train, w.n_test,
                 w.market_brier, w.strategy_brier, w.brier_delta, w.pnl, w.n_trades)
    log.info("---")
    log.info("aggregate: pnl=%+0.3f  trades=%d  wins=%d (%.0f%%)  brier-better windows=%d/%d  pnl-positive windows=%d/%d",
             wf.total_pnl, wf.total_trades, wf.total_wins,
             100 * wf.total_wins / max(1, wf.total_trades),
             wf.n_brier_better_windows, len(wf.windows),
             wf.n_pnl_positive_windows, len(wf.windows))
    all_payoffs = [t.payoff for w in wf.windows for t in w._trades]
    if all_payoffs:
        ci_lo, ci_hi = bootstrap_pnl_ci(all_payoffs)
        log.info("walk-forward bootstrap 95%% CI on aggregate PnL: [%+0.3f, %+0.3f]", ci_lo, ci_hi)

    # ----- save full report -----
    out = Path("runs/live_iter/iterations_v5.json")
    out.write_text(json.dumps({
        "wall_seconds": time.time() - t0,
        "n_markets": len(markets),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "base_rates": base_rates,
        "market_baseline_holdout": {
            "brier": market_holdout["brier_overall"], "pnl": market_holdout["pnl"],
        },
        "top1_holdout": {
            "params": top1.to_dict(),
            "brier": top1_hold["brier_overall"],
            "pnl": top1_hold["pnl"],
            "n_trades": top1_hold["n_trades"],
        },
        "ensemble_holdout": {
            "n_members": len(ensemble_members),
            "weights": members_weights,
            "brier": ensemble_hold["brier_overall"],
            "pnl": ensemble_hold["pnl"],
            "n_trades": ensemble_hold["n_trades"],
        },
        "best_evolved_holdout": {
            "params": best_evolved.to_dict(),
            "brier": best_evolved_hold["brier_overall"],
            "pnl": best_evolved_hold["pnl"],
        },
        "walk_forward": {
            "n_windows": len(wf.windows),
            "total_pnl": wf.total_pnl,
            "total_trades": wf.total_trades,
            "n_brier_better_windows": wf.n_brier_better_windows,
            "n_pnl_positive_windows": wf.n_pnl_positive_windows,
            "windows": [
                {
                    "test_start": str(w.test_start), "test_end": str(w.test_end),
                    "n_train": w.n_train, "n_test": w.n_test,
                    "market_brier": w.market_brier, "strategy_brier": w.strategy_brier,
                    "brier_delta": w.brier_delta, "pnl": w.pnl, "n_trades": w.n_trades,
                    "chosen_params": w.chosen_params.to_dict(),
                }
                for w in wf.windows
            ],
        },
    }, indent=2, default=str))
    log.info("\nwrote %s in %.1fs", out, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
