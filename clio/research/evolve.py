"""Genetic / evolutionary search over StrategyParams.

Population evolves across generations:
- Selection: rank by train PnL, keep top K
- Crossover: blend two parents' parameters
- Mutation: jitter each parameter by some sigma

Pareto-aware version keeps non-dominated members, not just top-PnL.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

from clio.frozen.harness import Market
from clio.research.ensemble import pareto_filter_strategies
from clio.research.strategies import StrategyParams, simulate
from clio.research.tuner import _random_params


def _crossover(a: StrategyParams, b: StrategyParams, rng: random.Random) -> StrategyParams:
    pick = lambda f1, f2: f1 if rng.random() < 0.5 else f2
    new_lambda = []
    a_dict = a.lambda_dict
    b_dict = b.lambda_dict
    keys = sorted(set(a_dict) | set(b_dict))
    for k in keys:
        v = pick(a_dict.get(k, 0.0), b_dict.get(k, 0.0))
        new_lambda.append((k, round(v, 2)))
    return StrategyParams(
        shrink_lambda=tuple(new_lambda),
        trend_alpha=pick(a.trend_alpha, b.trend_alpha),
        time_decay=pick(a.time_decay, b.time_decay),
        edge_threshold=pick(a.edge_threshold, b.edge_threshold),
        kelly_fraction=pick(a.kelly_fraction, b.kelly_fraction),
        symmetric=pick(a.symmetric, b.symmetric),
    )


def _mutate(p: StrategyParams, rng: random.Random, sigma: float = 0.05) -> StrategyParams:
    def jitter(v: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, v + rng.gauss(0, sigma)))

    new_lambda = []
    for k, v in p.shrink_lambda:
        new_v = round(jitter(v, 0.0, 0.5), 2)
        new_lambda.append((k, new_v))
    return StrategyParams(
        shrink_lambda=tuple(new_lambda),
        trend_alpha=round(jitter(p.trend_alpha, -0.3, 0.5), 2),
        time_decay=round(jitter(p.time_decay, 0.0, 2.0), 1),
        edge_threshold=round(jitter(p.edge_threshold, 0.03, 0.12), 2),
        kelly_fraction=round(jitter(p.kelly_fraction, 0.0, 1.0), 2),
        symmetric=p.symmetric if rng.random() < 0.9 else not p.symmetric,
    )


@dataclass
class EvolutionLog:
    generations: list[dict] = field(default_factory=list)


def evolve(
    train_markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    base_rates: dict[str, float],
    *,
    population_size: int = 60,
    elite_size: int = 20,
    n_generations: int = 8,
    mutation_sigma: float = 0.05,
    seed: int = 7,
    pareto_aware: bool = True,
) -> tuple[list[tuple[StrategyParams, dict]], EvolutionLog]:
    rng = random.Random(seed)
    log = EvolutionLog()

    # Init: random
    population = [_random_params(rng) for _ in range(population_size)]
    evaluated: list[tuple[StrategyParams, dict]] = [
        (p, simulate(p, train_markets, qtypes, resolutions, base_rates)) for p in population
    ]

    for gen in range(n_generations):
        evaluated.sort(key=lambda x: -x[1]["pnl"])
        elites = evaluated[:elite_size]
        if pareto_aware:
            pf = pareto_filter_strategies(evaluated, axes=("brier_overall", "pnl"), minimize=(True, False))
            # Merge elites + Pareto front, dedupe.
            merged: list[tuple[StrategyParams, dict]] = []
            seen = set()
            for p, r in elites + pf:
                key = (tuple(p.shrink_lambda), p.trend_alpha, p.time_decay, p.edge_threshold, p.kelly_fraction, p.symmetric)
                if key in seen:
                    continue
                seen.add(key)
                merged.append((p, r))
            elites = merged[:elite_size]

        gen_log = {
            "generation": gen,
            "best_pnl": evaluated[0][1]["pnl"],
            "best_brier": min(r["brier_overall"] for _, r in evaluated),
            "median_pnl": sorted(r["pnl"] for _, r in evaluated)[len(evaluated) // 2],
        }
        log.generations.append(gen_log)

        # Generate new population from elites: crossover + mutation
        new_pop_params: list[StrategyParams] = [p for p, _ in elites]  # carry over elites
        while len(new_pop_params) < population_size:
            a = elites[rng.randrange(len(elites))][0]
            b = elites[rng.randrange(len(elites))][0]
            child = _crossover(a, b, rng)
            child = _mutate(child, rng, sigma=mutation_sigma)
            new_pop_params.append(child)

        evaluated = [
            (p, simulate(p, train_markets, qtypes, resolutions, base_rates))
            for p in new_pop_params
        ]

    evaluated.sort(key=lambda x: -x[1]["pnl"])
    return evaluated, log
