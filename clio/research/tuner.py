"""Grid search and random search over StrategyParams.

Two surfaces:
- `grid_search`: exhaustive over a small parameter grid
- `random_search`: random samples from continuous distributions

Both return a list of (params, train_result) sorted by training PnL desc.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterable

from clio.frozen.harness import Market
from clio.research.strategies import StrategyParams, simulate


def grid_search(
    train_markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    base_rates: dict[str, float],
    objective: str = "pnl",  # "pnl" or "brier"
    lambda_grid: dict[str, list[float]] | None = None,
    trend_alphas: list[float] = (0.0, 0.25),
    time_decays: list[float] = (0.0, 1.0),
    edge_thresholds: list[float] = (0.04, 0.05, 0.07),
    kelly_fractions: list[float] = (0.0, 0.5),
    symmetric_options: list[bool] = (False, True),
) -> list[tuple[StrategyParams, dict]]:
    if lambda_grid is None:
        lambda_grid = {
            "event":    [0.0, 0.1, 0.2, 0.3],
            "field":    [0.0, 0.2, 0.3, 0.4],
            "deadline": [0.0, 0.1],
            "durative": [0.0],
        }
    qtypes_in_grid = list(lambda_grid.keys())
    out: list[tuple[StrategyParams, dict]] = []

    # Use itertools but bounded by sensible product size.
    from itertools import product as iproduct
    for combo in iproduct(*[lambda_grid[q] for q in qtypes_in_grid]):
        lam = tuple(zip(qtypes_in_grid, combo))
        for ta, td, et, kf, sym in iproduct(
            trend_alphas, time_decays, edge_thresholds, kelly_fractions, symmetric_options,
        ):
            params = StrategyParams(
                shrink_lambda=lam,
                trend_alpha=ta,
                time_decay=td,
                edge_threshold=et,
                kelly_fraction=kf,
                symmetric=sym,
            )
            res = simulate(params, train_markets, qtypes, resolutions, base_rates)
            out.append((params, res))

    if objective == "pnl":
        out.sort(key=lambda x: -x[1]["pnl"])
    else:
        out.sort(key=lambda x: x[1]["brier_overall"])
    return out


def random_search(
    train_markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    base_rates: dict[str, float],
    n_samples: int = 200,
    seed: int = 7,
    objective: str = "pnl",
) -> list[tuple[StrategyParams, dict]]:
    rng = random.Random(seed)
    out: list[tuple[StrategyParams, dict]] = []
    for _ in range(n_samples):
        params = _random_params(rng)
        res = simulate(params, train_markets, qtypes, resolutions, base_rates)
        out.append((params, res))
    if objective == "pnl":
        out.sort(key=lambda x: -x[1]["pnl"])
    else:
        out.sort(key=lambda x: x[1]["brier_overall"])
    return out


def _random_params(rng: random.Random) -> StrategyParams:
    return StrategyParams(
        shrink_lambda=(
            ("event",    round(rng.uniform(0.0, 0.4), 2)),
            ("field",    round(rng.uniform(0.0, 0.5), 2)),
            ("deadline", round(rng.uniform(0.0, 0.2), 2)),
            ("durative", 0.0),
        ),
        trend_alpha=round(rng.uniform(-0.3, 0.5), 2),
        time_decay=round(rng.uniform(0.0, 2.0), 1),
        edge_threshold=round(rng.uniform(0.03, 0.12), 2),
        kelly_fraction=round(rng.uniform(0.0, 1.0), 2),
        symmetric=rng.random() < 0.5,
    )
