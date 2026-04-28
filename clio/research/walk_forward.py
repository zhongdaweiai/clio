"""Walk-forward validation: the gold standard for backtest honesty.

Roll through time. At each window:
  - Train on the past K months
  - Test on the next 1 month
  - Re-tune parameters per train window
  - Aggregate results across all test windows

If the strategy holds up across multiple non-overlapping test windows, that's
substantially stronger evidence than a single train/holdout split.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from clio.frozen.harness import Market
from clio.research.strategies import StrategyParams, simulate


def compute_base_rates(markets: list[Market], qtypes: dict[str, str], resolutions: dict[str, int]) -> dict[str, float]:
    by: dict[str, list[int]] = defaultdict(list)
    for m in markets:
        if m.market_id in resolutions:
            by[qtypes[m.market_id]].append(resolutions[m.market_id])
    return {t: sum(o) / len(o) for t, o in by.items() if o}


@dataclass
class WindowResult:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_train: int
    n_test: int
    train_base_rates: dict[str, float]
    chosen_params: StrategyParams
    market_brier: float
    strategy_brier: float
    brier_delta: float
    pnl: float
    n_trades: int
    n_wins: int


@dataclass
class WalkForwardReport:
    windows: list[WindowResult] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(w.pnl for w in self.windows)

    @property
    def total_trades(self) -> int:
        return sum(w.n_trades for w in self.windows)

    @property
    def total_wins(self) -> int:
        return sum(w.n_wins for w in self.windows)

    @property
    def n_pnl_positive_windows(self) -> int:
        return sum(1 for w in self.windows if w.pnl > 0)

    @property
    def n_brier_better_windows(self) -> int:
        return sum(1 for w in self.windows if w.brier_delta < 0)

    def all_payoffs(self, all_trades: list) -> list[float]:
        return [t.payoff for w_idx, w in enumerate(self.windows) for t in w._trades]


def walk_forward(
    markets: list[Market],
    qtypes: dict[str, str],
    resolutions: dict[str, int],
    *,
    train_window_days: int = 180,
    test_window_days: int = 60,
    step_days: int = 30,
    tune_fn: Callable[[list[Market], dict, dict, dict], StrategyParams] | None = None,
    market_baseline_params: StrategyParams | None = None,
) -> WalkForwardReport:
    """Roll a (train, test) window across the dataset.

    `tune_fn(train_markets, qtypes, resolutions, base_rates) -> best_params`
    is called per window to choose parameters from the train window. If None,
    a default mid-range fixed StrategyParams is used.
    """
    sorted_markets = sorted(markets, key=lambda m: m.closes_at)
    if not sorted_markets:
        return WalkForwardReport()

    earliest = sorted_markets[0].closes_at
    latest = sorted_markets[-1].closes_at
    if market_baseline_params is None:
        from clio.research.strategies import StrategyParams as _SP
        market_baseline_params = _SP(shrink_lambda=(("event", 0.0), ("field", 0.0), ("deadline", 0.0), ("durative", 0.0)))

    report = WalkForwardReport()
    test_start = earliest + timedelta(days=train_window_days)
    while test_start + timedelta(days=test_window_days) <= latest + timedelta(days=1):
        train_start = test_start - timedelta(days=train_window_days)
        train_end = test_start
        test_end = test_start + timedelta(days=test_window_days)

        train = [m for m in sorted_markets if train_start <= m.closes_at < train_end]
        test = [m for m in sorted_markets if test_start <= m.closes_at < test_end]

        if len(train) < 20 or len(test) < 5:
            test_start += timedelta(days=step_days)
            continue

        base_rates = compute_base_rates(train, qtypes, resolutions)
        if tune_fn is not None:
            chosen = tune_fn(train, qtypes, resolutions, base_rates)
        else:
            chosen = StrategyParams()

        market_res = simulate(market_baseline_params, test, qtypes, resolutions, base_rates)
        strat_res = simulate(chosen, test, qtypes, resolutions, base_rates)

        wr = WindowResult(
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            n_train=len(train), n_test=len(test),
            train_base_rates=base_rates,
            chosen_params=chosen,
            market_brier=market_res["brier_overall"],
            strategy_brier=strat_res["brier_overall"],
            brier_delta=strat_res["brier_overall"] - market_res["brier_overall"],
            pnl=strat_res["pnl"],
            n_trades=strat_res["n_trades"],
            n_wins=strat_res["n_wins"],
        )
        # Stash the trade list for later analysis.
        wr._trades = strat_res["trades"]  # type: ignore[attr-defined]
        report.windows.append(wr)
        test_start += timedelta(days=step_days)
    return report


def bootstrap_pnl_ci(payoffs: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap CI on total PnL given the realized trade payoffs."""
    if not payoffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(payoffs)
    boot = []
    for _ in range(n_resamples):
        sample = [payoffs[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample))
    boot.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1
    return boot[lo_idx], boot[hi_idx]
