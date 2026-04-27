"""Backtest harness — replays markets through a strategy.

The harness is the thing agents cannot change. It defines:
- the order in which markets are presented
- the as_of timeline for each market (multiple time points)
- the cutoff enforcement (delegated to Corpus)
- the scoring (delegated to scoring.py)

A strategy returns a forecast at each (market, as_of). The harness records
all of them and produces a ScoreVector at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, Sequence

from clio.frozen.corpus import Corpus
from clio.frozen.oracle import ResolutionOracle
from clio.frozen.scoring import ScoreVector, score_vector


@dataclass(frozen=True)
class Market:
    market_id: str
    question: str
    regime: str  # "election" | "sport" | "financial" | "geo" | ...
    observed_at: date  # first time the market existed
    closes_at: date
    timeline: tuple[date, ...]  # snapshot dates we score forecasts at
    market_prices: dict[date, float]  # market mid at each snapshot date


@dataclass
class Forecast:
    market_id: str
    as_of: date
    prob: float
    rationale: str = ""
    evidence_doc_ids: tuple[str, ...] = field(default_factory=tuple)


class StrategyProtocol(Protocol):
    name: str

    def forecast(
        self,
        market: Market,
        as_of: date,
        corpus: Corpus,
    ) -> Forecast: ...


@dataclass
class BacktestRun:
    strategy_name: str
    forecasts: list[Forecast] = field(default_factory=list)
    final_probs: list[float] = field(default_factory=list)
    final_outcomes: list[int] = field(default_factory=list)
    final_market_prices: list[float] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    score: ScoreVector | None = None
    regime_breakdown: dict[str, ScoreVector] = field(default_factory=dict)


class BacktestHarness:
    def __init__(
        self,
        corpus: Corpus,
        oracle: ResolutionOracle,
        markets: Sequence[Market],
    ) -> None:
        self.corpus = corpus
        self.oracle = oracle
        self.markets = list(markets)

    def run(self, strategy: StrategyProtocol) -> BacktestRun:
        run = BacktestRun(strategy_name=strategy.name)
        regime_buckets: dict[str, list[tuple[float, int, float]]] = {}

        for market in self.markets:
            if market.market_id not in self.oracle:
                run.skipped.append(market.market_id)
                continue

            for as_of in market.timeline:
                if as_of >= market.closes_at:
                    continue
                fc = strategy.forecast(market, as_of, self.corpus)
                fc = self._sanitize(fc, market, as_of)
                run.forecasts.append(fc)

            timeline_before_close = [
                d for d in market.timeline if d < market.closes_at
            ]
            if not timeline_before_close:
                run.skipped.append(market.market_id)
                continue
            final_as_of = max(timeline_before_close)
            final_fc = next(
                f
                for f in reversed(run.forecasts)
                if f.market_id == market.market_id and f.as_of == final_as_of
            )
            outcome = self.oracle.lookup(market.market_id)
            mkt_price = market.market_prices.get(final_as_of, 0.5)

            run.final_probs.append(final_fc.prob)
            run.final_outcomes.append(outcome)
            run.final_market_prices.append(mkt_price)
            regime_buckets.setdefault(market.regime, []).append(
                (final_fc.prob, outcome, mkt_price)
            )

        run.score = score_vector(
            run.final_probs,
            run.final_outcomes,
            run.final_market_prices,
            coverage=1.0 - len(run.skipped) / max(1, len(self.markets)),
            regime_breadth=len(regime_buckets),
        )
        for regime, rows in regime_buckets.items():
            ps, os_, mps = zip(*rows)
            run.regime_breakdown[regime] = score_vector(
                list(ps), list(os_), list(mps), coverage=1.0, regime_breadth=1
            )
        return run

    @staticmethod
    def _sanitize(fc: Forecast, market: Market, as_of: date) -> Forecast:
        if not (0.0 <= fc.prob <= 1.0):
            raise ValueError(
                f"strategy returned out-of-range prob={fc.prob} on {market.market_id}@{as_of}"
            )
        if fc.market_id != market.market_id or fc.as_of != as_of:
            raise ValueError(
                f"strategy returned wrong market/date in forecast: {fc}"
            )
        return fc
