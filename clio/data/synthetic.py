"""Synthetic market generator.

Produces a small self-consistent world:
- N markets across a few regimes
- For each market, a hidden true probability sampled per regime
- A timeline of as_of snapshot dates between observation and close
- A market price series that drifts toward the true probability over time
  (to simulate efficient information incorporation by other traders)
- A small corpus of news documents, each tagged to the market's keywords,
  with publish dates spread across the timeline. Documents have a `signal`
  field controlling how informative they are about the true probability.
- A resolution sampled from the true probability at close.

The generator is deterministic given a seed. This is the substrate the test
suite uses — anything the strategy "learns" should be learnable from this
fictional world.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from clio.frozen.corpus import Corpus, Document
from clio.frozen.oracle import ResolutionOracle
from clio.frozen.harness import Market


REGIMES = ("election", "sport", "financial", "geo", "scientific")

_REGIME_BASE_RATE = {
    "election": 0.50,
    "sport": 0.50,
    "financial": 0.40,
    "geo": 0.20,
    "scientific": 0.30,
}

_REGIME_KEYWORDS = {
    "election": ["candidate", "vote", "primary", "governor", "senate"],
    "sport": ["team", "championship", "match", "playoff", "tournament"],
    "financial": ["earnings", "revenue", "rate", "fed", "merger"],
    "geo": ["sanction", "ceasefire", "treaty", "border", "summit"],
    "scientific": ["trial", "approval", "phase", "fda", "study"],
}


@dataclass
class SyntheticConfig:
    n_markets: int = 40
    n_docs_per_market: int = 6
    timeline_steps: int = 4
    seed: int = 42
    market_efficiency: float = 0.6  # how strongly market prices drift toward truth
    news_signal_strength: float = 0.7  # how informative news doc LRs are
    start_date: date = field(default_factory=lambda: date(2024, 1, 1))
    market_window_days: int = 30
    news_lead_days: int = 25  # spread news docs over this many days before close


@dataclass
class SyntheticWorld:
    corpus: Corpus
    oracle: ResolutionOracle
    markets: list[Market]
    truths: dict[str, float]  # hidden true probabilities, for diagnostics


def generate_synthetic_world(config: SyntheticConfig | None = None) -> SyntheticWorld:
    cfg = config or SyntheticConfig()
    rng = random.Random(cfg.seed)

    corpus = Corpus()
    oracle = ResolutionOracle()
    markets: list[Market] = []
    truths: dict[str, float] = {}

    for i in range(cfg.n_markets):
        regime = rng.choice(REGIMES)
        keywords = _REGIME_KEYWORDS[regime]
        kw = rng.choice(keywords)

        base = _REGIME_BASE_RATE[regime]
        truth = max(0.02, min(0.98, rng.gauss(base, 0.20)))
        truths[f"M{i:04d}"] = truth

        observed = cfg.start_date + timedelta(days=rng.randint(0, 200))
        close = observed + timedelta(days=cfg.market_window_days)

        timeline = tuple(
            observed + timedelta(
                days=int((cfg.market_window_days - 1) * t / max(1, cfg.timeline_steps - 1))
            )
            for t in range(cfg.timeline_steps)
        )

        # Market prices drift toward truth with noise.
        market_prices = {}
        for step, d in enumerate(timeline):
            progress = step / max(1, cfg.timeline_steps - 1)
            anchor = 0.5
            mean = anchor + (truth - anchor) * cfg.market_efficiency * progress
            noise = rng.gauss(0, 0.04)
            market_prices[d] = max(0.01, min(0.99, mean + noise))

        question = f"Will {kw} event #{i} resolve YES by {close}?"
        market = Market(
            market_id=f"M{i:04d}",
            question=question,
            regime=regime,
            observed_at=observed,
            closes_at=close,
            timeline=timeline,
            market_prices=market_prices,
        )
        markets.append(market)

        # Resolution
        outcome = 1 if rng.random() < truth else 0
        oracle.record(market.market_id, outcome)

        # Generate news docs for this market.
        for j in range(cfg.n_docs_per_market):
            ds = observed + timedelta(
                days=rng.randint(-cfg.news_lead_days, cfg.market_window_days - 1)
            )
            ds = max(observed - timedelta(days=cfg.news_lead_days), ds)
            ds = min(close - timedelta(days=1), ds)

            # The doc's stance: P(YES | this doc) drifts toward outcome
            # proportional to news_signal_strength.
            yes_evidence = (
                cfg.news_signal_strength * outcome
                + (1 - cfg.news_signal_strength) * rng.random()
            )
            stance = "supports" if yes_evidence > 0.5 else "contradicts"
            content = (
                f"In the {kw} situation, sources indicate that the event {stance} "
                f"a YES resolution. {kw} {kw} {kw}. Detail level {j}."
            )

            doc = Document(
                doc_id=f"D{i:04d}-{j:02d}",
                published_at=ds,
                title=f"{kw.title()} update {j} for #{i}",
                content=content,
                source="synthetic",
                tags=(regime, kw),
            )
            corpus.add(doc)

    return SyntheticWorld(corpus=corpus, oracle=oracle, markets=markets, truths=truths)
