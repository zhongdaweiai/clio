"""Strategy = composition of micro-agents into a forecast pipeline.

The pipeline is intentionally rigid: prior → evidence → posterior → calibrate.
Replacing the order or skipping a stage is an architectural decision that should
be made explicit (a different `Strategy` subclass), not a flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from clio.agents.base_rater import BaseRater
from clio.agents.news_scout import NewsScout
from clio.agents.calibrator import Calibrator, IdentityCalibrator
from clio.frozen.corpus import Corpus
from clio.frozen.harness import Forecast, Market, StrategyProtocol


def _bayes_update(prior: float, lrs: list[float]) -> float:
    """Combine independent likelihood ratios with a prior.

    posterior_odds = prior_odds * prod(LR)
    """
    if prior <= 0.0:
        return 0.0
    if prior >= 1.0:
        return 1.0
    log_odds = math.log(prior / (1 - prior))
    for lr in lrs:
        if lr <= 0:
            continue
        log_odds += math.log(lr)
    # Numerical safety
    log_odds = max(-30.0, min(30.0, log_odds))
    return 1 / (1 + math.exp(-log_odds))


@dataclass
class ForecastTrace:
    """Diagnostic trace, emitted alongside the forecast for memory/audit."""
    prior: float
    n_evidence: int
    log_lr_total: float
    raw_posterior: float
    calibrated: float
    evidence_doc_ids: tuple[str, ...] = field(default_factory=tuple)


class BayesianStrategy:
    """Reference implementation: Base-rater → News Scout → Bayes → Calibrator."""

    def __init__(
        self,
        name: str,
        base_rater: BaseRater,
        news_scout: NewsScout,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.name = name
        self.base_rater = base_rater
        self.news_scout = news_scout
        self.calibrator = calibrator or IdentityCalibrator()
        self._last_traces: dict[tuple[str, date], ForecastTrace] = {}

    def forecast(self, market: Market, as_of: date, corpus: Corpus) -> Forecast:
        prior = self.base_rater(market, as_of)
        evidence = self.news_scout(market, as_of, corpus)
        lrs = [e.lr for e in evidence]
        raw_post = _bayes_update(prior, lrs)
        calibrated = self.calibrator(raw_post)

        trace = ForecastTrace(
            prior=prior,
            n_evidence=len(evidence),
            log_lr_total=sum(math.log(lr) for lr in lrs if lr > 0),
            raw_posterior=raw_post,
            calibrated=calibrated,
            evidence_doc_ids=tuple(e.doc.doc_id for e in evidence),
        )
        self._last_traces[(market.market_id, as_of)] = trace

        return Forecast(
            market_id=market.market_id,
            as_of=as_of,
            prob=calibrated,
            rationale=(
                f"prior={prior:.3f}, "
                f"n_evidence={len(evidence)}, "
                f"raw_posterior={raw_post:.3f}, "
                f"calibrated={calibrated:.3f}"
            ),
            evidence_doc_ids=trace.evidence_doc_ids,
        )

    def trace_for(self, market_id: str, as_of: date) -> ForecastTrace | None:
        return self._last_traces.get((market_id, as_of))


class MarketPriceStrategy:
    """Trivial baseline: just echo the market price.

    Useful as a Pareto reference. A strategy that cannot beat this on Brier is
    not contributing edge — it's just market-following with extra steps.
    """

    name = "market_baseline"

    def forecast(self, market: Market, as_of: date, corpus: Corpus) -> Forecast:
        p = market.market_prices.get(as_of, 0.5)
        return Forecast(market_id=market.market_id, as_of=as_of, prob=p)


# Static type assertion that these implement the protocol.
_: StrategyProtocol = MarketPriceStrategy()  # type: ignore[assignment]
