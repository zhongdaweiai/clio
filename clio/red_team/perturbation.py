"""Perturbation tests — attack a single forecast counterfactually.

Three perturbations:

1. **drop_strongest**: remove the evidence doc with the most extreme LR. If
   the prediction barely moves, the strategy isn't using its strongest signal
   (or is over-anchored on the prior). If it moves wildly, the strategy is
   too brittle.

2. **flip_strongest**: flip the LR of the strongest evidence to its
   reciprocal. If the prediction doesn't change much, the strategy is
   ignoring its evidence. If it flips entirely, the strategy is signal-
   dominated and one bad doc can poison the forecast.

3. **inject_adversary**: add an adversarially crafted document with a
   strong opposite-direction LR. Measures graceful degradation.

Outputs robustness scores. The gate uses these to fail strategies that are
either too rigid (sensitivity ≈ 0 to all perturbations) or too volatile
(any single doc flip changes the answer by > 0.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from statistics import mean, median
from typing import Sequence

from clio.agents.news_scout import Evidence
from clio.frozen.corpus import Corpus, Document
from clio.frozen.harness import Market
from clio.strategy import BayesianStrategy, _bayes_update


@dataclass
class PerturbationResult:
    market_id: str
    base_prob: float
    drop_strongest_prob: float
    flip_strongest_prob: float
    inject_adversary_prob: float
    n_evidence: int
    strongest_lr: float

    @property
    def drop_delta(self) -> float:
        return abs(self.base_prob - self.drop_strongest_prob)

    @property
    def flip_delta(self) -> float:
        return abs(self.base_prob - self.flip_strongest_prob)

    @property
    def inject_delta(self) -> float:
        return abs(self.base_prob - self.inject_adversary_prob)


@dataclass
class PerturbationReport:
    n_markets: int
    results: list[PerturbationResult] = field(default_factory=list)

    @property
    def mean_drop_sensitivity(self) -> float:
        return mean([r.drop_delta for r in self.results]) if self.results else 0.0

    @property
    def mean_flip_sensitivity(self) -> float:
        return mean([r.flip_delta for r in self.results]) if self.results else 0.0

    @property
    def mean_inject_sensitivity(self) -> float:
        return mean([r.inject_delta for r in self.results]) if self.results else 0.0

    @property
    def max_flip_sensitivity(self) -> float:
        return max([r.flip_delta for r in self.results], default=0.0)

    @property
    def fragility_score(self) -> float:
        """0 = totally rigid, 1 = totally volatile. Sweet spot ~0.2-0.5."""
        if not self.results:
            return 0.0
        return self.mean_flip_sensitivity

    def summary(self) -> dict[str, float]:
        return {
            "n_markets": float(self.n_markets),
            "mean_drop_sensitivity": self.mean_drop_sensitivity,
            "mean_flip_sensitivity": self.mean_flip_sensitivity,
            "mean_inject_sensitivity": self.mean_inject_sensitivity,
            "max_flip_sensitivity": self.max_flip_sensitivity,
            "fragility_score": self.fragility_score,
        }


def _compute_posterior(prior: float, lrs: Sequence[float], calibrator) -> float:
    raw = _bayes_update(prior, list(lrs))
    return calibrator(raw) if calibrator else raw


def run_perturbation(
    strategy: BayesianStrategy,
    markets: Sequence[Market],
    corpus: Corpus,
    *,
    inject_adversary_lr: float | None = None,
) -> PerturbationReport:
    """Replay each market's forecast under three perturbations.

    Re-uses the strategy's micro-agents directly so the test stays in-process.
    """
    results: list[PerturbationResult] = []

    for m in markets:
        # Use the latest pre-close timeline date for each market.
        timeline_pre_close = [d for d in m.timeline if d < m.closes_at]
        if not timeline_pre_close:
            continue
        as_of = max(timeline_pre_close)

        prior = strategy.base_rater(m, as_of)
        evidence = strategy.news_scout(m, as_of, corpus)
        lrs = [e.lr for e in evidence]
        if not lrs:
            base = _compute_posterior(prior, [], strategy.calibrator)
            results.append(
                PerturbationResult(
                    market_id=m.market_id,
                    base_prob=base,
                    drop_strongest_prob=base,
                    flip_strongest_prob=base,
                    inject_adversary_prob=base,
                    n_evidence=0,
                    strongest_lr=1.0,
                )
            )
            continue

        # Identify "strongest" — the one whose log|LR| is largest.
        idx_strongest = max(range(len(lrs)), key=lambda i: abs(math.log(max(1e-9, lrs[i]))))
        strongest_lr = lrs[idx_strongest]

        base = _compute_posterior(prior, lrs, strategy.calibrator)

        # Drop strongest
        lrs_drop = [lr for i, lr in enumerate(lrs) if i != idx_strongest]
        drop_p = _compute_posterior(prior, lrs_drop, strategy.calibrator)

        # Flip strongest (LR -> 1/LR, clamped to scout's own bounds)
        scout_floor = getattr(strategy.news_scout, "lr_floor", 0.1)
        scout_ceil = getattr(strategy.news_scout, "lr_ceil", 10.0)
        flipped = max(scout_floor, min(scout_ceil, 1.0 / max(1e-6, strongest_lr)))
        lrs_flip = list(lrs)
        lrs_flip[idx_strongest] = flipped
        flip_p = _compute_posterior(prior, lrs_flip, strategy.calibrator)

        # Inject adversary: add an opposite-direction LR equal in magnitude to
        # the strongest. Default: cap at scout's ceiling.
        if inject_adversary_lr is None:
            adv_lr = scout_floor if strongest_lr > 1 else scout_ceil
        else:
            adv_lr = inject_adversary_lr
        lrs_inject = list(lrs) + [adv_lr]
        inject_p = _compute_posterior(prior, lrs_inject, strategy.calibrator)

        results.append(
            PerturbationResult(
                market_id=m.market_id,
                base_prob=base,
                drop_strongest_prob=drop_p,
                flip_strongest_prob=flip_p,
                inject_adversary_prob=inject_p,
                n_evidence=len(lrs),
                strongest_lr=strongest_lr,
            )
        )

    return PerturbationReport(n_markets=len(results), results=results)
