"""Subset attack: mine holdout for strategy blind spots.

The Red Team's job is to read the strategy's holdout output and find
*features* of the question that predict failure. Concretely, partition the
holdout by feature buckets — regime, prior bucket, evidence-volume bucket,
market-confidence bucket, timeline-position bucket — and compute, per bucket:

- bucket Brier vs overall Brier
- bucket P&L delta (in fractional Kelly bankroll terms)
- bootstrap p-value: how often a same-size random subset shows this much
  degradation by chance?

Buckets with degradation > threshold AND bootstrap p-value < alpha are
returned as `BlindSpot`s.

This is deliberately model-free. We don't fit a classifier on top of the
forecasts — we just intersect with named features. That makes the output
interpretable ("the strategy fails on geo-regime markets when prior > 0.7")
and immediately actionable for the next iteration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from clio.frozen.harness import BacktestRun
from clio.frozen.scoring import brier_score, kelly_pnl


@dataclass(frozen=True)
class FeatureSlot:
    """A (feature_name, bucket_label) pair. Buckets are produced by `_bucketize`."""
    feature: str
    bucket: str

    def __str__(self) -> str:
        return f"{self.feature}={self.bucket}"


@dataclass
class BlindSpot:
    """A subset where the strategy systematically underperforms."""
    slot: FeatureSlot
    n: int
    bucket_brier: float
    overall_brier: float
    brier_degradation: float  # bucket - overall
    bucket_pnl_per_market: float
    overall_pnl_per_market: float
    bootstrap_p_value: float
    sample_market_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class SubsetAttackReport:
    overall_brier: float
    overall_pnl_per_market: float
    n_holdout: int
    blind_spots: list[BlindSpot]

    def worst(self) -> BlindSpot | None:
        if not self.blind_spots:
            return None
        return self.blind_spots[0]


def _bucketize_prob(p: float) -> str:
    if p < 0.25:
        return "low"
    if p < 0.50:
        return "lo-mid"
    if p < 0.75:
        return "hi-mid"
    return "high"


def _bucketize_count(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def _make_slots(
    market_ids: list[str],
    regimes: list[str],
    forecasts: list[float],
    market_prices: list[float],
    n_evidence: list[int],
) -> dict[FeatureSlot, list[int]]:
    """Return slot -> list of holdout indices."""
    slots: dict[FeatureSlot, list[int]] = {}

    def add(slot: FeatureSlot, idx: int) -> None:
        slots.setdefault(slot, []).append(idx)

    for i, _ in enumerate(market_ids):
        add(FeatureSlot("regime", regimes[i]), i)
        add(FeatureSlot("forecast_band", _bucketize_prob(forecasts[i])), i)
        add(FeatureSlot("market_price_band", _bucketize_prob(market_prices[i])), i)
        add(FeatureSlot("n_evidence", _bucketize_count(n_evidence[i])), i)
        # Cross-feature: did we strongly disagree with the market?
        gap = abs(forecasts[i] - market_prices[i])
        if gap < 0.05:
            disag = "tight"
        elif gap < 0.20:
            disag = "moderate"
        else:
            disag = "strong"
        add(FeatureSlot("disagreement", disag), i)

    return slots


def _bootstrap_p_value(
    overall_values: Sequence[float],
    bucket_values: Sequence[float],
    n_resamples: int,
    rng: random.Random,
) -> float:
    """Probability that a random same-size subset of `overall_values` has mean
    >= the bucket's mean (one-sided, where higher=worse). For Brier, higher
    means worse, so this is the "subset is at least this bad" tail."""
    n = len(bucket_values)
    if n == 0 or n >= len(overall_values):
        return 1.0
    bucket_mean = sum(bucket_values) / n
    pop = list(overall_values)
    hits = 0
    for _ in range(n_resamples):
        sample = rng.sample(pop, n)
        if sum(sample) / n >= bucket_mean:
            hits += 1
    return hits / n_resamples


def run_subset_attack(
    run: BacktestRun,
    regimes: list[str],
    n_evidence_per_market: list[int],
    *,
    min_bucket_size: int = 5,
    min_brier_degradation: float = 0.03,
    significance_level: float = 0.10,
    bootstrap_resamples: int = 1000,
    seed: int = 7,
) -> SubsetAttackReport:
    """Run the subset attack on a completed backtest run.

    `regimes` and `n_evidence_per_market` are aligned with the order of
    `run.final_probs` — i.e. one entry per scored market.
    """
    if not run.final_probs:
        return SubsetAttackReport(
            overall_brier=0.0,
            overall_pnl_per_market=0.0,
            n_holdout=0,
            blind_spots=[],
        )

    # Per-market squared error and per-market P&L contribution.
    sq_errors = [(p - o) ** 2 for p, o in zip(run.final_probs, run.final_outcomes)]
    overall_brier = sum(sq_errors) / len(sq_errors)

    # P&L per market: re-run kelly_pnl per single market with shared bankroll seed
    # Simple decomposition: simulate each market in isolation at the same fraction;
    # report stake-weighted PnL per market.
    pnls = _per_market_pnls(run.final_probs, run.final_outcomes, run.final_market_prices)
    overall_pnl = sum(pnls) / len(pnls)

    n_evi = list(n_evidence_per_market)
    if len(n_evi) != len(run.final_probs):
        n_evi = [0] * len(run.final_probs)

    market_ids = _ids_from_run(run)
    slots = _make_slots(market_ids, regimes, run.final_probs, run.final_market_prices, n_evi)

    rng = random.Random(seed)
    blind: list[BlindSpot] = []
    for slot, idxs in slots.items():
        if len(idxs) < min_bucket_size:
            continue
        bucket_sq = [sq_errors[i] for i in idxs]
        bucket_pnl = [pnls[i] for i in idxs]
        bucket_brier = sum(bucket_sq) / len(bucket_sq)
        bucket_pnl_avg = sum(bucket_pnl) / len(bucket_pnl)
        degradation = bucket_brier - overall_brier
        if degradation < min_brier_degradation:
            continue
        p = _bootstrap_p_value(sq_errors, bucket_sq, bootstrap_resamples, rng)
        if p > significance_level:
            continue
        sample_ids = tuple(market_ids[i] for i in idxs[:5])
        blind.append(
            BlindSpot(
                slot=slot,
                n=len(idxs),
                bucket_brier=bucket_brier,
                overall_brier=overall_brier,
                brier_degradation=degradation,
                bucket_pnl_per_market=bucket_pnl_avg,
                overall_pnl_per_market=overall_pnl,
                bootstrap_p_value=p,
                sample_market_ids=sample_ids,
            )
        )
    blind.sort(key=lambda b: -b.brier_degradation)
    return SubsetAttackReport(
        overall_brier=overall_brier,
        overall_pnl_per_market=overall_pnl,
        n_holdout=len(run.final_probs),
        blind_spots=blind,
    )


def _ids_from_run(run: BacktestRun) -> list[str]:
    # The harness orders final_probs in the same order it iterates `markets`.
    # We don't get explicit IDs out, so reconstruct from `forecasts` last-per-market.
    ids: list[str] = []
    seen: set[str] = set()
    for f in run.forecasts:
        if f.market_id not in seen:
            seen.add(f.market_id)
            ids.append(f.market_id)
    return ids


def _per_market_pnls(
    probs: Sequence[float],
    outcomes: Sequence[int],
    market_prices: Sequence[float],
    fraction: float = 0.05,
) -> list[float]:
    """Compute a stake-normalized per-market P&L (in units of bankroll
    fraction). Used for subset comparisons. Not a substitute for the
    full Kelly trajectory."""
    out: list[float] = []
    for p_hat, o, mkt in zip(probs, outcomes, market_prices):
        edge = p_hat - mkt
        if edge <= 0 or mkt <= 0 or mkt >= 1:
            out.append(0.0)
            continue
        b = (1 - mkt) / mkt
        f_kelly = (p_hat * b - (1 - p_hat)) / b
        f = max(0.0, min(fraction, 0.25 * f_kelly))
        if f == 0:
            out.append(0.0)
            continue
        if o == 1:
            out.append(f * (1 - mkt) / mkt)
        else:
            out.append(-f)
    return out
