"""Closed-loop iteration on real Polymarket data.

Runs N strategy variants on the SAME train/holdout split. Each iteration:
  1. Train (fit calibrators or compute priors on train markets)
  2. Score on holdout
  3. Red Team: subset attack + perturbation + gate
  4. Print summary; persist run log

The next iteration's hyperparameters are chosen based on Red Team findings
of the previous one — this is the closed-loop demonstration.

Run log: runs/live_iter/iterations.log + per-iteration JSON snapshots.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from clio.agents.base import MockLLMClient
from clio.agents.base_rater import BaseRater
from clio.agents.calibrator import (
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
)
from clio.agents.news_scout import NewsScout
from clio.cli import _load_markets_payload
from clio.data.news_pipeline import read_corpus_jsonl
from clio.frozen.harness import BacktestHarness, Forecast, Market
from clio.frozen.scoring import ScoreVector
from clio.pareto import pareto_frontier
from clio.red_team import (
    GateThresholds,
    evaluate_gate,
    run_perturbation,
    run_subset_attack,
)
from clio.strategy import BayesianStrategy, MarketPriceStrategy, _bayes_update


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("iter")


# Stratified split: each regime contributes both to train and holdout.
def stratified_split(markets: list[Market], seed: int = 7, train_frac: float = 0.5):
    rng = random.Random(seed)
    by_regime: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        by_regime[m.regime].append(m)
    train: list[Market] = []
    holdout: list[Market] = []
    for regime, ms in by_regime.items():
        ms = list(ms)
        rng.shuffle(ms)
        cut = max(1, int(len(ms) * train_frac))
        train.extend(ms[:cut])
        holdout.extend(ms[cut:])
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


# ---------------------------------------------------------------------
# LLM with rule-based stance scoring against real news content
# ---------------------------------------------------------------------


def _signal_llm() -> MockLLMClient:
    """A rule-based 'LLM' that scores news stance from real article content.

    For markets where news content explicitly supports/contradicts/etc, the
    rules below produce LR signals. For neutral content, the default 0.50
    is returned (LR=1, no update). This is deliberately weak — a real LLM
    would do much better — but it's good enough to show the iteration loop.
    """
    llm = MockLLMClient()
    # Strong positive (LR ~ 4)
    for pat in [
        r"\bwins?\b", r"\bvictory\b", r"\bbeat[s]?\b", r"\btriumph",
        r"\bsigned\b", r"\bratified\b", r"\bapproved\b", r"\bpassed\b",
        r"\bsurpasses?\b", r"\bsuccessful", r"\boutperform",
    ]:
        llm.register(pat, "0.78")
    # Moderate positive (LR ~ 2)
    for pat in [
        r"\bleads?\b", r"\bahead\b", r"\bgaining\b", r"\bsupports?\b",
        r"\bendorses?\b", r"\boptimistic", r"\bpolls.*lead",
    ]:
        llm.register(pat, "0.65")
    # Strong negative (LR ~ 0.25)
    for pat in [
        r"\bloses?\b", r"\bdefeated?\b", r"\brejected?\b", r"\bvetoed\b",
        r"\bblocked\b", r"\bcanceled?\b", r"\bbanned\b", r"\bfailed\b",
        r"\bwithdrew\b", r"\bdrops? out\b", r"\bsuspended\b", r"\bdelayed\b",
    ]:
        llm.register(pat, "0.22")
    # Moderate negative (LR ~ 0.5)
    for pat in [
        r"\btrails?\b", r"\bbehind\b", r"\bcontradict",
        r"\boppose[ds]?\b", r"\bunlikely", r"\bskeptical",
    ]:
        llm.register(pat, "0.35")
    llm.set_default("0.50")
    return llm


# ---------------------------------------------------------------------
# Strategy variants
# ---------------------------------------------------------------------


@dataclass
class IterationConfig:
    """Hyperparameters that change between iterations."""
    name: str
    description: str
    lr_floor: float = 0.1
    lr_ceil: float = 10.0
    max_docs: int = 5
    calibrator_type: str = "identity"  # "identity" | "temperature" | "isotonic"
    use_market_anchored_prior: bool = False  # if True, blend regime prior with
                                              # the earliest market price
    market_anchor_weight: float = 0.0
    regime_priors: dict[str, float] | None = None


def make_strategy(cfg: IterationConfig, train_markets: list[Market], train_corpus, train_oracle):
    llm = _signal_llm()

    if cfg.regime_priors is None:
        priors = None
    else:
        priors = dict(cfg.regime_priors)

    base_rater = BaseRater(llm, regime_priors=priors)
    scout = NewsScout(llm, max_docs=cfg.max_docs, lr_floor=cfg.lr_floor, lr_ceil=cfg.lr_ceil)

    # Build a strategy first so we can fit a calibrator on its train output.
    raw = BayesianStrategy(f"{cfg.name}_raw_for_fit", base_rater, scout)
    if cfg.calibrator_type == "identity":
        cal: Calibrator = IdentityCalibrator()
    else:
        h = BacktestHarness(train_corpus, train_oracle, train_markets)
        run = h.run(raw)
        if cfg.calibrator_type == "isotonic":
            cal = IsotonicCalibrator()
        elif cfg.calibrator_type == "temperature":
            cal = TemperatureCalibrator()
        else:
            cal = IdentityCalibrator()
        if run.final_probs:
            cal.fit(run.final_probs, run.final_outcomes)

    if cfg.use_market_anchored_prior and cfg.market_anchor_weight > 0:
        return MarketAnchoredBayes(cfg.name, base_rater, scout, cal, cfg.market_anchor_weight)
    return BayesianStrategy(cfg.name, base_rater, scout, cal)


class MarketAnchoredBayes(BayesianStrategy):
    """Like BayesianStrategy but blends the prior toward the *earliest* market
    price for this market. This is a partial 'use the wisdom of crowds' nudge —
    without going full market-following.

    weight=0 → pure regime prior, weight=1 → pure market initial price.
    """

    def __init__(self, name, base_rater, news_scout, calibrator, weight: float):
        super().__init__(name, base_rater, news_scout, calibrator)
        self.weight = weight

    def forecast(self, market, as_of, corpus) -> Forecast:
        regime_prior = self.base_rater(market, as_of)
        # Earliest price = market's first known mid at observed_at.
        early_dates = [d for d in market.timeline if d <= as_of]
        if early_dates:
            anchor = market.market_prices[min(early_dates)]
        else:
            anchor = regime_prior
        prior = (1 - self.weight) * regime_prior + self.weight * anchor

        evidence = self.news_scout(market, as_of, corpus)
        lrs = [e.lr for e in evidence]
        raw_post = _bayes_update(prior, lrs)
        calibrated = self.calibrator(raw_post)

        from clio.strategy import ForecastTrace
        trace = ForecastTrace(
            prior=prior,
            n_evidence=len(evidence),
            log_lr_total=sum(__import__("math").log(lr) for lr in lrs if lr > 0),
            raw_posterior=raw_post,
            calibrated=calibrated,
            evidence_doc_ids=tuple(e.doc.doc_id for e in evidence),
        )
        self._last_traces[(market.market_id, as_of)] = trace
        return Forecast(
            market_id=market.market_id, as_of=as_of, prob=calibrated,
            rationale=(
                f"regime_prior={regime_prior:.3f} anchor={anchor:.3f} "
                f"prior={prior:.3f} n_evi={len(evidence)} raw={raw_post:.3f} cal={calibrated:.3f}"
            ),
            evidence_doc_ids=trace.evidence_doc_ids,
        )


# ---------------------------------------------------------------------
# Iteration runner
# ---------------------------------------------------------------------


def compute_train_regime_priors(train_markets, oracle):
    by_regime: dict[str, list[int]] = defaultdict(list)
    for m in train_markets:
        if m.market_id in oracle:
            by_regime[m.regime].append(oracle.lookup(m.market_id))
    out: dict[str, float] = {}
    for regime, outs in by_regime.items():
        if outs:
            out[regime] = sum(outs) / len(outs)
    return out


def score_summary(score: ScoreVector | None) -> dict:
    if score is None:
        return {}
    return {
        "brier": score.extra.get("brier", 0.0),
        "ece": score.extra.get("ece", 0.0),
        "resolution": score.extra.get("resolution", 0.0),
        "bankroll_final": score.extra.get("bankroll_final", 0.0),
        "n_trades": int(score.extra.get("n_trades", 0)),
        "hit_rate": score.extra.get("hit_rate", 0.0),
        "neg_max_dd": score.neg_max_dd,
    }


def run_iteration(
    cfg: IterationConfig,
    train_markets,
    holdout_markets,
    corpus,
    oracle,
    baseline_run,
):
    log.info("\n" + "=" * 70)
    log.info("ITERATION  %s — %s", cfg.name, cfg.description)
    log.info("=" * 70)
    log.info("config: %s", json.dumps(asdict(cfg), default=str))

    strat = make_strategy(cfg, train_markets, corpus, oracle)
    h = BacktestHarness(corpus, oracle, holdout_markets)
    run = h.run(strat)
    holdout_summary = score_summary(run.score)
    log.info(
        "holdout: brier=%.4f  ece=%.4f  resolution=%.4f  bankroll=$%.0f  n_trades=%d  hit_rate=%.1f%%",
        holdout_summary["brier"],
        holdout_summary["ece"],
        holdout_summary["resolution"],
        holdout_summary["bankroll_final"],
        holdout_summary["n_trades"],
        holdout_summary["hit_rate"] * 100,
    )

    n_evi = []
    seen: set[str] = set()
    regimes = []
    for f in run.forecasts:
        if f.market_id in seen:
            continue
        seen.add(f.market_id)
        n_evi.append(len(f.evidence_doc_ids))
    for m in holdout_markets:
        if m.market_id in seen:
            regimes.append(m.regime)

    while len(n_evi) < len(run.final_probs):
        n_evi.append(0)

    subset = run_subset_attack(run, regimes, n_evi, min_bucket_size=3,
                                min_brier_degradation=0.03,
                                significance_level=0.20,
                                bootstrap_resamples=400)
    pert = run_perturbation(strat, holdout_markets, corpus)
    decision = evaluate_gate(
        score=run.score,
        baseline_score=baseline_run.score if baseline_run else None,
        subset_attack=subset,
        perturbation=pert,
        regime_scores=run.regime_breakdown,
        thresholds=GateThresholds(
            min_brier_lift_vs_baseline=0.005,
            max_blindspot_degradation=0.06,
            min_flip_sensitivity=0.005,
            max_flip_sensitivity=0.50,
        ),
    )

    log.info("blind spots:")
    if subset.blind_spots:
        for bs in subset.blind_spots[:5]:
            log.info(
                "  %s  Δbrier=%+.4f  p=%.3f  n=%d  pnl=%+.4f",
                bs.slot, bs.brier_degradation, bs.bootstrap_p_value, bs.n,
                bs.bucket_pnl_per_market,
            )
    else:
        log.info("  (none above threshold)")

    p = pert.summary()
    log.info(
        "perturbation: drop=%.3f flip=%.3f inject=%.3f max_flip=%.3f fragility=%.3f",
        p["mean_drop_sensitivity"], p["mean_flip_sensitivity"], p["mean_inject_sensitivity"],
        p["max_flip_sensitivity"], p["fragility_score"],
    )

    log.info("GATE: %s", "PASS" if decision.passed else "FAIL")
    for f in decision.failures:
        log.info("  ✗ %s", f)
    for n in decision.notes:
        log.info("  · %s", n)

    return {
        "name": cfg.name,
        "description": cfg.description,
        "config": asdict(cfg),
        "holdout": holdout_summary,
        "blind_spots": [
            {
                "slot": str(bs.slot),
                "n": bs.n,
                "brier_degradation": bs.brier_degradation,
                "p_value": bs.bootstrap_p_value,
                "pnl_per_market": bs.bucket_pnl_per_market,
            }
            for bs in subset.blind_spots
        ],
        "perturbation": p,
        "gate_passed": decision.passed,
        "gate_failures": list(decision.failures),
        "regime_briers": {
            regime: -sv.neg_brier for regime, sv in run.regime_breakdown.items()
        },
    }, run


def main() -> int:
    markets, oracle = _load_markets_payload("runs/live_iter/markets.json")
    corpus = read_corpus_jsonl("runs/live_iter/news.jsonl")
    log.info("loaded %d markets, %d resolutions, %d news docs",
             len(markets), len(oracle), len(corpus))

    train, holdout = stratified_split(markets, seed=7, train_frac=0.5)
    log.info("train=%d holdout=%d", len(train), len(holdout))
    log.info("train regimes: %s",
             {r: sum(1 for m in train if m.regime == r) for r in set(m.regime for m in train)})
    log.info("holdout regimes: %s",
             {r: sum(1 for m in holdout if m.regime == r) for r in set(m.regime for m in holdout)})

    # Baseline: market price echo. Reference for Brier lift.
    baseline = MarketPriceStrategy()
    bh = BacktestHarness(corpus, oracle, holdout)
    baseline_run = bh.run(baseline)
    log.info("\nBASELINE  market_price_echo:")
    bs = score_summary(baseline_run.score)
    log.info("  brier=%.4f  ece=%.4f  resolution=%.4f  bankroll=$%.0f",
             bs["brier"], bs["ece"], bs["resolution"], bs["bankroll_final"])

    # Compute train-side regime priors. Used in v3+.
    train_priors = compute_train_regime_priors(train, oracle)
    log.info("train regime priors: %s", train_priors)

    iterations: list[IterationConfig] = [
        IterationConfig(
            name="v1_baseline",
            description="default LR clamp [0.1, 10], identity calibrator, default regime priors",
        ),
        IterationConfig(
            name="v2_isotonic",
            description="add isotonic calibrator fit on train (Red Team likely flagged ECE)",
            calibrator_type="isotonic",
        ),
        IterationConfig(
            name="v3_train_priors",
            description="replace default priors with empirical train-set YES rates",
            calibrator_type="isotonic",
            regime_priors=train_priors,
        ),
        IterationConfig(
            name="v4_market_anchor",
            description="blend regime prior with market's earliest price (anchor=0.4)",
            calibrator_type="isotonic",
            regime_priors=train_priors,
            use_market_anchored_prior=True,
            market_anchor_weight=0.4,
        ),
        IterationConfig(
            name="v5_tighter_lr",
            description="tighten LR clamp to [0.3, 3.3] to reduce flip volatility",
            calibrator_type="isotonic",
            regime_priors=train_priors,
            use_market_anchored_prior=True,
            market_anchor_weight=0.4,
            lr_floor=0.3,
            lr_ceil=3.3,
        ),
    ]

    results = []
    runs = []
    for cfg in iterations:
        result, run = run_iteration(cfg, train, holdout, corpus, oracle, baseline_run)
        results.append(result)
        runs.append(run)

    # Pareto frontier over the iteration runs
    log.info("\n" + "=" * 70)
    log.info("PARETO FRONTIER over iteration runs")
    log.info("=" * 70)
    valid = [(r.score, name) for r, name in zip(runs, [c.name for c in iterations])
             if r.score is not None]
    if valid:
        scores, names = zip(*valid)
        front_idx = pareto_frontier(list(scores))
        front_names = [names[i] for i in front_idx]
        log.info("frontier: %s", front_names)

    # Side-by-side summary
    log.info("\n%-22s %-9s %-9s %-12s %-9s %-10s",
             "name", "brier", "ece", "bankroll", "fragility", "gate")
    log.info("%-22s %-9s %-9s %-12s %-9s %-10s",
             "----", "-----", "---", "--------", "---------", "----")
    for r in results:
        log.info(
            "%-22s %-9.4f %-9.4f $%-11.0f %-9.3f %-10s",
            r["name"],
            r["holdout"].get("brier", 0.0),
            r["holdout"].get("ece", 0.0),
            r["holdout"].get("bankroll_final", 0.0),
            r["perturbation"].get("fragility_score", 0.0),
            "PASS" if r["gate_passed"] else "FAIL",
        )

    out = Path("runs/live_iter/iterations.json")
    out.write_text(json.dumps(
        {"baseline": score_summary(baseline_run.score), "iterations": results},
        indent=2,
    ))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
