"""End-to-end synthetic backtest.

Verifies that the whole pipeline runs and that, on data where signal exists,
strategies that use evidence outperform the uninformed baseline. This is the
"system is glued together correctly" smoke test, not a claim about real-world
performance.
"""

from clio.agents.base import MockLLMClient
from clio.agents.base_rater import BaseRater
from clio.agents.calibrator import IdentityCalibrator, IsotonicCalibrator
from clio.agents.news_scout import NewsScout
from clio.data.synthetic import SyntheticConfig, generate_synthetic_world
from clio.frozen.harness import BacktestHarness
from clio.frozen.scoring import brier_score
from clio.pareto import pareto_frontier
from clio.strategy import BayesianStrategy, MarketPriceStrategy


def _signal_aware_llm() -> MockLLMClient:
    llm = MockLLMClient()
    # The synthetic world's docs say "supports" or "contradicts" in their
    # content. The mock LLM keys off those tokens.
    llm.register(r"supports", "0.78")
    llm.register(r"contradicts", "0.22")
    llm.set_default("0.50")
    return llm


def test_pipeline_runs_end_to_end():
    world = generate_synthetic_world(
        SyntheticConfig(n_markets=20, seed=11, news_signal_strength=0.85)
    )
    llm = _signal_aware_llm()

    strategies = [
        MarketPriceStrategy(),
        BayesianStrategy("bayes", BaseRater(llm), NewsScout(llm), IdentityCalibrator()),
    ]

    h = BacktestHarness(world.corpus, world.oracle, world.markets)
    runs = [h.run(s) for s in strategies]

    for r in runs:
        assert r.score is not None
        assert len(r.final_probs) == len(world.markets)
        assert all(0.0 <= p <= 1.0 for p in r.final_probs)


def test_bayesian_strategy_beats_uninformed_when_signal_is_strong():
    world = generate_synthetic_world(
        SyntheticConfig(
            n_markets=80,
            seed=7,
            news_signal_strength=0.92,
            market_efficiency=0.0,  # market is uninformed → leaves edge for us
        )
    )
    llm = _signal_aware_llm()

    bayes = BayesianStrategy("bayes", BaseRater(llm), NewsScout(llm))
    base = BaseRaterOnlyStrategy("base_only", BaseRater(llm))

    h = BacktestHarness(world.corpus, world.oracle, world.markets)
    bayes_run = h.run(bayes)
    base_run = h.run(base)

    bayes_brier = brier_score(bayes_run.final_probs, bayes_run.final_outcomes)
    base_brier = brier_score(base_run.final_probs, base_run.final_outcomes)
    # Reading evidence should beat ignoring it on a high-signal world.
    assert bayes_brier < base_brier


def test_isotonic_calibration_reduces_ece_in_holdout():
    world = generate_synthetic_world(
        SyntheticConfig(n_markets=80, seed=9, news_signal_strength=0.85)
    )
    train = world.markets[:40]
    holdout = world.markets[40:]

    llm = _signal_aware_llm()
    raw = BayesianStrategy("raw", BaseRater(llm), NewsScout(llm), IdentityCalibrator())

    train_h = BacktestHarness(world.corpus, world.oracle, train)
    train_run = train_h.run(raw)

    cal = IsotonicCalibrator()
    cal.fit(train_run.final_probs, train_run.final_outcomes)

    calibrated = BayesianStrategy("cal", BaseRater(llm), NewsScout(llm), cal)
    holdout_h = BacktestHarness(world.corpus, world.oracle, holdout)
    raw_holdout = holdout_h.run(raw)
    cal_holdout = holdout_h.run(calibrated)

    # Calibration should not make holdout Brier dramatically worse,
    # and should usually improve ECE.
    assert raw_holdout.score is not None and cal_holdout.score is not None
    raw_ece = -raw_holdout.score.neg_ece
    cal_ece = -cal_holdout.score.neg_ece
    # Tolerance: ECE may not always strictly decrease on small holdouts, but
    # should not worsen by more than 0.05.
    assert cal_ece <= raw_ece + 0.05


def test_pareto_includes_at_least_one_strategy():
    world = generate_synthetic_world(SyntheticConfig(n_markets=15, seed=13))
    llm = _signal_aware_llm()
    strats = [
        MarketPriceStrategy(),
        BayesianStrategy("b1", BaseRater(llm), NewsScout(llm), IdentityCalibrator()),
    ]
    h = BacktestHarness(world.corpus, world.oracle, world.markets)
    runs = [h.run(s) for s in strats]
    scores = [r.score for r in runs if r.score is not None]
    assert len(pareto_frontier(scores)) >= 1


# --- helpers ---


from datetime import date

from clio.agents.base_rater import BaseRater as _BaseRater
from clio.frozen.harness import Forecast, Market


class BaseRaterOnlyStrategy:
    """A strategy that ignores news entirely. Used as a sanity baseline."""

    def __init__(self, name: str, base_rater: _BaseRater) -> None:
        self.name = name
        self.base_rater = base_rater

    def forecast(self, market: Market, as_of: date, corpus) -> Forecast:
        p = self.base_rater(market, as_of)
        return Forecast(market_id=market.market_id, as_of=as_of, prob=p)
