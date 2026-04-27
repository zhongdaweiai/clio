"""Strategy pipeline tests using the mock LLM and synthetic world."""

from datetime import date

from clio.agents.base import MockLLMClient
from clio.agents.base_rater import BaseRater
from clio.agents.calibrator import IdentityCalibrator
from clio.agents.news_scout import NewsScout
from clio.data.synthetic import SyntheticConfig, generate_synthetic_world
from clio.frozen.harness import BacktestHarness
from clio.strategy import BayesianStrategy, MarketPriceStrategy


def _llm():
    llm = MockLLMClient()
    llm.register(r"supports", "0.75")
    llm.register(r"contradicts", "0.30")
    llm.set_default("0.55")
    return llm


def test_strategy_produces_valid_probabilities():
    world = generate_synthetic_world(SyntheticConfig(n_markets=10, seed=1))
    llm = _llm()
    strat = BayesianStrategy("test", BaseRater(llm), NewsScout(llm), IdentityCalibrator())
    h = BacktestHarness(world.corpus, world.oracle, world.markets)
    run = h.run(strat)
    for p in run.final_probs:
        assert 0.0 <= p <= 1.0


def test_strategy_makes_one_final_forecast_per_market():
    world = generate_synthetic_world(SyntheticConfig(n_markets=8, seed=2))
    llm = _llm()
    strat = BayesianStrategy("test", BaseRater(llm), NewsScout(llm))
    h = BacktestHarness(world.corpus, world.oracle, world.markets)
    run = h.run(strat)
    assert len(run.final_probs) == len(world.markets)
    assert len(run.final_outcomes) == len(world.markets)


def test_market_baseline_returns_market_price():
    world = generate_synthetic_world(SyntheticConfig(n_markets=5, seed=3))
    h = BacktestHarness(world.corpus, world.oracle, world.markets)
    run = h.run(MarketPriceStrategy())
    assert run.score is not None


def test_strategy_trace_records_evidence():
    world = generate_synthetic_world(SyntheticConfig(n_markets=3, seed=4))
    llm = _llm()
    strat = BayesianStrategy("trace_test", BaseRater(llm), NewsScout(llm))
    h = BacktestHarness(world.corpus, world.oracle, world.markets)
    h.run(strat)
    m = world.markets[0]
    a = m.timeline[-2] if len(m.timeline) >= 2 else m.timeline[0]
    trace = strat.trace_for(m.market_id, a)
    assert trace is not None
    assert 0.0 <= trace.calibrated <= 1.0


def test_strategy_does_not_leak_future_evidence():
    """If the strategy ever sees a doc with published_at >= as_of, the harness
    or the news scout's own assert_no_leak guard will raise."""
    world = generate_synthetic_world(SyntheticConfig(n_markets=15, seed=5))
    llm = _llm()
    strat = BayesianStrategy("leak_check", BaseRater(llm), NewsScout(llm))
    h = BacktestHarness(world.corpus, world.oracle, world.markets)
    # No exception means cutoff was respected.
    h.run(strat)
