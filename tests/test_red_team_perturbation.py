"""Perturbation test suite — verifies we can detect strategies that are
either too rigid (don't react to flipping evidence) or too volatile (one
flip swings everything)."""

from clio.agents.base import MockLLMClient
from clio.agents.base_rater import BaseRater
from clio.agents.calibrator import IdentityCalibrator
from clio.agents.news_scout import NewsScout
from clio.data.synthetic import SyntheticConfig, generate_synthetic_world
from clio.red_team.perturbation import run_perturbation
from clio.strategy import BayesianStrategy


def _signal_aware_llm():
    llm = MockLLMClient()
    llm.register(r"document title:.*supports", "0.78")
    llm.register(r"document title:.*contradicts", "0.22")
    llm.register(r"document content:.*supports", "0.78")
    llm.register(r"document content:.*contradicts", "0.22")
    llm.set_default("0.50")
    return llm


def test_perturbation_runs_and_produces_results():
    world = generate_synthetic_world(SyntheticConfig(n_markets=20, seed=10))
    llm = _signal_aware_llm()
    strat = BayesianStrategy("test", BaseRater(llm), NewsScout(llm), IdentityCalibrator())
    rep = run_perturbation(strat, world.markets, world.corpus)
    assert rep.n_markets > 0
    for r in rep.results:
        assert 0.0 <= r.base_prob <= 1.0
        assert 0.0 <= r.flip_strongest_prob <= 1.0


def test_zero_information_scout_has_zero_flip_sensitivity():
    """An LR range of [1.0, 1.0] means evidence carries no information — every
    flip is a no-op. The fragility score must be (numerically) zero."""
    world = generate_synthetic_world(SyntheticConfig(n_markets=10, seed=11))
    llm = _signal_aware_llm()
    no_info_scout = NewsScout(llm, lr_floor=1.0, lr_ceil=1.0)
    strat = BayesianStrategy("no_info", BaseRater(llm), no_info_scout, IdentityCalibrator())
    rep = run_perturbation(strat, world.markets, world.corpus)
    assert rep.fragility_score < 1e-9


def test_real_scout_has_nonzero_flip_sensitivity():
    """The standard scout reacts to evidence; flip should produce *some* change."""
    world = generate_synthetic_world(SyntheticConfig(n_markets=10, seed=11))
    llm = _signal_aware_llm()
    strat = BayesianStrategy("real", BaseRater(llm), NewsScout(llm), IdentityCalibrator())
    rep = run_perturbation(strat, world.markets, world.corpus)
    # At least one market must show non-zero flip impact.
    assert any(r.flip_delta > 1e-6 for r in rep.results)


def test_perturbation_summary_keys():
    world = generate_synthetic_world(SyntheticConfig(n_markets=8, seed=12))
    llm = _signal_aware_llm()
    strat = BayesianStrategy("test", BaseRater(llm), NewsScout(llm), IdentityCalibrator())
    rep = run_perturbation(strat, world.markets, world.corpus)
    summary = rep.summary()
    for k in (
        "mean_drop_sensitivity",
        "mean_flip_sensitivity",
        "mean_inject_sensitivity",
        "max_flip_sensitivity",
        "fragility_score",
    ):
        assert k in summary


def test_perturbation_handles_no_evidence_gracefully():
    """If no evidence is found, all four predictions should be equal."""
    from clio.frozen.corpus import Corpus

    world = generate_synthetic_world(SyntheticConfig(n_markets=5, seed=13))
    empty_corpus = Corpus()  # no docs
    llm = _signal_aware_llm()
    strat = BayesianStrategy("test", BaseRater(llm), NewsScout(llm), IdentityCalibrator())
    rep = run_perturbation(strat, world.markets, empty_corpus)
    for r in rep.results:
        assert r.n_evidence == 0
        assert r.drop_delta == 0.0
        assert r.flip_delta == 0.0
        assert r.inject_delta == 0.0
