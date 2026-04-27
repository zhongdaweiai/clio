"""Promotion gate tests."""

from clio.frozen.scoring import ScoreVector
from clio.red_team.gate import GateThresholds, evaluate_gate
from clio.red_team.perturbation import PerturbationReport, PerturbationResult
from clio.red_team.subset_attack import (
    BlindSpot,
    FeatureSlot,
    SubsetAttackReport,
)


def _good_score(brier: float = 0.10) -> ScoreVector:
    return ScoreVector(
        neg_brier=-brier, neg_ece=-0.04, resolution=0.15,
        total_return=0.10, sharpe=1.2, neg_max_dd=-0.05,
        coverage=1.0, regime_breadth=4,
    )


def _empty_subset_report(brier: float = 0.10) -> SubsetAttackReport:
    return SubsetAttackReport(
        overall_brier=brier, overall_pnl_per_market=0.01, n_holdout=50,
        blind_spots=[],
    )


def _good_perturbation() -> PerturbationReport:
    rep = PerturbationReport(n_markets=10)
    rep.results = [
        PerturbationResult(
            market_id=f"M{i}",
            base_prob=0.5 + 0.1 * (i % 3),
            drop_strongest_prob=0.5,
            flip_strongest_prob=0.5 + (-0.15 if i % 2 == 0 else 0.10),
            inject_adversary_prob=0.45,
            n_evidence=3,
            strongest_lr=2.0,
        )
        for i in range(10)
    ]
    return rep


def test_gate_passes_on_clean_results():
    decision = evaluate_gate(
        score=_good_score(0.10),
        baseline_score=_good_score(0.18),  # baseline is worse → strategy has lift
        subset_attack=_empty_subset_report(),
        perturbation=_good_perturbation(),
        regime_scores={"election": _good_score(), "geo": _good_score()},
    )
    assert decision.passed
    assert not decision.failures


def test_gate_fails_when_no_brier_lift():
    decision = evaluate_gate(
        score=_good_score(0.18),
        baseline_score=_good_score(0.18),  # equal → no lift
        subset_attack=_empty_subset_report(),
        perturbation=_good_perturbation(),
    )
    assert not decision.passed
    assert any("Brier lift" in f for f in decision.failures)


def test_gate_fails_on_uncovered_blindspot():
    bs = BlindSpot(
        slot=FeatureSlot("regime", "geo"),
        n=20, bucket_brier=0.30, overall_brier=0.10,
        brier_degradation=0.20, bucket_pnl_per_market=-0.05,
        overall_pnl_per_market=0.02, bootstrap_p_value=0.01,
    )
    report = SubsetAttackReport(
        overall_brier=0.10, overall_pnl_per_market=0.02,
        n_holdout=50, blind_spots=[bs],
    )
    decision = evaluate_gate(
        score=_good_score(0.10),
        baseline_score=_good_score(0.18),
        subset_attack=report,
        perturbation=_good_perturbation(),
    )
    assert not decision.passed
    assert decision.uncovered_blindspots == [bs]


def test_gate_passes_when_blindspot_explicitly_excluded():
    bs = BlindSpot(
        slot=FeatureSlot("regime", "geo"),
        n=20, bucket_brier=0.30, overall_brier=0.10,
        brier_degradation=0.20, bucket_pnl_per_market=-0.05,
        overall_pnl_per_market=0.02, bootstrap_p_value=0.01,
    )
    report = SubsetAttackReport(
        overall_brier=0.10, overall_pnl_per_market=0.02,
        n_holdout=50, blind_spots=[bs],
    )
    decision = evaluate_gate(
        score=_good_score(0.10),
        baseline_score=_good_score(0.18),
        subset_attack=report,
        perturbation=_good_perturbation(),
        excluded_slots={"regime=geo"},
    )
    assert decision.passed


def test_gate_fails_on_too_rigid_strategy():
    # All flips produce zero change.
    rigid = PerturbationReport(n_markets=5)
    rigid.results = [
        PerturbationResult(
            market_id=f"M{i}",
            base_prob=0.5,
            drop_strongest_prob=0.5,
            flip_strongest_prob=0.5,  # didn't move
            inject_adversary_prob=0.5,
            n_evidence=3,
            strongest_lr=1.0,
        )
        for i in range(5)
    ]
    decision = evaluate_gate(
        score=_good_score(0.10),
        baseline_score=_good_score(0.18),
        subset_attack=_empty_subset_report(),
        perturbation=rigid,
    )
    assert not decision.passed
    assert any("rigid" in f for f in decision.failures)


def test_gate_fails_on_too_volatile_strategy():
    volatile = PerturbationReport(n_markets=5)
    volatile.results = [
        PerturbationResult(
            market_id=f"M{i}",
            base_prob=0.10,
            drop_strongest_prob=0.50,
            flip_strongest_prob=0.95,  # huge swing
            inject_adversary_prob=0.05,
            n_evidence=3,
            strongest_lr=10.0,
        )
        for i in range(5)
    ]
    decision = evaluate_gate(
        score=_good_score(0.10),
        baseline_score=_good_score(0.18),
        subset_attack=_empty_subset_report(),
        perturbation=volatile,
    )
    assert not decision.passed
    assert any("volatile" in f or "single-doc influence" in f for f in decision.failures)


def test_gate_fails_on_high_regime_ece():
    bad_regime = ScoreVector(
        neg_brier=-0.10, neg_ece=-0.20,  # ECE 0.20 > 0.07 cap
        resolution=0.15, total_return=0.10, sharpe=1.0, neg_max_dd=-0.05,
        coverage=1.0, regime_breadth=1,
    )
    decision = evaluate_gate(
        score=_good_score(0.10),
        baseline_score=_good_score(0.18),
        subset_attack=_empty_subset_report(),
        perturbation=_good_perturbation(),
        regime_scores={"geo": bad_regime},
    )
    assert not decision.passed
    assert any("ECE" in f for f in decision.failures)


def test_gate_thresholds_can_be_overridden():
    decision = evaluate_gate(
        score=_good_score(0.18),
        baseline_score=_good_score(0.18),
        subset_attack=_empty_subset_report(),
        perturbation=_good_perturbation(),
        thresholds=GateThresholds(min_brier_lift_vs_baseline=-1.0),  # very lenient
    )
    assert decision.passed
