"""Promotion gate.

Combines subset_attack + perturbation results into a single PASS/FAIL decision
per the constitution's red lines (M5/T1 transition).

A strategy passes the gate iff:
  1. Holdout Brier improvement over baseline >= MIN_LIFT
  2. Subset attack: max blind-spot Brier degradation <= MAX_DEGRADATION,
     OR the strategy has registered an explicit `excluded_slots` set that
     covers every blind spot above the cap.
  3. Perturbation: mean flip sensitivity in [MIN_FLIP_DELTA, MAX_FLIP_DELTA]
     AND max flip sensitivity <= MAX_SINGLE_DOC_INFLUENCE.
  4. Per-regime ECE <= MAX_REGIME_ECE for every regime claimed.

Failures are accumulated; the report tells you everything that's broken,
not just the first thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from clio.frozen.scoring import ScoreVector
from clio.red_team.perturbation import PerturbationReport
from clio.red_team.subset_attack import BlindSpot, SubsetAttackReport


@dataclass(frozen=True)
class GateThresholds:
    """All thresholds in one place. Tune in the constitution, not in code."""

    min_brier_lift_vs_baseline: float = 0.005
    max_blindspot_degradation: float = 0.05
    max_regime_ece: float = 0.07
    min_flip_sensitivity: float = 0.02
    max_flip_sensitivity: float = 0.50
    max_single_doc_influence: float = 0.70


@dataclass
class GateDecision:
    passed: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    uncovered_blindspots: list[BlindSpot] = field(default_factory=list)


def evaluate_gate(
    score: ScoreVector,
    baseline_score: ScoreVector | None,
    subset_attack: SubsetAttackReport,
    perturbation: PerturbationReport,
    regime_scores: dict[str, ScoreVector] | None = None,
    excluded_slots: set[str] | None = None,
    thresholds: GateThresholds | None = None,
) -> GateDecision:
    t = thresholds or GateThresholds()
    excluded = excluded_slots or set()
    decision = GateDecision(passed=True)

    # 1. Brier lift vs baseline
    strategy_brier = -score.neg_brier
    if baseline_score is not None:
        baseline_brier = -baseline_score.neg_brier
        lift = baseline_brier - strategy_brier
        if lift < t.min_brier_lift_vs_baseline:
            decision.passed = False
            decision.failures.append(
                f"Brier lift over baseline {lift:+.4f} < required {t.min_brier_lift_vs_baseline:+.4f}"
            )
        else:
            decision.notes.append(f"Brier lift over baseline: {lift:+.4f}")

    # 2. Subset attack: each blind spot must either be below the cap, or
    #    explicitly excluded.
    uncovered: list[BlindSpot] = []
    for bs in subset_attack.blind_spots:
        if bs.brier_degradation <= t.max_blindspot_degradation:
            continue
        slot_str = str(bs.slot)
        if slot_str in excluded:
            decision.notes.append(f"blindspot {slot_str} excluded by strategy")
            continue
        uncovered.append(bs)
    if uncovered:
        decision.passed = False
        decision.uncovered_blindspots = uncovered
        slot_names = [str(b.slot) for b in uncovered]
        decision.failures.append(
            f"uncovered blindspots: {slot_names} "
            f"(max degradation {max(b.brier_degradation for b in uncovered):.4f})"
        )

    # 3. Perturbation sensitivity
    pert = perturbation.summary()
    flip = pert["mean_flip_sensitivity"]
    if perturbation.n_markets > 0:
        if flip < t.min_flip_sensitivity:
            decision.passed = False
            decision.failures.append(
                f"strategy too rigid: mean flip sensitivity {flip:.3f} < {t.min_flip_sensitivity:.3f} "
                "— evidence is barely affecting predictions"
            )
        if flip > t.max_flip_sensitivity:
            decision.passed = False
            decision.failures.append(
                f"strategy too volatile: mean flip sensitivity {flip:.3f} > {t.max_flip_sensitivity:.3f} "
                "— a single doc flip swings predictions"
            )
        if pert["max_flip_sensitivity"] > t.max_single_doc_influence:
            decision.passed = False
            decision.failures.append(
                f"single-doc influence too high: max flip sensitivity "
                f"{pert['max_flip_sensitivity']:.3f} > {t.max_single_doc_influence:.3f}"
            )

    # 4. Per-regime ECE caps
    if regime_scores:
        for regime, sv in regime_scores.items():
            ece = -sv.neg_ece
            if ece > t.max_regime_ece:
                decision.passed = False
                decision.failures.append(
                    f"regime {regime!r} ECE {ece:.3f} > cap {t.max_regime_ece:.3f}"
                )

    return decision
