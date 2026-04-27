"""Adversarial validation — Red Team agent.

Three components:

- `subset_attack`: mine the holdout for feature buckets where the strategy
  systematically underperforms its overall metrics. Outputs ranked blind
  spots with statistical confidence.
- `perturbation`: counterfactually modify the input to a single forecast
  (drop evidence, flip stance, inject adversarial doc) and measure how
  the strategy's prediction moves. Detects strategies that are too anchored
  on prior, or too volatile to noise.
- `gate`: combines subset_attack and perturbation results into a binary
  promotion decision. A strategy that does not pass the gate may not move
  past T1 (paper-trade) per the constitution.
"""

from clio.red_team.subset_attack import (
    BlindSpot,
    SubsetAttackReport,
    run_subset_attack,
)
from clio.red_team.perturbation import (
    PerturbationReport,
    PerturbationResult,
    run_perturbation,
)
from clio.red_team.gate import GateDecision, GateThresholds, evaluate_gate

__all__ = [
    "BlindSpot",
    "SubsetAttackReport",
    "run_subset_attack",
    "PerturbationReport",
    "PerturbationResult",
    "run_perturbation",
    "GateDecision",
    "GateThresholds",
    "evaluate_gate",
]
