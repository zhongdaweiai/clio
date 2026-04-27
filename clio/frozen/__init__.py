"""Frozen evaluation layer. Immutable to agents.

The frozen layer is the ground truth. Agents must not modify anything in this
package. Every component here exists to make agent claims falsifiable.
"""

from clio.frozen.corpus import Corpus, Document, CutoffViolation
from clio.frozen.cost_model import CostModel
from clio.frozen.harness import BacktestHarness, BacktestRun
from clio.frozen.oracle import ResolutionOracle
from clio.frozen.scoring import (
    brier_score,
    calibration_ece,
    resolution_decomposition,
    kelly_pnl,
    score_vector,
    ScoreVector,
)

__all__ = [
    "Corpus",
    "Document",
    "CutoffViolation",
    "CostModel",
    "BacktestHarness",
    "BacktestRun",
    "ResolutionOracle",
    "brier_score",
    "calibration_ece",
    "resolution_decomposition",
    "kelly_pnl",
    "score_vector",
    "ScoreVector",
]
