"""The Researcher Agent: Karpathy's autoresearch loop, applied to live paper trading.

Each week, this agent reads the past N days of resolved trades + cumulative
PnL, identifies the single most actionable parameter change with statistical
evidence, and submits it as a draft PR for human review.

The agent is conservative by default: it has a hard "insufficient data"
threshold (≥20 resolved trades) and prefers `hold` to `propose` when evidence
is weak.
"""

from clio.research.agent.metrics import LiveMetrics, compute_live_metrics
from clio.research.agent.proposal import (
    Proposal,
    ALLOWED_PARAMS,
    parse_proposal,
    validate_proposal,
)

__all__ = [
    "LiveMetrics",
    "compute_live_metrics",
    "Proposal",
    "ALLOWED_PARAMS",
    "parse_proposal",
    "validate_proposal",
]
