"""Pareto frontier computation over ScoreVectors.

Given N strategies with their score vectors, return the index set that is
*not strictly dominated* by any other. This replaces "pick the best" with
"pick the non-dominated set" — see Constitution M5.
"""

from __future__ import annotations

from typing import Sequence

from clio.frozen.scoring import ScoreVector


def pareto_frontier(scores: Sequence[ScoreVector]) -> list[int]:
    """Return indices of strategies on the Pareto frontier.

    Worst case O(n^2). Fine for n in the low hundreds.
    """
    keep: list[int] = []
    for i, si in enumerate(scores):
        dominated = False
        for j, sj in enumerate(scores):
            if i == j:
                continue
            if sj.dominates(si):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def select_for_regime(
    regime_scores: Sequence[dict[str, ScoreVector]],
    regime: str,
) -> list[int]:
    """Pareto frontier within a single regime.

    Strategies that don't cover this regime are excluded. This is how
    regime-conditional routing is implemented: each regime has its own
    Pareto front, and the live router picks accordingly.
    """
    indexed = [
        (i, rs[regime])
        for i, rs in enumerate(regime_scores)
        if regime in rs
    ]
    if not indexed:
        return []
    idxs, scores = zip(*indexed)
    front = pareto_frontier(list(scores))
    return [idxs[k] for k in front]
