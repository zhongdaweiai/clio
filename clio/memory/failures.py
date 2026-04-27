"""Failure-mode taxonomy.

For the MVP we use rule-based labels rather than embedding clustering. The
labels are coarse but actionable: each maps to a specific micro-agent
improvement.

Labels:
- `over_confident_yes`: predicted >= 0.85, actual = 0
- `over_confident_no`:  predicted <= 0.15, actual = 1
- `under_reactive`: |forecast - market_price| < 0.05 AND |forecast - outcome| > 0.4
                    (we agreed with the market and the market was very wrong)
- `over_reactive`:  |forecast - market_price| > 0.30 AND |forecast - outcome| > 0.4
                    (we strongly disagreed with the market and we were wrong)
- `regime_mismatch`: forecast significantly off, no other label fits — likely
                     base-rate mismatch.

Each label points to which agent to suspect: see `FailureLabel.suspect_agent`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from clio.memory.traces import Trace


@dataclass(frozen=True)
class FailureLabel:
    name: str
    suspect_agent: str
    description: str


LABELS = {
    "over_confident_yes": FailureLabel(
        "over_confident_yes",
        "calibrator",
        "Predicted >=0.85 but resolved NO. Calibrator is too sharp at high end.",
    ),
    "over_confident_no": FailureLabel(
        "over_confident_no",
        "calibrator",
        "Predicted <=0.15 but resolved YES. Calibrator is too sharp at low end.",
    ),
    "under_reactive": FailureLabel(
        "under_reactive",
        "news_scout",
        "Agreed with market, both were very wrong. News Scout is missing signal.",
    ),
    "over_reactive": FailureLabel(
        "over_reactive",
        "news_scout",
        "Strongly disagreed with market and we were wrong. News Scout LRs are too aggressive.",
    ),
    "regime_mismatch": FailureLabel(
        "regime_mismatch",
        "base_rater",
        "Forecast far from outcome with no other obvious cause. Base rate likely wrong for this regime.",
    ),
}


@dataclass(frozen=True)
class FailureCluster:
    label: FailureLabel
    count: int
    examples: tuple[str, ...]  # market_ids


class FailureClusterer:
    @staticmethod
    def label_one(
        forecast: float,
        market_price: float,
        outcome: int,
    ) -> FailureLabel | None:
        err = abs(forecast - outcome)
        if err < 0.4:
            return None
        if forecast >= 0.85 and outcome == 0:
            return LABELS["over_confident_yes"]
        if forecast <= 0.15 and outcome == 1:
            return LABELS["over_confident_no"]
        if abs(forecast - market_price) < 0.05:
            return LABELS["under_reactive"]
        if abs(forecast - market_price) > 0.30:
            return LABELS["over_reactive"]
        return LABELS["regime_mismatch"]

    @staticmethod
    def cluster(
        traces: Sequence[Trace],
        outcomes: dict[str, int],
        max_examples: int = 3,
    ) -> list[FailureCluster]:
        """Return clusters sorted by frequency desc."""
        buckets: dict[str, list[str]] = defaultdict(list)
        for t in traces:
            outcome = outcomes.get(t.market_id)
            if outcome is None:
                continue
            label = FailureClusterer.label_one(t.forecast, t.market_price, outcome)
            if label is None:
                continue
            buckets[label.name].append(t.market_id)

        out = []
        for name, mids in buckets.items():
            out.append(
                FailureCluster(
                    label=LABELS[name],
                    count=len(mids),
                    examples=tuple(mids[:max_examples]),
                )
            )
        out.sort(key=lambda c: -c.count)
        return out

    @staticmethod
    def suspect_summary(clusters: Sequence[FailureCluster]) -> dict[str, int]:
        ctr: Counter[str] = Counter()
        for c in clusters:
            ctr[c.label.suspect_agent] += c.count
        return dict(ctr)
