"""Memory layer tests."""

from datetime import date

from clio.memory.failures import FailureClusterer, LABELS
from clio.memory.traces import Trace, TraceStore


def _trace(mid: str, fc: float, mp: float = 0.5, regime: str = "election") -> Trace:
    return Trace(
        strategy_name="test",
        market_id=mid,
        regime=regime,
        as_of=date(2024, 6, 1),
        forecast=fc,
        market_price=mp,
        rationale="",
        evidence_doc_ids=("d1",),
        extra={},
    )


def test_trace_store_roundtrip():
    s = TraceStore(":memory:")
    s.write(_trace("M1", 0.7))
    s.write(_trace("M2", 0.3))
    rows = s.read_by_strategy("test")
    assert len(rows) == 2
    assert {r.market_id for r in rows} == {"M1", "M2"}


def test_failure_label_over_confident_yes():
    label = FailureClusterer.label_one(forecast=0.95, market_price=0.6, outcome=0)
    assert label is LABELS["over_confident_yes"]


def test_failure_label_over_confident_no():
    label = FailureClusterer.label_one(forecast=0.05, market_price=0.4, outcome=1)
    assert label is LABELS["over_confident_no"]


def test_failure_label_under_reactive():
    # Forecast very close to market, both very far from outcome.
    label = FailureClusterer.label_one(forecast=0.50, market_price=0.50, outcome=1)
    # |0.5 - 1| = 0.5 > 0.4 → labeled
    assert label is LABELS["under_reactive"]


def test_failure_label_over_reactive():
    label = FailureClusterer.label_one(forecast=0.85, market_price=0.40, outcome=0)
    # err = 0.85, distance from market = 0.45 > 0.30, but forecast is 0.85 so
    # it would also match "over_confident_yes". Verify the specific over_reactive
    # case where forecast is not in the extreme bands.
    label2 = FailureClusterer.label_one(forecast=0.75, market_price=0.30, outcome=0)
    assert label2 is LABELS["over_reactive"]


def test_failure_label_returns_none_for_small_error():
    # err = |0.65 - 1| = 0.35, below the 0.4 threshold → no failure label
    label = FailureClusterer.label_one(forecast=0.65, market_price=0.5, outcome=1)
    assert label is None


def test_cluster_groups_and_orders_by_count():
    traces = [
        _trace("M1", 0.95),  # over_confident_yes
        _trace("M2", 0.95),  # over_confident_yes
        _trace("M3", 0.05),  # over_confident_no
    ]
    outcomes = {"M1": 0, "M2": 0, "M3": 1}
    clusters = FailureClusterer.cluster(traces, outcomes)
    assert clusters[0].label.name == "over_confident_yes"
    assert clusters[0].count == 2
    assert clusters[1].label.name == "over_confident_no"
    assert clusters[1].count == 1


def test_suspect_summary_sums_per_agent():
    traces = [
        _trace("M1", 0.95),  # over_confident_yes → calibrator
        _trace("M2", 0.05),  # over_confident_no  → calibrator
        _trace("M3", 0.50, mp=0.50),  # under_reactive → news_scout
    ]
    outcomes = {"M1": 0, "M2": 1, "M3": 1}
    clusters = FailureClusterer.cluster(traces, outcomes)
    summary = FailureClusterer.suspect_summary(clusters)
    assert summary["calibrator"] == 2
    assert summary["news_scout"] == 1
