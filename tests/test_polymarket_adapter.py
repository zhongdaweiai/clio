"""Polymarket adapter tests against the recorded fixture.

These tests exercise everything except the live HTTP layer:
- parsing of the gamma response shape
- regime classification
- timeline construction
- price history lookup with as-of-or-before semantics
- resolution inference from outcomePrices
"""

from datetime import date
from pathlib import Path

import pytest

from clio.data.polymarket_adapter import (
    PolymarketAdapter,
    adapter_from_recorded_dump,
    classify_regime,
)


FIXTURE = Path(__file__).parent / "fixtures" / "polymarket_dump.json"


def test_classify_regime_election():
    assert classify_regime("Will Trump win the 2024 Iowa primary?") == "election"


def test_classify_regime_financial():
    assert classify_regime("Will the Fed raise rates in March?") == "financial"


def test_classify_regime_sport():
    assert classify_regime("Will the Chiefs win the Super Bowl?") == "sport"


def test_classify_regime_scientific():
    assert classify_regime("Will SpaceX launch Starship before April?") == "scientific"


def test_classify_regime_falls_through_to_other():
    assert classify_regime("Will Taylor Swift release a new album?") == "other"


def test_adapter_from_dump_loads_markets(tmp_path):
    adapter = adapter_from_recorded_dump(FIXTURE)
    markets = adapter.load_markets(date(2023, 1, 1), date(2025, 1, 1), max_markets=10)
    assert len(markets) == 4

    by_id = {m.market_id: m for m in markets}
    assert "100001" in by_id
    fed = by_id["100001"]
    assert fed.regime == "financial"
    assert fed.observed_at == date(2024, 1, 15)
    assert fed.closes_at == date(2024, 3, 20)
    # Timeline must be 4 dates within the market window.
    assert len(fed.timeline) == 4
    for d in fed.timeline:
        assert fed.observed_at <= d < fed.closes_at
    # Prices must be present at every timeline point.
    for d in fed.timeline:
        assert 0.0 <= fed.market_prices[d] <= 1.0


def test_adapter_window_filter():
    adapter = adapter_from_recorded_dump(FIXTURE)
    markets = adapter.load_markets(date(2024, 2, 1), date(2024, 3, 31), max_markets=10)
    market_ids = {m.market_id for m in markets}
    # Tesla closes 2024-01-25 (out), Trump closes 2024-01-16 (out),
    # Fed closes 2024-03-20 (in), Starship closes 2024-04-01 (out).
    assert market_ids == {"100001"}


def test_adapter_resolutions_from_dump():
    adapter = adapter_from_recorded_dump(FIXTURE)
    markets = adapter.load_markets(date(2023, 1, 1), date(2025, 1, 1), max_markets=10)
    oracle = adapter.load_resolutions(markets)
    # Fed (100001): YES wins → 1
    # Tesla (100002): NO wins → 0
    # Trump (100003): YES wins → 1
    # Starship (100004): NO wins → 0
    assert oracle.lookup("100001") == 1
    assert oracle.lookup("100002") == 0
    assert oracle.lookup("100003") == 1
    assert oracle.lookup("100004") == 0


def test_adapter_max_markets_respected():
    adapter = adapter_from_recorded_dump(FIXTURE)
    markets = adapter.load_markets(
        date(2023, 1, 1), date(2025, 1, 1), max_markets=2
    )
    assert len(markets) == 2


def test_adapter_only_binary_filter():
    adapter = adapter_from_recorded_dump(FIXTURE)
    # All fixtures are binary — should keep them all.
    markets = adapter.load_markets(
        date(2023, 1, 1), date(2025, 1, 1), only_binary=True
    )
    assert len(markets) == 4
