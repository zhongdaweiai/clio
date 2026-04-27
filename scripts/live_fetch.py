"""Live-fetch a curated, regime-diverse set of resolved Polymarket markets.

Strategy:
1. Fetch high-volume closed markets in 2024–2025 (via Polymarket gamma).
2. Bucket by regime, take a diverse cut.
3. For each, build the Market with full price-history timeline.

We sort by volume because thin markets have sparse price history (day-of-
event markets often have <3 trading days), which we can't backtest at the
timeline granularity we need.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

from clio.data.polymarket_adapter import (
    PolymarketAdapter,
    PolymarketRawMarket,
    classify_regime,
)
from clio.frozen.harness import Market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("live_fetch")

_HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}


def _fetch_high_volume_markets(since: date, until: date, max_markets: int) -> list[dict]:
    """Bypass the standard adapter pagination — fetch by volume directly."""
    out: list[dict] = []
    offset = 0
    page = 100
    while len(out) < max_markets:
        params = {
            "closed": "true",
            "limit": str(page),
            "offset": str(offset),
            "order": "volumeNum",
            "ascending": "false",
            "end_date_max": until.isoformat(),
            "end_date_min": since.isoformat(),
        }
        url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                payload = json.loads(r.read().decode())
        except Exception as exc:
            log.warning("page fetch failed at offset=%d: %s", offset, exc)
            break
        if not payload:
            break
        out.extend(payload)
        if len(payload) < page:
            break
        offset += page
        time.sleep(0.2)
    return out[:max_markets]


def _market_to_dict(m: Market) -> dict:
    return {
        "market_id": m.market_id,
        "question": m.question,
        "regime": m.regime,
        "observed_at": m.observed_at.isoformat(),
        "closes_at": m.closes_at.isoformat(),
        "timeline": [d.isoformat() for d in m.timeline],
        "market_prices": {d.isoformat(): p for d, p in m.market_prices.items()},
    }


def main() -> int:
    out_path = Path("runs/live_iter/markets.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adapter = PolymarketAdapter(cache_dir="data_cache/polymarket")

    log.info("fetching high-volume resolved markets from Polymarket gamma...")
    raw_dicts = _fetch_high_volume_markets(date(2024, 1, 1), date(2025, 6, 1), max_markets=300)
    log.info("got %d high-volume resolved market dicts", len(raw_dicts))

    raw_markets: list[PolymarketRawMarket] = []
    for d in raw_dicts:
        r = adapter._parse_market(d)
        if r is None or not r.token_ids:
            continue
        # Need a real binary resolved (one outcome at >=0.99)
        try:
            prices = [float(x) for x in r.outcome_prices]
        except (TypeError, ValueError):
            continue
        if not prices or len(prices) != 2:
            continue
        if max(prices) < 0.95 or min(prices) > 0.05:
            continue
        raw_markets.append(r)
    log.info("after parse + binary-resolved filter: %d", len(raw_markets))

    by_regime: dict[str, list[PolymarketRawMarket]] = defaultdict(list)
    for r in raw_markets:
        by_regime[classify_regime(r.question)].append(r)
    target_per_regime = {
        "election": 8,
        "financial": 5,
        "geo": 4,
        "scientific": 4,
        "sport": 5,
        "other": 4,
    }
    selected: list[PolymarketRawMarket] = []
    for regime, target in target_per_regime.items():
        pool = by_regime.get(regime, [])
        # Stable: just take the first N (which are highest volume since the
        # parent list was volume-sorted).
        selected.extend(pool[:target])
    log.info(
        "selected %d markets across %d regimes: %s",
        len(selected),
        len(by_regime),
        {k: len(v) for k, v in by_regime.items()},
    )

    markets: list[Market] = []
    for i, r in enumerate(selected, 1):
        try:
            m = adapter._build_market(r, timeline_steps=4)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%d/%d] build failed for %s: %s", i, len(selected), r.market_id, exc)
            continue
        if m is None:
            log.info(
                "[%d/%d] skipped %s [%s] (no full price history)",
                i, len(selected), r.market_id, classify_regime(r.question),
            )
            continue
        markets.append(m)
        log.info(
            "[%d/%d] %s [%s] %s..%s prices=%s  Q: %s",
            i, len(selected), m.market_id, m.regime,
            m.timeline[0], m.timeline[-1],
            [round(m.market_prices[d], 3) for d in m.timeline],
            m.question[:50],
        )
        time.sleep(0.15)

    log.info("got %d markets with full timelines", len(markets))

    log.info("fetching resolutions...")
    oracle = adapter.load_resolutions(markets)
    resolved_markets = [m for m in markets if m.market_id in oracle]
    log.info("got %d markets with confirmed resolutions", len(resolved_markets))

    out_payload = {
        "markets": [_market_to_dict(m) for m in resolved_markets],
        "resolutions": {m.market_id: oracle.lookup(m.market_id) for m in resolved_markets},
    }
    out_path.write_text(json.dumps(out_payload, indent=2, default=str))
    log.info("wrote %s (%d markets)", out_path, len(resolved_markets))

    # Print regime breakdown of final dataset.
    final_by_regime: dict[str, int] = defaultdict(int)
    final_outcomes: dict[str, list[int]] = defaultdict(list)
    for m in resolved_markets:
        final_by_regime[m.regime] += 1
        final_outcomes[m.regime].append(oracle.lookup(m.market_id))
    log.info("=== final dataset by regime ===")
    for regime, n in sorted(final_by_regime.items(), key=lambda x: -x[1]):
        outs = final_outcomes[regime]
        rate = sum(outs) / len(outs) if outs else 0.0
        log.info("  %s: %d markets, YES rate %.0f%%", regime, n, rate * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
