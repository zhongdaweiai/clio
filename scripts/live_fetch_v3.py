"""Live-fetch v3: scale to 500+ markets across wider time + volume windows.

Wider window (Jan 2024 – Apr 2026), more permissive volume range. Aggressive
deduplication (same conditionId rolled over). Output: markets_v3.json.
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

from clio.data.polymarket_adapter import PolymarketAdapter, PolymarketRawMarket
from clio.frozen.harness import Market
sys.path.insert(0, str(Path(__file__).parent))
from live_fetch_v2 import classify_question_type, _market_to_dict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fetch_v3")

_HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}


def _fetch_pages(since: date, until: date, max_markets: int,
                 min_vol: float, max_vol: float, order: str = "volumeNum") -> list[dict]:
    out: list[dict] = []
    offset = 0
    page = 100
    while len(out) < max_markets:
        params = {
            "closed": "true", "limit": str(page), "offset": str(offset),
            "order": order, "ascending": "false",
            "end_date_max": until.isoformat(),
            "end_date_min": since.isoformat(),
        }
        url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.loads(r.read().decode())
        except Exception as exc:
            log.warning("page fetch failed offset=%d: %s", offset, exc)
            break
        if not payload:
            break
        for m in payload:
            try:
                vol = float(m.get("volumeNum") or 0)
            except (TypeError, ValueError):
                continue
            if min_vol <= vol <= max_vol:
                out.append(m)
        if len(payload) < page:
            break
        offset += page
        time.sleep(0.15)
        if len(out) >= max_markets:
            break
    return out[:max_markets]


def main() -> int:
    out_path = Path("runs/live_iter/markets_v3.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adapter = PolymarketAdapter(cache_dir="data_cache/polymarket")

    log.info("fetching markets across multiple volume bands...")
    # Three-band sweep: keeps the dataset diverse without hitting the same
    # mega-volume markets again and again.
    raws_all: list[dict] = []
    for band, (lo, hi, target) in [
        ("high",     (5_000_000,  50_000_000,  300)),
        ("mid",      (500_000,    5_000_000,   500)),
        ("lower",    (100_000,    500_000,     400)),
    ]:
        log.info("  band=%s vol=[%.0fK, %.0fM] target=%d", band, lo / 1_000, hi / 1_000_000, target)
        chunk = _fetch_pages(date(2024, 1, 1), date(2026, 5, 1), target, lo, hi)
        log.info("    got %d", len(chunk))
        raws_all.extend(chunk)

    # Dedupe by conditionId.
    seen: set[str] = set()
    unique = []
    for m in raws_all:
        cid = m.get("conditionId") or m.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(m)
    log.info("after dedup: %d unique markets", len(unique))

    raw_markets: list[tuple[PolymarketRawMarket, str]] = []
    type_counts: dict[str, int] = defaultdict(int)
    for d in unique:
        r = adapter._parse_market(d)
        if r is None or not r.token_ids:
            continue
        try:
            prices = [float(x) for x in r.outcome_prices]
        except (TypeError, ValueError):
            continue
        if len(prices) != 2:
            continue
        if max(prices) < 0.95 or min(prices) > 0.05:
            continue
        if (r.end_date - r.start_date).days < 14:
            continue
        qtype = classify_question_type(r.question)
        type_counts[qtype] += 1
        raw_markets.append((r, qtype))
    log.info("after binary+lifetime+parse filters: %d  types=%s", len(raw_markets), dict(type_counts))

    # Build full timelines. Cache means re-runs are fast.
    markets: list[tuple[Market, str]] = []
    for i, (r, qtype) in enumerate(raw_markets, 1):
        try:
            m = adapter._build_market(r, timeline_steps=4)
        except Exception as exc:
            continue
        if m is None:
            continue
        markets.append((m, qtype))
        if i % 100 == 0:
            log.info("[%d/%d] built %d markets so far", i, len(raw_markets), len(markets))
        time.sleep(0.05)
    log.info("got %d markets with full timelines", len(markets))

    log.info("fetching resolutions...")
    oracle = adapter.load_resolutions([m for m, _ in markets])
    resolved = [(m, t) for m, t in markets if m.market_id in oracle]
    log.info("got %d resolved markets", len(resolved))

    by_type: dict[str, list[int]] = defaultdict(list)
    for m, t in resolved:
        by_type[t].append(oracle.lookup(m.market_id))
    log.info("=== final dataset ===")
    for t, outs in sorted(by_type.items()):
        log.info("  %s: %d markets, YES rate %.1f%%", t, len(outs), 100 * sum(outs) / len(outs) if outs else 0)

    payload = {
        "markets": [_market_to_dict(m, t) for m, t in resolved],
        "resolutions": {m.market_id: oracle.lookup(m.market_id) for m, _ in resolved},
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote %s (%d markets)", out_path, len(resolved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
