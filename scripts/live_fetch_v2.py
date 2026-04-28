"""Live-fetch v2: broader, more beatable market set.

Strategy: instead of top-volume (where the market is most efficient), pull a
LARGER set across a wider volume range. We specifically want:

- Deadline questions: "X by date Y", where markets historically over-price
  the deadline due to FOMO/recency.
- Field-of-N questions: "who wins X" with 5+ contestants, where base rate
  1/N is informative and markets tend to overprice favorites.
- Durative questions: "X remains Y", where status quo is the right prior.

Output: runs/live_iter/markets_v2.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

from clio.data.polymarket_adapter import PolymarketAdapter, PolymarketRawMarket
from clio.frozen.harness import Market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fetch_v2")

_HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}


# ---- question-type classifier (also used by the strategy at scoring time) ----


_DEADLINE_PATTERNS = [
    re.compile(r"\bbefore\s+\w+\s+\d", re.IGNORECASE),
    re.compile(r"\bby\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE),
    re.compile(r"\bby\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\bin\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(year|month|week|quarter)\b", re.IGNORECASE),
]

_FIELD_PATTERNS = [
    re.compile(r"\bwins?\b.+\b(20\d{2}|champion|cup|bowl|primary|election|tournament|finals?)\b", re.IGNORECASE),
    re.compile(r"\bwinner\b", re.IGNORECASE),
]

_DURATIVE_PATTERNS = [
    re.compile(r"\bremain[s]?\b", re.IGNORECASE),
    re.compile(r"\bstay[s]?\b.+\b(in|as|the)\b", re.IGNORECASE),
    re.compile(r"\bcontinue[s]?\b", re.IGNORECASE),
    re.compile(r"\bstill\b", re.IGNORECASE),
]


def classify_question_type(question: str) -> str:
    """Return one of {deadline, field, durative, event}."""
    q = question.lower()
    if any(p.search(q) for p in _DURATIVE_PATTERNS):
        return "durative"
    is_deadline = any(p.search(q) for p in _DEADLINE_PATTERNS)
    is_field = any(p.search(q) for p in _FIELD_PATTERNS)
    if is_field and not is_deadline:
        return "field"
    if is_deadline:
        return "deadline"
    return "event"


# ---- broader fetcher with volume range ----


def _fetch_markets_paginated(
    since: date,
    until: date,
    max_markets: int,
    min_volume: float = 100_000,
    max_volume: float = 100_000_000,
    order: str = "volumeNum",
) -> list[dict]:
    out: list[dict] = []
    offset = 0
    page = 100
    while len(out) < max_markets:
        params = {
            "closed": "true",
            "limit": str(page),
            "offset": str(offset),
            "order": order,
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
        for m in payload:
            try:
                vol = float(m.get("volumeNum") or 0)
            except (TypeError, ValueError):
                continue
            if vol < min_volume or vol > max_volume:
                continue
            out.append(m)
        if len(payload) < page:
            break
        offset += page
        time.sleep(0.15)
        if len(out) >= max_markets:
            break
    return out[:max_markets]


def _market_to_dict(m: Market, qtype: str) -> dict:
    return {
        "market_id": m.market_id,
        "question": m.question,
        "regime": m.regime,
        "qtype": qtype,
        "observed_at": m.observed_at.isoformat(),
        "closes_at": m.closes_at.isoformat(),
        "timeline": [d.isoformat() for d in m.timeline],
        "market_prices": {d.isoformat(): p for d, p in m.market_prices.items()},
    }


def main() -> int:
    out_path = Path("runs/live_iter/markets_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adapter = PolymarketAdapter(cache_dir="data_cache/polymarket")

    # Sweep volume range that excludes the very largest (most efficient) and
    # the smallest (insufficient price history) markets.
    log.info("fetching mid-volume resolved markets...")
    raw_dicts = _fetch_markets_paginated(
        since=date(2024, 1, 1),
        until=date(2025, 6, 1),
        max_markets=600,
        min_volume=200_000,
        max_volume=50_000_000,
    )
    log.info("got %d candidate markets in volume window", len(raw_dicts))

    raw_markets: list[tuple[PolymarketRawMarket, str]] = []
    type_counts: dict[str, int] = defaultdict(int)
    for d in raw_dicts:
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
        # Need substantive lifetime
        if (r.end_date - r.start_date).days < 14:
            continue
        qtype = classify_question_type(r.question)
        type_counts[qtype] += 1
        raw_markets.append((r, qtype))

    log.info("after binary+lifetime filter: %d  types=%s", len(raw_markets), dict(type_counts))

    # Aim for type diversity: more deadline + field markets, fewer pure event.
    targets = {"deadline": 40, "field": 35, "durative": 15, "event": 30}
    selected: list[tuple[PolymarketRawMarket, str]] = []
    by_type: dict[str, list] = defaultdict(list)
    for r, t in raw_markets:
        by_type[t].append((r, t))
    for qtype, target in targets.items():
        pool = by_type.get(qtype, [])
        # Already volume-sorted descending from the parent fetch.
        selected.extend(pool[:target])
    log.info(
        "selected %d markets: %s",
        len(selected),
        {k: sum(1 for _, t in selected if t == k) for k in targets},
    )

    # Build full-timeline Markets.
    markets: list[tuple[Market, str]] = []
    for i, (r, qtype) in enumerate(selected, 1):
        try:
            m = adapter._build_market(r, timeline_steps=4)
        except Exception as exc:
            log.warning("[%d/%d] build failed %s: %s", i, len(selected), r.market_id, exc)
            continue
        if m is None:
            continue
        markets.append((m, qtype))
        if i % 20 == 0:
            log.info("[%d/%d] kept %d so far", i, len(selected), len(markets))
        time.sleep(0.10)
    log.info("got %d markets with full timelines", len(markets))

    log.info("fetching resolutions...")
    oracle = adapter.load_resolutions([m for m, _ in markets])
    resolved = [(m, t) for m, t in markets if m.market_id in oracle]
    log.info("got %d markets with confirmed resolutions", len(resolved))

    # Type & outcome breakdown
    yes_by_type: dict[str, list[int]] = defaultdict(list)
    for m, t in resolved:
        yes_by_type[t].append(oracle.lookup(m.market_id))
    log.info("=== final dataset ===")
    for t, outs in sorted(yes_by_type.items()):
        log.info("  %s: %d markets, YES rate %.0f%%", t, len(outs), 100 * sum(outs) / len(outs) if outs else 0)

    out_payload = {
        "markets": [_market_to_dict(m, t) for m, t in resolved],
        "resolutions": {m.market_id: oracle.lookup(m.market_id) for m, _ in resolved},
    }
    out_path.write_text(json.dumps(out_payload, indent=2, default=str))
    log.info("wrote %s (%d markets)", out_path, len(resolved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
