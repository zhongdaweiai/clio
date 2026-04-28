"""Backfill polymarket_url in existing paper_trades JSON files.

For each rec, fetch the market from gamma API and use its event slug.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}


def fetch_event_slug(market_id: str) -> str | None:
    """market_id is the conditionId. Try condition_ids first, then id."""
    for q in [{"condition_ids": market_id}, {"id": market_id}]:
        url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode(q)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=15) as r:
                data = json.loads(r.read().decode())
        except Exception:
            continue
        if not data:
            continue
        m = data[0] if isinstance(data, list) else data
        events = m.get("events") or []
        if events and isinstance(events[0], dict) and events[0].get("slug"):
            return events[0]["slug"]
        return m.get("slug")
    return None


def main() -> int:
    pt = Path("paper_trades")
    files = sorted(pt.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    if not files:
        print("no files")
        return 0
    n_total = 0
    n_changed = 0
    for f in files:
        scan = json.loads(f.read_text())
        for rec in scan["recommendations"]:
            n_total += 1
            cur = rec.get("polymarket_url", "")
            slug = fetch_event_slug(rec["market_id"])
            time.sleep(0.10)
            if not slug:
                continue
            new_url = f"https://polymarket.com/event/{slug}"
            if new_url != cur:
                rec["polymarket_url"] = new_url
                n_changed += 1
                print(f"  ✓ {rec['question'][:60]}: → {slug}")
        f.write_text(json.dumps(scan, indent=2))
    print(f"\n{n_changed}/{n_total} URLs updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
