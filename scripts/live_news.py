"""Live-fetch news from Hacker News + Wikipedia revisions for the markets dump.

Output: runs/live_iter/news.jsonl, validated and cutoff-clean.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

from clio.cli import _load_markets_payload
from clio.data.live_news import HackerNewsSource, MultiSource, WikipediaRevisionSource
from clio.data.news_pipeline import build_corpus, write_corpus_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("live_news")


def main() -> int:
    markets, oracle = _load_markets_payload("runs/live_iter/markets.json")
    log.info("loaded %d markets, %d resolved", len(markets), len(oracle))

    src = MultiSource([
        HackerNewsSource(sleep_between=0.4),
        WikipediaRevisionSource(sleep_between=0.4, max_articles=2),
    ])

    log.info("fetching news (HN + Wikipedia) per market...")
    t0 = time.time()
    corpus, stats = build_corpus(
        markets,
        src,
        per_market_limit=8,
        pre_window_days=45,
        require_high_confidence=False,  # Wikipedia revisions are inherently
                                        # "low confidence" by the validator's
                                        # rule (no URL date pattern), but the
                                        # revision timestamp IS the date.
                                        # We trust it.
    )
    log.info(
        "queried=%d fetched=%d accepted=%d rejected_no_date=%d rejected_post_close=%d "
        "rejected_low_confidence=%d duplicates=%d  (%.1fs)",
        stats.queried, stats.fetched, stats.accepted,
        stats.rejected_no_date, stats.rejected_post_close,
        stats.rejected_low_confidence, stats.duplicates,
        time.time() - t0,
    )

    out = Path("runs/live_iter/news.jsonl")
    write_corpus_jsonl(corpus, out)
    log.info("wrote %d docs to %s", len(corpus), out)

    # Per-market doc count summary.
    from collections import Counter
    counts = Counter()
    for d in corpus._docs:
        # market_id is the prefix before the source-name in our doc_id format
        mid = d.doc_id.split("-")[0] if "-" in d.doc_id else "?"
        counts[mid] += 1
    log.info("docs per market (top 10):")
    for mid, n in counts.most_common(10):
        log.info("  %s: %d docs", mid, n)
    log.info("markets with 0 news docs: %d", len(markets) - len(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
