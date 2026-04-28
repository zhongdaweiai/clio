"""Fetch news (HN+Wikipedia) for the 638 v3 markets, save to v3 news.jsonl."""

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
log = logging.getLogger("news_v3")


def main() -> int:
    markets, oracle = _load_markets_payload("runs/live_iter/markets_v3.json")
    log.info("loaded %d markets", len(markets))

    src = MultiSource([
        HackerNewsSource(sleep_between=0.20),
        WikipediaRevisionSource(sleep_between=0.20, max_articles=2),
    ])

    t0 = time.time()
    corpus, stats = build_corpus(
        markets, src, per_market_limit=8, pre_window_days=45,
        require_high_confidence=False,
    )
    log.info(
        "queried=%d fetched=%d accepted=%d rejected_no_date=%d rejected_post_close=%d  (%.1fs)",
        stats.queried, stats.fetched, stats.accepted,
        stats.rejected_no_date, stats.rejected_post_close, time.time() - t0,
    )
    out = Path("runs/live_iter/news_v3.jsonl")
    write_corpus_jsonl(corpus, out)
    log.info("wrote %d docs to %s", len(corpus), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
