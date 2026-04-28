"""Parallel news fetch using ThreadPoolExecutor.

Each market gets fetched concurrently (8 workers), bringing 638-market
fetch from ~50 min serial down to ~5-7 min.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

from clio.cli import _load_markets_payload
from clio.data.date_validator import validate_published_date
from clio.data.live_news import HackerNewsSource, MultiSource, WikipediaRevisionSource
from clio.data.news_pipeline import write_corpus_jsonl
from clio.frozen.corpus import Corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("news_par")


def fetch_one_market(m, src, pre_window_days=45, per_market_limit=8):
    """Returns list of validated Documents for this market, or [] on failure."""
    try:
        articles = src.search(
            query=m.question,
            published_after=m.observed_at - timedelta(days=pre_window_days),
            published_before=m.closes_at,
            max_results=per_market_limit,
        )
    except Exception as exc:
        return []
    docs = []
    for art in articles:
        v = validate_published_date(claimed_date=art.published_at, url=art.url, html=art.raw_html)
        if v.confidence == "rejected" or v.consensus_date is None:
            continue
        if v.consensus_date >= m.closes_at:
            continue
        d = art.to_document(consensus_date=v.consensus_date, market_id=m.market_id)
        docs.append(d)
    return docs


def main() -> int:
    markets, _ = _load_markets_payload("runs/live_iter/markets_v3.json")
    log.info("loaded %d markets", len(markets))

    src = MultiSource([
        HackerNewsSource(sleep_between=0.1),
        WikipediaRevisionSource(sleep_between=0.1, max_articles=2),
    ])

    corpus = Corpus()
    seen_ids: set[str] = set()
    t0 = time.time()
    n_done = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_one_market, m, src): m for m in markets}
        for fut in as_completed(futures):
            n_done += 1
            try:
                docs = fut.result()
            except Exception as exc:
                continue
            for d in docs:
                if d.doc_id in seen_ids:
                    continue
                seen_ids.add(d.doc_id)
                corpus.add(d)
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(markets) - n_done) / rate if rate > 0 else 0
                log.info(
                    f"[{n_done}/{len(markets)}] corpus={len(corpus)} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )

    log.info("done in %.1fs, %d docs", time.time() - t0, len(corpus))
    out = Path("runs/live_iter/news_v3.jsonl")
    write_corpus_jsonl(corpus, out)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
