"""News pipeline: fetch → validate → corpus.

Used by the CLI's `clio fetch news` subcommand and by integration tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

from clio.data.date_validator import validate_published_date
from clio.data.news_sources import NewsSourceProtocol, RawArticle
from clio.frozen.corpus import Corpus
from clio.frozen.harness import Market


log = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    queried: int = 0
    fetched: int = 0
    rejected_no_date: int = 0
    rejected_post_close: int = 0
    rejected_low_confidence: int = 0
    accepted: int = 0
    duplicates: int = 0


def build_corpus(
    markets: Iterable[Market],
    source: NewsSourceProtocol,
    per_market_limit: int = 8,
    pre_window_days: int = 30,
    require_high_confidence: bool = False,
) -> tuple[Corpus, PipelineStats]:
    corpus = Corpus()
    stats = PipelineStats()
    seen_doc_ids: set[str] = set()

    for m in markets:
        stats.queried += 1
        articles = source.search(
            query=m.question,
            published_after=m.observed_at - timedelta(days=pre_window_days),
            published_before=m.closes_at,
            max_results=per_market_limit,
        )
        stats.fetched += len(articles)

        for raw in articles:
            v = validate_published_date(
                claimed_date=raw.published_at,
                url=raw.url,
                html=raw.raw_html,
            )
            if v.confidence == "rejected" or v.consensus_date is None:
                stats.rejected_no_date += 1
                continue
            if require_high_confidence and v.confidence != "high":
                stats.rejected_low_confidence += 1
                continue
            if v.consensus_date >= m.closes_at:
                stats.rejected_post_close += 1
                continue

            doc = raw.to_document(consensus_date=v.consensus_date, market_id=m.market_id)
            if doc.doc_id in seen_doc_ids:
                stats.duplicates += 1
                continue
            seen_doc_ids.add(doc.doc_id)
            corpus.add(doc)
            stats.accepted += 1

    return corpus, stats


def write_corpus_jsonl(corpus: Corpus, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for d in corpus._docs:  # noqa: SLF001 — internal access for export
            f.write(
                json.dumps(
                    {
                        "doc_id": d.doc_id,
                        "published_at": d.published_at.isoformat(),
                        "title": d.title,
                        "content": d.content,
                        "source": d.source,
                        "tags": list(d.tags),
                    }
                )
                + "\n"
            )


def read_corpus_jsonl(path: str | Path) -> Corpus:
    from datetime import date

    from clio.frozen.corpus import Document

    corpus = Corpus()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            corpus.add(
                Document(
                    doc_id=obj["doc_id"],
                    published_at=date.fromisoformat(obj["published_at"]),
                    title=obj.get("title", ""),
                    content=obj.get("content", ""),
                    source=obj.get("source", "unknown"),
                    tags=tuple(obj.get("tags") or ()),
                )
            )
    return corpus
