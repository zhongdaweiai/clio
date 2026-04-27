"""News source tests."""

from datetime import date
from pathlib import Path

import pytest

from clio.data.news_sources import (
    LocalJSONLNewsSource,
    NullNewsSource,
    RawArticle,
    TavilyNewsSource,
)


FIXTURE = Path(__file__).parent / "fixtures" / "news_sample.jsonl"


def test_local_jsonl_filters_by_date_window():
    src = LocalJSONLNewsSource(FIXTURE)
    docs = src.search(
        query="Fed rate hike",
        published_after=date(2024, 1, 1),
        published_before=date(2024, 3, 1),
        max_results=5,
    )
    assert all(date(2024, 1, 1) <= d.published_at <= date(2024, 3, 1) for d in docs)
    assert any("fed" in d.title.lower() or "powell" in d.title.lower() for d in docs)


def test_local_jsonl_term_match_required():
    src = LocalJSONLNewsSource(FIXTURE)
    docs = src.search(
        query="completely unrelated zzzqxx",
        published_after=date(2020, 1, 1),
        published_before=date(2030, 1, 1),
        max_results=10,
    )
    assert docs == []


def test_local_jsonl_respects_max_results():
    src = LocalJSONLNewsSource(FIXTURE)
    # broad query that matches many docs
    docs = src.search(
        query="resolution outcome",
        published_after=date(2020, 1, 1),
        published_before=date(2030, 1, 1),
        max_results=2,
    )
    assert len(docs) <= 2


def test_null_source_returns_empty():
    src = NullNewsSource()
    assert (
        src.search("anything", date(2020, 1, 1), date(2030, 1, 1), max_results=5)
        == []
    )


def test_raw_article_to_document_uses_consensus_date():
    art = RawArticle(
        title="t", url="https://example.com/x", content="c",
        source_name="src", published_at=date(2024, 6, 1),
    )
    doc = art.to_document(consensus_date=date(2024, 5, 30))
    assert doc.published_at == date(2024, 5, 30)
    # the published_at on the article (claim) is overridden


def test_tavily_requires_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TavilyNewsSource()
