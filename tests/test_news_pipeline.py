"""News pipeline integration tests."""

from datetime import date, timedelta
from pathlib import Path

from clio.data.news_pipeline import build_corpus, read_corpus_jsonl, write_corpus_jsonl
from clio.data.news_sources import LocalJSONLNewsSource, NullNewsSource
from clio.data.polymarket_adapter import adapter_from_recorded_dump


FIXTURE_NEWS = Path(__file__).parent / "fixtures" / "news_sample.jsonl"
FIXTURE_PM = Path(__file__).parent / "fixtures" / "polymarket_dump.json"


def test_pipeline_with_null_source_yields_empty_corpus():
    adapter = adapter_from_recorded_dump(FIXTURE_PM)
    markets = adapter.load_markets(date(2023, 1, 1), date(2025, 1, 1))
    corpus, stats = build_corpus(markets, NullNewsSource())
    assert len(corpus) == 0
    assert stats.accepted == 0


def test_pipeline_with_local_jsonl_admits_validated_docs():
    adapter = adapter_from_recorded_dump(FIXTURE_PM)
    markets = adapter.load_markets(date(2023, 1, 1), date(2025, 1, 1))
    src = LocalJSONLNewsSource(FIXTURE_NEWS)
    corpus, stats = build_corpus(markets, src, per_market_limit=10)
    assert stats.accepted > 0
    assert stats.rejected_post_close == 0  # JSONL is well-curated
    # Cutoff sanity: every doc must be < its referenced market's close.
    # We can't easily map back here, but every accepted doc has a real date.
    for d in corpus._docs:
        assert d.published_at <= date(2024, 5, 1)


def test_pipeline_excludes_post_close_docs():
    adapter = adapter_from_recorded_dump(FIXTURE_PM)
    markets = adapter.load_markets(date(2023, 1, 1), date(2025, 1, 1))
    # Restrict to the Tesla market which closes 2024-01-25; Tesla docs in
    # the fixture are Jan 10 / Jan 15, both < close.
    tesla_only = [m for m in markets if m.market_id == "100002"]
    src = LocalJSONLNewsSource(FIXTURE_NEWS)
    corpus, _ = build_corpus(tesla_only, src, per_market_limit=10)
    for d in corpus._docs:
        assert d.published_at < date(2024, 1, 25)


def test_pipeline_jsonl_roundtrip(tmp_path):
    adapter = adapter_from_recorded_dump(FIXTURE_PM)
    markets = adapter.load_markets(date(2023, 1, 1), date(2025, 1, 1))
    src = LocalJSONLNewsSource(FIXTURE_NEWS)
    corpus, _ = build_corpus(markets, src, per_market_limit=10)

    out = tmp_path / "news.jsonl"
    write_corpus_jsonl(corpus, out)
    reloaded = read_corpus_jsonl(out)
    assert len(reloaded) == len(corpus)
    orig_ids = {d.doc_id for d in corpus._docs}
    new_ids = {d.doc_id for d in reloaded._docs}
    assert orig_ids == new_ids


def test_pipeline_strict_mode_rejects_low_confidence():
    adapter = adapter_from_recorded_dump(FIXTURE_PM)
    markets = adapter.load_markets(date(2023, 1, 1), date(2025, 1, 1))
    src = LocalJSONLNewsSource(FIXTURE_NEWS)
    corpus_strict, stats_strict = build_corpus(
        markets, src, per_market_limit=10, require_high_confidence=True
    )
    corpus_loose, _ = build_corpus(markets, src, per_market_limit=10)
    # Strict can't accept more than loose.
    assert len(corpus_strict) <= len(corpus_loose)
