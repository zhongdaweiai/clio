"""Data adapters.

- `synthetic`: generates a self-consistent toy world with markets, news docs,
  and resolutions. Used by tests and the CLI demo.
- `polymarket_adapter`: live Polymarket fetcher (gamma + clob).
- `news_sources`: pluggable news fetchers (Tavily, local JSONL, null).
- `date_validator`: paranoid published-date verification.
- `news_pipeline`: fetch → validate → cutoff-clean corpus.
- `cache`: disk-backed adapter cache.
"""

from clio.data.cache import DiskCache
from clio.data.date_validator import (
    DateValidation,
    extract_date_from_url,
    extract_dates_from_html,
    validate_published_date,
)
from clio.data.news_pipeline import PipelineStats, build_corpus, read_corpus_jsonl, write_corpus_jsonl
from clio.data.news_sources import (
    LocalJSONLNewsSource,
    NewsSourceProtocol,
    NullNewsSource,
    RawArticle,
    TavilyNewsSource,
)
from clio.data.polymarket_adapter import (
    PolymarketAdapter,
    PolymarketFetchError,
    adapter_from_recorded_dump,
    classify_regime,
)
from clio.data.synthetic import SyntheticConfig, generate_synthetic_world

__all__ = [
    "DiskCache",
    "DateValidation",
    "extract_date_from_url",
    "extract_dates_from_html",
    "validate_published_date",
    "PipelineStats",
    "build_corpus",
    "read_corpus_jsonl",
    "write_corpus_jsonl",
    "LocalJSONLNewsSource",
    "NewsSourceProtocol",
    "NullNewsSource",
    "RawArticle",
    "TavilyNewsSource",
    "PolymarketAdapter",
    "PolymarketFetchError",
    "adapter_from_recorded_dump",
    "classify_regime",
    "SyntheticConfig",
    "generate_synthetic_world",
]
