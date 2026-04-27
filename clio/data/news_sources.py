"""News sources for the corpus.

The contract: a `NewsSource.search(...)` returns a list of `RawArticle` objects.
Date filtering is best-effort here — the caller (Polymarket adapter) re-validates
every article's published_at via `clio.data.date_validator` before admitting it
to the corpus. Defense in depth: never trust a source's claim of when something
was published.

Three implementations:
- TavilyNewsSource: hits Tavily Search API. Requires TAVILY_API_KEY.
- LocalJSONLNewsSource: replays a JSONL file. Used by tests + offline backtests.
- NullNewsSource: returns nothing. Useful baseline.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from clio.frozen.corpus import Document


@dataclass
class RawArticle:
    """An article fetched from a news source. Not yet validated.

    `published_at` is the source's claim. The date validator may override it
    or reject the article entirely.
    """
    title: str
    url: str
    content: str
    source_name: str
    published_at: date | None = None
    raw_html: str | None = None
    extra: dict = field(default_factory=dict)

    def to_document(self, consensus_date: date, market_id: str | None = None) -> Document:
        # doc_id is content-hash-derived so the same URL doesn't double-count.
        h = hashlib.sha256(self.url.encode()).hexdigest()[:12]
        prefix = f"{market_id}-" if market_id else ""
        return Document(
            doc_id=f"{prefix}{self.source_name}-{h}",
            published_at=consensus_date,
            title=self.title,
            content=self.content,
            source=self.source_name,
            tags=tuple(),
        )


class NewsSourceProtocol(Protocol):
    name: str

    def search(
        self,
        query: str,
        published_after: date,
        published_before: date,
        max_results: int = 10,
    ) -> list[RawArticle]: ...


# ---------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------


class TavilyNewsSource:
    """Tavily Search API wrapper.

    POST https://api.tavily.com/search
    body: {api_key, query, search_depth, max_results, include_raw_content,
           topic, days, ...}

    Honesty disclosure: like the Polymarket adapter, this code follows the
    public Tavily contract but is not network-tested in this session. Set
    TAVILY_API_KEY and run `clio fetch news --source tavily --dry-run` to
    smoke test against the real endpoint before backtesting on its output.
    """

    name = "tavily"
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
        search_depth: str = "advanced",
    ) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TavilyNewsSource requires an api_key or TAVILY_API_KEY env var"
            )
        self.timeout = timeout
        self.search_depth = search_depth

    def search(
        self,
        query: str,
        published_after: date,
        published_before: date,
        max_results: int = 10,
    ) -> list[RawArticle]:
        # Tavily's `days` parameter is "go back N days from today" — useless for
        # historical backtests. We post-filter on the dates the API returns.
        body = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": max(max_results * 2, 10),
            "include_raw_content": True,
            "topic": "news",
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self._ENDPOINT,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "clio/0.1"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise NewsFetchError(f"Tavily query failed: {exc}") from exc

        out: list[RawArticle] = []
        for r in payload.get("results", []):
            pub = _parse_date_field(r.get("published_date"))
            if pub is not None:
                if pub < published_after or pub > published_before:
                    continue
            out.append(
                RawArticle(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", "") or r.get("raw_content", ""),
                    source_name="tavily",
                    published_at=pub,
                    raw_html=r.get("raw_content"),
                    extra={"score": r.get("score")},
                )
            )
            if len(out) >= max_results:
                break
        return out


# ---------------------------------------------------------------------
# Local JSONL (used by tests and offline)
# ---------------------------------------------------------------------


class LocalJSONLNewsSource:
    """Reads articles from a local JSONL file. Format:

        {"title": ..., "url": ..., "content": ..., "published_at": "YYYY-MM-DD",
         "source": "..."}

    Used by tests to give the adapter a deterministic news feed without
    depending on the network.
    """

    def __init__(self, path: str | Path, name: str = "local_jsonl") -> None:
        self.path = Path(path)
        self.name = name
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def _iter_articles(self) -> Iterator[RawArticle]:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield RawArticle(
                    title=obj.get("title", ""),
                    url=obj.get("url", ""),
                    content=obj.get("content", ""),
                    source_name=obj.get("source", self.name),
                    published_at=_parse_date_field(obj.get("published_at")),
                    raw_html=obj.get("raw_html"),
                )

    def search(
        self,
        query: str,
        published_after: date,
        published_before: date,
        max_results: int = 10,
    ) -> list[RawArticle]:
        terms = [t.lower() for t in re.split(r"\W+", query) if t]
        out: list[RawArticle] = []
        for art in self._iter_articles():
            pub = art.published_at
            if pub is None:
                continue
            if pub < published_after or pub > published_before:
                continue
            haystack = (art.title + " " + art.content).lower()
            if not any(t in haystack for t in terms):
                continue
            out.append(art)
            if len(out) >= max_results:
                break
        return out


# ---------------------------------------------------------------------
# Null
# ---------------------------------------------------------------------


class NullNewsSource:
    """Returns no articles. Useful baseline strategy."""

    name = "null"

    def search(
        self,
        query: str,
        published_after: date,
        published_before: date,
        max_results: int = 10,
    ) -> list[RawArticle]:
        return []


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class NewsFetchError(RuntimeError):
    pass


def _parse_date_field(s: str | None) -> date | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return dt.datetime.strptime(s[: len(fmt)], fmt).date()
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None
