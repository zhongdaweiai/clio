"""Live, free, date-filterable news sources.

Three implementations, all free and date-correct:

- `HackerNewsSource`: Algolia HN API. Every story has a real Unix timestamp.
  Best signal for tech / crypto / startup markets.
- `WikipediaRevisionSource`: Wikipedia article snapshots at a given date,
  via the REST API's revision-by-date lookup. Best for general background
  on named entities. Inherently a *snapshot* source — gives the article as
  it stood on or before the cutoff.
- `MultiSource`: combines several sources, deduplicates by URL.

These are not a replacement for a paid news API. They are good enough to
demonstrate the closed loop on real data, with rigorous date guarantees.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from typing import Sequence

from clio.data.news_sources import RawArticle


log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}


def _get_json(url: str, timeout: int = 20) -> dict | list | None:
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        log.debug("fetch failed %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------
# Hacker News (via Algolia)
# ---------------------------------------------------------------------


class HackerNewsSource:
    """Algolia HN search. Returns stories ranked by relevance.

    The Algolia API guarantees a real `created_at_i` Unix timestamp for every
    story, so date filtering is trustworthy at the source. We additionally
    pass it through the validator downstream.
    """

    name = "hacker_news"
    _ENDPOINT = "https://hn.algolia.com/api/v1/search"

    def __init__(self, sleep_between: float = 0.4) -> None:
        self.sleep_between = sleep_between

    def search(
        self,
        query: str,
        published_after: date,
        published_before: date,
        max_results: int = 10,
    ) -> list[RawArticle]:
        # Algolia takes Unix timestamps for numeric filters.
        after_ts = int(datetime(
            published_after.year, published_after.month, published_after.day,
            tzinfo=timezone.utc,
        ).timestamp())
        before_ts = int(datetime(
            published_before.year, published_before.month, published_before.day,
            tzinfo=timezone.utc,
        ).timestamp())

        # HN search is keyword-AND. We progressively shorten the query until
        # we get hits, prioritizing the most informative tokens.
        hits: list[dict] = []
        for n_terms in (4, 3, 2):
            cleaned = _strip_market_phrasing(query, max_terms=n_terms)
            if not cleaned:
                continue
            params = {
                "query": cleaned,
                "tags": "story",
                "numericFilters": f"created_at_i>{after_ts},created_at_i<{before_ts}",
                "hitsPerPage": str(max(max_results * 2, 10)),
            }
            url = f"{self._ENDPOINT}?{urllib.parse.urlencode(params)}"
            payload = _get_json(url)
            time.sleep(self.sleep_between)
            if payload and payload.get("hits"):
                hits = payload["hits"]
                break
        if not hits:
            return []

        out: list[RawArticle] = []
        for hit in hits:
            ts = hit.get("created_at_i")
            if ts is None:
                continue
            d = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            if d < published_after or d > published_before:
                continue
            title = hit.get("title") or ""
            story_text = hit.get("story_text") or ""
            content = title + ". " + (story_text[:1500] if story_text else "")
            url_field = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            out.append(
                RawArticle(
                    title=title,
                    url=url_field,
                    content=content,
                    source_name="hacker_news",
                    published_at=d,
                    raw_html=None,
                    extra={
                        "objectID": hit.get("objectID"),
                        "points": hit.get("points"),
                        "num_comments": hit.get("num_comments"),
                    },
                )
            )
            if len(out) >= max_results:
                break
        return out


# ---------------------------------------------------------------------
# Wikipedia revisions
# ---------------------------------------------------------------------


class WikipediaRevisionSource:
    """Wikipedia article *as it stood on or before* a given date.

    Two-step query:
      1. Search for an article matching the question's named entities.
      2. Fetch the most recent revision <= published_before via the
         MediaWiki API.

    This is a snapshot source — the "publication date" of the document is
    the actual revision timestamp. That's how we honor the cutoff.

    Coverage: best for named-entity questions ("Will Trump win Iowa?",
    "Will Bitcoin hit 100k?"). Poor for novelty questions.
    """

    name = "wikipedia_rev"
    _SEARCH = "https://en.wikipedia.org/w/api.php"
    _REST = "https://en.wikipedia.org/api/rest_v1"

    def __init__(self, sleep_between: float = 0.3, max_articles: int = 3) -> None:
        self.sleep_between = sleep_between
        self.max_articles = max_articles

    def search(
        self,
        query: str,
        published_after: date,
        published_before: date,
        max_results: int = 5,
    ) -> list[RawArticle]:
        candidates = self._find_candidate_titles(query)
        if not candidates:
            return []
        out: list[RawArticle] = []
        for title in candidates[: self.max_articles]:
            art = self._fetch_revision(title, published_before)
            if art is None:
                continue
            if art.published_at and art.published_at < published_after:
                # Article hasn't been edited in the window — still informative
                # because we want the snapshot AS OF, but the validator might
                # demote its confidence. We keep it.
                pass
            out.append(art)
            if len(out) >= max_results:
                break
            time.sleep(self.sleep_between)
        return out

    def _find_candidate_titles(self, query: str) -> list[str]:
        cleaned = _strip_market_phrasing(query)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": cleaned,
            "format": "json",
            "srlimit": "5",
        }
        url = f"{self._SEARCH}?{urllib.parse.urlencode(params)}"
        payload = _get_json(url)
        time.sleep(self.sleep_between)
        if not payload:
            return []
        return [hit["title"] for hit in payload.get("query", {}).get("search", [])]

    def _fetch_revision(self, title: str, as_of: date) -> RawArticle | None:
        # Use revisions API: get the latest revision before as_of.
        rvend = datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc).isoformat()
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "timestamp|ids|content",
            "rvslots": "main",
            "rvlimit": "1",
            "rvstart": rvend,  # rvstart is "list newer than this" when dir=older
            "rvdir": "older",
            "format": "json",
        }
        url = f"{self._SEARCH}?{urllib.parse.urlencode(params)}"
        payload = _get_json(url, timeout=30)
        if not payload:
            return None
        pages = payload.get("query", {}).get("pages", {}) or {}
        for _, page in pages.items():
            revs = page.get("revisions") or []
            if not revs:
                continue
            rev = revs[0]
            ts = rev.get("timestamp")
            if not ts:
                continue
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
            except ValueError:
                continue
            content_obj = rev.get("slots", {}).get("main", {}) or {}
            raw_text = content_obj.get("*") or content_obj.get("content") or ""
            # Wikitext can be huge; trim aggressively.
            content = _strip_wikitext(raw_text)[:3000]
            url_field = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}?oldid={rev.get('revid')}"
            return RawArticle(
                title=f"Wikipedia: {title}",
                url=url_field,
                content=content,
                source_name="wikipedia_rev",
                published_at=d,
                raw_html=None,
                extra={"revid": rev.get("revid"), "title": title},
            )
        return None


# ---------------------------------------------------------------------
# Multi
# ---------------------------------------------------------------------


class MultiSource:
    """Concatenate results from multiple sources, deduplicate by URL."""

    name = "multi"

    def __init__(self, sources: Sequence) -> None:
        self.sources = list(sources)

    def search(
        self,
        query: str,
        published_after: date,
        published_before: date,
        max_results: int = 10,
    ) -> list[RawArticle]:
        per_src = max(2, max_results // max(1, len(self.sources)))
        seen: set[str] = set()
        out: list[RawArticle] = []
        for src in self.sources:
            try:
                arts = src.search(query, published_after, published_before, per_src)
            except Exception as exc:  # noqa: BLE001 — never let one source crash the run
                log.warning("source %s failed: %s", getattr(src, "name", "?"), exc)
                continue
            for a in arts:
                if a.url in seen:
                    continue
                seen.add(a.url)
                out.append(a)
                if len(out) >= max_results:
                    return out
        return out


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_STRIP_PATTERNS = [
    re.compile(r"^will\s+", re.IGNORECASE),
    re.compile(r"\?\s*$"),
    re.compile(r"\bby\s+\d{4}-\d{2}-\d{2}\b", re.IGNORECASE),
    re.compile(r"\bbefore\s+\w+\s+\d+,?\s*\d{4}\b", re.IGNORECASE),
    re.compile(r"\bbefore\s+\w+\s+\d+\b", re.IGNORECASE),
]

# Words that consume a query slot without adding signal. The set is tuned
# for prediction-market questions, not general English.
_STOPWORDS = {
    "a", "an", "the", "to", "of", "for", "in", "on", "at", "by", "with",
    "is", "be", "been", "being", "are", "was", "were",
    "win", "wins", "won", "lose", "loses", "lost", "beat", "beats",
    "have", "has", "had",
    "this", "that", "these", "those",
    "any", "all", "or", "and",
    "their", "his", "her", "its",
    "than", "more", "most", "less", "least",
    "before", "after", "during", "between",
    "yes", "no",
}


def _strip_market_phrasing(question: str, max_terms: int = 5) -> str:
    """Reduce a market question to its most informative ~5 query terms.

    Strategy:
      1. Strip market boilerplate ("Will ...?", date phrases).
      2. Tokenize on word boundaries.
      3. Drop stopwords.
      4. Keep proper nouns (capitalized, multi-letter), 4-digit years, and
         the longest remaining tokens up to `max_terms`.
    """
    q = question
    for pat in _STRIP_PATTERNS:
        q = pat.sub("", q)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+|\d{4}", q)
    keep: list[str] = []
    seen: set[str] = set()
    # Pass 1: proper nouns and years.
    for t in tokens:
        low = t.lower()
        if low in _STOPWORDS:
            continue
        if (t[0].isupper() and len(t) > 1) or t.isdigit():
            if low not in seen:
                keep.append(t)
                seen.add(low)
    # Pass 2: fill with remaining content words, longest first.
    remaining = sorted(
        [t for t in tokens if t.lower() not in _STOPWORDS and t.lower() not in seen],
        key=lambda t: -len(t),
    )
    for t in remaining:
        if len(keep) >= max_terms:
            break
        if t.lower() in seen:
            continue
        keep.append(t)
        seen.add(t.lower())
    return " ".join(keep[:max_terms])


def _strip_wikitext(s: str) -> str:
    # Lightweight wikitext-to-plaintext: remove templates, refs, links.
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.DOTALL)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()
