"""Polymarket adapter — fetch resolved markets and historical prices.

Endpoints used:
- Gamma API:   https://gamma-api.polymarket.com/markets
               https://gamma-api.polymarket.com/markets/<id>
- CLOB API:    https://clob.polymarket.com/prices-history?market=<token_id>&interval=...

The adapter's contract:
  load_markets(since, until) -> list[Market]      (resolved markets only)
  load_resolutions(markets)  -> ResolutionOracle
  load_news(markets, source) -> Corpus            (delegates to NewsSource)

Honesty disclosure: this code is written to the public Polymarket API spec but
**has not been network-tested in this session**. The unit tests exercise it
against a recorded fixture (`tests/fixtures/polymarket_*.json`) that mirrors
the real response shape. Before relying on it for real backtests, run
`clio fetch polymarket --dry-run` against live and verify the field names still
match — Polymarket has historically renamed fields without warning.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from clio.data.cache import DiskCache
from clio.frozen.corpus import Corpus
from clio.frozen.harness import Market
from clio.frozen.oracle import ResolutionOracle


_GAMMA = "https://gamma-api.polymarket.com"
_CLOB = "https://clob.polymarket.com"
_DEFAULT_TIMEOUT = 30
_DEFAULT_TTL = None  # forever — resolved markets are immutable


log = logging.getLogger(__name__)


@dataclass
class PolymarketRawMarket:
    """A trimmed view of the gamma /markets response. Kept loose because the
    schema drifts; we only need the fields below."""

    market_id: str
    condition_id: str
    question: str
    slug: str
    outcomes: list[str]
    outcome_prices: list[float]
    token_ids: list[str]  # CLOB ERC-1155 token IDs, one per outcome
    start_date: date
    end_date: date
    closed: bool
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _parse_iso_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _http_get_json(url: str, timeout: int = _DEFAULT_TIMEOUT) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "clio/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise PolymarketFetchError(f"GET {url}: {exc}") from exc


class PolymarketFetchError(RuntimeError):
    pass


_REGIME_KEYWORDS = {
    "election": [
        r"\belect", r"\bvote", r"\bprimary", r"\bsenate", r"\bgovernor",
        r"\bnominee", r"\bpoll", r"\bcandidat", r"\bpresident",
    ],
    "sport": [
        r"\bnba\b", r"\bnfl\b", r"\bmlb\b", r"\bnhl\b", r"\bsoccer",
        r"\bplayoff", r"\bchampionship", r"\bsuper bowl", r"\bworld cup",
    ],
    "financial": [
        r"\bfed\b", r"\brate", r"\bcpi\b", r"\bgdp\b", r"\bearnings",
        r"\bipo\b", r"\bs&p", r"\bnasdaq", r"\bbitcoin", r"\bcrypto",
    ],
    "geo": [
        r"\bsanction", r"\bceasefire", r"\btreaty", r"\bborder",
        r"\binvasion", r"\bnato\b", r"\bun resolution",
    ],
    "scientific": [
        r"\bfda\b", r"\bphase\b", r"\btrial", r"\bvaccine",
        r"\blaunch", r"\brocket", r"\bspacex", r"\bnasa",
    ],
}


def classify_regime(question: str) -> str:
    q = question.lower()
    for regime, patterns in _REGIME_KEYWORDS.items():
        for p in patterns:
            if re.search(p, q):
                return regime
    return "other"


class PolymarketAdapter:
    """Adapter to Polymarket's public APIs.

    Pass `cache_dir=None` for an in-memory dict; pass a path to persist between
    runs. For backtest reproducibility, **always use a persistent cache**.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = "data_cache/polymarket",
        gamma_url: str = _GAMMA,
        clob_url: str = _CLOB,
        timeout: int = _DEFAULT_TIMEOUT,
        rate_limit_sleep: float = 0.2,
    ) -> None:
        self.cache = DiskCache(cache_dir) if cache_dir else _NullCache()
        self.gamma = gamma_url.rstrip("/")
        self.clob = clob_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_sleep = rate_limit_sleep

    # ------------------- public contract -------------------

    def load_markets(
        self,
        since: date,
        until: date,
        max_markets: int = 500,
        timeline_steps: int = 4,
        only_binary: bool = True,
    ) -> list[Market]:
        """Fetch resolved markets that closed in [since, until]."""
        raws = self._fetch_resolved_markets(since, until, max_markets)
        markets: list[Market] = []
        for r in raws:
            if only_binary and len(r.outcomes) != 2:
                continue
            mkt = self._build_market(r, timeline_steps)
            if mkt is None:
                continue
            markets.append(mkt)
        return markets

    def load_resolutions(self, markets: Iterable[Market]) -> ResolutionOracle:
        oracle = ResolutionOracle()
        for m in markets:
            outcome = self._infer_outcome(m.market_id)
            if outcome is None:
                continue
            oracle.record(m.market_id, outcome)
        return oracle

    def load_news(
        self,
        markets: Iterable[Market],
        source: "NewsSourceProtocol",
        per_market_limit: int = 8,
    ) -> Corpus:
        """Delegate news fetch to a NewsSource (e.g., Tavily). Returns a
        cutoff-clean Corpus (each doc validated to have a real published_at)."""
        from clio.data.news_sources import NewsSourceProtocol  # noqa
        from clio.data.date_validator import validate_published_date  # noqa

        corpus = Corpus()
        seen_doc_ids: set[str] = set()
        for m in markets:
            window_start = m.observed_at - timedelta(days=30)
            window_end = m.closes_at
            docs = source.search(
                query=m.question,
                published_after=window_start,
                published_before=window_end,
                max_results=per_market_limit,
            )
            for raw in docs:
                v = validate_published_date(
                    claimed_date=raw.published_at,
                    url=raw.url,
                    html=raw.raw_html,
                )
                if v.confidence == "rejected" or v.consensus_date is None:
                    log.debug("rejected doc %s: %s", raw.url, v.reason)
                    continue
                if v.consensus_date >= m.closes_at:
                    continue
                doc = raw.to_document(consensus_date=v.consensus_date, market_id=m.market_id)
                if doc.doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc.doc_id)
                corpus.add(doc)
        return corpus

    # ------------------- internals -------------------

    def _fetch_resolved_markets(
        self, since: date, until: date, max_markets: int
    ) -> list[PolymarketRawMarket]:
        out: list[PolymarketRawMarket] = []
        offset = 0
        page_size = 100
        while len(out) < max_markets:
            cache_key = ("gamma_markets", since.isoformat(), until.isoformat(), offset, page_size)
            cached = self.cache.get(cache_key)
            if cached is None:
                qs = urllib.parse.urlencode(
                    {
                        "closed": "true",
                        "limit": page_size,
                        "offset": offset,
                        "order": "endDate",
                        "ascending": "false",
                        "end_date_min": since.isoformat(),
                        "end_date_max": until.isoformat(),
                    }
                )
                url = f"{self.gamma}/markets?{qs}"
                cached = _http_get_json(url, timeout=self.timeout)
                self.cache.put(cache_key, cached, ttl_seconds=_DEFAULT_TTL)
                time.sleep(self.rate_limit_sleep)

            if not cached:
                break

            for entry in cached:
                raw = self._parse_market(entry)
                if raw is None:
                    continue
                if raw.end_date < since or raw.end_date > until:
                    continue
                if not raw.closed:
                    continue
                out.append(raw)
                if len(out) >= max_markets:
                    break

            if len(cached) < page_size:
                break
            offset += page_size
        return out

    def _parse_market(self, entry: dict[str, Any]) -> PolymarketRawMarket | None:
        try:
            outcomes = entry.get("outcomes")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            outcome_prices = entry.get("outcomePrices")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            outcome_prices = [float(x) for x in outcome_prices or []]
            token_ids = entry.get("clobTokenIds")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            token_ids = list(token_ids or [])

            start = _parse_iso_date(entry.get("startDate") or entry.get("createdAt"))
            end = _parse_iso_date(entry.get("endDate") or entry.get("closedTime"))
            if not start or not end:
                return None

            return PolymarketRawMarket(
                market_id=str(entry.get("id") or entry.get("conditionId")),
                condition_id=str(entry.get("conditionId") or ""),
                question=str(entry.get("question") or "").strip(),
                slug=str(entry.get("slug") or ""),
                outcomes=list(outcomes or []),
                outcome_prices=outcome_prices,
                token_ids=token_ids,
                start_date=start,
                end_date=end,
                closed=bool(entry.get("closed")),
                raw=entry,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.debug("skip malformed market: %s", exc)
            return None

    def _build_market(
        self, raw: PolymarketRawMarket, timeline_steps: int
    ) -> Market | None:
        if not raw.token_ids:
            return None
        # YES is conventionally the first outcome on Polymarket binary markets,
        # but we double-check by name to be safe.
        yes_idx = self._yes_index(raw.outcomes)
        if yes_idx is None:
            return None
        yes_token = raw.token_ids[yes_idx]

        # Build timeline: timeline_steps evenly spaced dates between start and
        # end (exclusive of end).
        days = (raw.end_date - raw.start_date).days
        if days < timeline_steps:
            return None
        step_days = days / max(1, timeline_steps)
        timeline = tuple(
            raw.start_date + timedelta(days=int(i * step_days))
            for i in range(timeline_steps)
        )

        prices_by_date = self._fetch_price_history(yes_token)
        market_prices: dict[date, float] = {}
        for d in timeline:
            p = self._price_at_or_before(prices_by_date, d)
            if p is None:
                # If no price exists by this date, skip the market — we don't
                # want to backfill mid prices.
                return None
            market_prices[d] = p

        return Market(
            market_id=raw.market_id,
            question=raw.question,
            regime=classify_regime(raw.question),
            observed_at=raw.start_date,
            closes_at=raw.end_date,
            timeline=timeline,
            market_prices=market_prices,
        )

    @staticmethod
    def _yes_index(outcomes: list[str]) -> int | None:
        for i, o in enumerate(outcomes):
            if str(o).strip().lower() == "yes":
                return i
        # Non-Yes/No binary; default to first.
        if len(outcomes) == 2:
            return 0
        return None

    def _fetch_price_history(self, token_id: str) -> dict[date, float]:
        cache_key = ("clob_prices_history_daily", token_id)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return {date.fromisoformat(k): float(v) for k, v in cached.items()}

        qs = urllib.parse.urlencode({"market": token_id, "interval": "1d", "fidelity": "1440"})
        url = f"{self.clob}/prices-history?{qs}"
        try:
            payload = _http_get_json(url, timeout=self.timeout)
        except PolymarketFetchError:
            return {}
        time.sleep(self.rate_limit_sleep)

        history = payload.get("history") if isinstance(payload, dict) else payload
        out: dict[date, float] = {}
        for point in history or []:
            ts = point.get("t")
            p = point.get("p")
            if ts is None or p is None:
                continue
            d = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            out[d] = float(p)
        self.cache.put(cache_key, {k.isoformat(): v for k, v in out.items()}, ttl_seconds=None)
        return out

    @staticmethod
    def _price_at_or_before(history: dict[date, float], target: date) -> float | None:
        if not history:
            return None
        candidates = [d for d in history if d <= target]
        if not candidates:
            return None
        return history[max(candidates)]

    def _infer_outcome(self, market_id: str) -> int | None:
        cache_key = ("gamma_market_detail", market_id)
        cached = self.cache.get(cache_key)
        if cached is None:
            url = f"{self.gamma}/markets/{market_id}"
            try:
                cached = _http_get_json(url, timeout=self.timeout)
            except PolymarketFetchError:
                return None
            self.cache.put(cache_key, cached, ttl_seconds=None)
            time.sleep(self.rate_limit_sleep)

        outcome_prices = cached.get("outcomePrices") if isinstance(cached, dict) else None
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except json.JSONDecodeError:
                return None
        if not outcome_prices:
            return None
        prices = [float(x) for x in outcome_prices]
        outcomes = cached.get("outcomes")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                outcomes = None
        yes_idx = self._yes_index(outcomes or [])
        if yes_idx is None:
            return None
        # Resolved binary market: the winner has price ~1, loser ~0.
        # Use 0.99 threshold to be safe.
        if prices[yes_idx] >= 0.99:
            return 1
        if prices[yes_idx] <= 0.01:
            return 0
        return None


# ------------------- protocol forward declaration -------------------


class NewsSourceProtocol:
    """Imported from clio.data.news_sources at runtime; declared here to
    avoid a circular import in the type signature only."""

    def search(
        self,
        query: str,
        published_after: date,
        published_before: date,
        max_results: int,
    ) -> list[Any]:
        raise NotImplementedError


# ------------------- offline-only fallback cache -------------------


class _NullCache:
    def get(self, key_parts):
        return None

    def put(self, key_parts, value, ttl_seconds=None):
        pass

    def __len__(self):
        return 0

    def clear(self):
        pass


# ------------------- offline replay constructor -------------------


def adapter_from_recorded_dump(dump_path: str | Path) -> PolymarketAdapter:
    """Build an adapter that reads only from a pre-recorded JSON dump.

    The dump is structured as:
        {"markets": [...gamma response...],
         "details": {"<market_id>": {...gamma detail response...}, ...},
         "prices":  {"<token_id>":  {"<iso_date>": <price>, ...}, ...}}

    Useful for sharing a backtest dataset without committing rate-limited fetch.
    """
    p = Path(dump_path)
    with open(p) as f:
        dump = json.load(f)

    cache_dir = p.parent / f"{p.stem}_cache"
    cache_dir.mkdir(exist_ok=True)
    adapter = PolymarketAdapter(cache_dir=cache_dir)

    # Pre-warm cache from dump.
    if dump.get("markets"):
        adapter.cache.put(
            ("gamma_markets_offline_dump",),
            dump["markets"],
            ttl_seconds=None,
        )
    for mid, detail in (dump.get("details") or {}).items():
        adapter.cache.put(("gamma_market_detail", mid), detail, ttl_seconds=None)
    for tid, prices in (dump.get("prices") or {}).items():
        adapter.cache.put(("clob_prices_history_daily", tid), prices, ttl_seconds=None)

    # Patch fetch to read from offline dump.
    markets_dump = dump.get("markets") or []

    def _offline_fetch(self, since: date, until: date, max_markets: int):
        out: list[PolymarketRawMarket] = []
        for entry in markets_dump:
            raw = self._parse_market(entry)
            if raw is None:
                continue
            if raw.end_date < since or raw.end_date > until:
                continue
            if not raw.closed:
                continue
            out.append(raw)
            if len(out) >= max_markets:
                break
        return out

    import types
    adapter._fetch_resolved_markets = types.MethodType(_offline_fetch, adapter)  # type: ignore[method-assign]
    return adapter
