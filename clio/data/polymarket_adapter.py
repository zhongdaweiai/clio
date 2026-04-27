"""Polymarket adapter — interface only.

The MVP intentionally does not ship a working Polymarket fetcher. The reasons:

1. Polymarket's GraphQL API and CLOB endpoints change. A fetcher in MVP is just
   bit rot waiting to happen.
2. The hard work isn't fetching — it's *date-tagging news* with proper cutoffs.
   Polymarket gives you market data; you also need a contemporaneous news
   archive (e.g., Common Crawl + Wayback Machine) to populate the corpus
   without leakage. That is a substantial separate project.
3. The architectural contract this adapter expresses is the entire point:
   anything that produces (Markets, ResolutionOracle, Corpus) plugs in.

This file documents the contract and provides a `NotImplementedError` shim.
"""

from __future__ import annotations

from datetime import date

from clio.frozen.corpus import Corpus
from clio.frozen.harness import Market
from clio.frozen.oracle import ResolutionOracle


class PolymarketAdapter:
    """Plug here when you want to load real markets.

    Implementation notes for the future builder:

    - `load_markets(since, until)` should return only markets that are *fully
      resolved* in [since, until], i.e. any backtest window must be in the past.
    - For each market, build a `timeline` of as_of dates. A reasonable default
      is daily snapshots from `observed_at` to `closes_at - 1`.
    - `market_prices` at each as_of must come from Polymarket's historical
      midpoint at that date — NOT the close price (look-ahead).
    - `load_news(corpus)` must populate `Corpus` with documents whose
      `published_at` is correct. If you are using a search API like Tavily,
      you must filter by published_at server-side.
    """

    def load_markets(self, since: date, until: date) -> list[Market]:
        raise NotImplementedError(
            "PolymarketAdapter.load_markets is not implemented in the MVP. "
            "See module docstring for the contract."
        )

    def load_resolutions(self, markets: list[Market]) -> ResolutionOracle:
        raise NotImplementedError("PolymarketAdapter.load_resolutions not implemented.")

    def load_news(self, markets: list[Market]) -> Corpus:
        raise NotImplementedError("PolymarketAdapter.load_news not implemented.")
