# Polymarket data pipeline

This doc covers how Clio ingests real Polymarket markets and time-correct news,
producing a backtest-ready dataset.

The pipeline has four stages:

```
                   ┌─────────────────────┐
   Polymarket  →   │ PolymarketAdapter   │  →  markets.json
   (gamma+clob)    │  (gamma+clob)       │     (Markets + ResolutionOracle)
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ NewsSource          │
                   │  (Tavily / JSONL)   │
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ DateValidator       │  → reject anything we can't
                   │  (URL/HTML cross-  │     date-verify
                   │   check)            │
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ build_corpus        │
                   │  (cutoff-clean      │
                   │   Documents)        │
                   └─────────────────────┘
                              │
                              ▼
                          news.jsonl  →  BacktestHarness
```

## Honest disclosures

- The HTTP layer in `polymarket_adapter.py` is written to the public Polymarket
  contract (`gamma-api.polymarket.com` and `clob.polymarket.com`) but has not
  been network-tested in the same session it was written. **Run a small
  smoke-test before relying on it for backtests**:

  ```bash
  clio fetch polymarket --since 2024-01-01 --until 2024-02-01 --max-markets 5 \
    --out runs/smoke.json
  ```

  Inspect the output. If field names look stale, the gamma schema has likely
  drifted; update `_parse_market` in the adapter.

- The `TavilyNewsSource` is also contract-true but unverified live in this
  session. Same advice: smoke-test with `--max-results 3`.

- Both adapters pessimistically retry-and-cache. The cache survives across
  runs at `data_cache/` by default. **Always use a persistent cache** for
  reproducibility — a backtest you can't replay is a backtest you can't trust.

## End-to-end usage

```bash
# 1. Fetch resolved markets in a window. Cached in data_cache/polymarket/.
clio fetch polymarket --since 2023-01-01 --until 2024-06-01 \
                      --max-markets 200 \
                      --out runs/2023h2_markets.json

# 2. Fetch news. Each article is date-validated; un-verifiable docs are
#    dropped. --strict requires "high" confidence (URL OR HTML cross-checks
#    the source's claimed date within 2 days).
clio fetch news --markets runs/2023h2_markets.json \
                --source tavily --tavily-key $TAVILY_API_KEY \
                --per-market 8 --strict \
                --out runs/2023h2_news.jsonl

# 3. Backtest. Splits markets train/holdout, fits the calibrator on train,
#    scores all strategies on holdout.
clio backtest --markets runs/2023h2_markets.json \
              --news runs/2023h2_news.jsonl \
              --train-size 100

# 4. Adversarial validation: subset attack + perturbation + promotion gate.
clio red-team --markets runs/2023h2_markets.json \
              --news runs/2023h2_news.jsonl \
              --train-size 100
```

## Replaying from an offline dump

For sharing a frozen backtest dataset (no network needed):

```python
from clio.data.polymarket_adapter import adapter_from_recorded_dump

adapter = adapter_from_recorded_dump("path/to/dump.json")
markets = adapter.load_markets(...)
```

Dump format:

```json
{
  "markets":     [/* gamma /markets array */],
  "details":     {"<market_id>": {/* gamma /markets/<id> */}},
  "prices":      {"<token_id>":  {"YYYY-MM-DD": <mid_price>, ...}}
}
```

`tests/fixtures/polymarket_dump.json` is the canonical example — small but
complete enough for an end-to-end test.

## Knowledge-cutoff guarantees

Three independent layers enforce no-look-ahead:

1. `Corpus.search(query, as_of)` filters with strict `<` on `published_at`.
   Documents on the cutoff day are excluded.
2. `BacktestHarness` calls `Corpus.assert_no_leak` after each search as defense
   in depth.
3. `validate_published_date` rejects any document whose source-claimed date
   cannot be cross-checked, AND requires `consensus_date < market.closes_at`
   when admitting to the corpus.

Tests in `tests/test_corpus_cutoff.py`, `tests/test_date_validator.py`, and
`tests/test_news_pipeline.py` exercise all three.

## Failure modes to expect

| Symptom | Likely cause | Fix |
|---|---|---|
| `PolymarketFetchError: HTTP 429` | Rate limited | Increase `rate_limit_sleep`; the cache will resume |
| Markets missing fields | gamma schema drift | Update `_parse_market` field names |
| Tavily returns docs with wrong dates | News API claims drift | `--strict` mode rejects them; or write a custom `NewsSource` |
| `assert_no_leak` raises | A doc snuck through with bad date | Don't suppress — investigate the doc; this is the system protecting you |
| Strategy beats baseline by huge margin in backtest | Probably look-ahead leak | Run `clio red-team`; check trace samples; tighten validator |
