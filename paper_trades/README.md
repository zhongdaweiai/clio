# Live forward paper trading

This directory is the **honest** test of the strategy. The backtest in
`docs/LLM_RUN.md` reports +259% CAGR / +11.25%/mo, but Sonnet 4.6 was
trained on data through some cutoff date — meaning when I asked it to
forecast historical 2024 markets, it may have already known the answer.
**Look-ahead bias is the most likely explanation if the backtest looks
too good.**

The only way to verify the strategy is real: predict markets *that haven't
resolved yet* and compare to outcomes after they do.

## What's in here

- **`YYYY-MM-DD.json`** — daily scan output. Each file lists every market
  the LLM flagged that day, with prediction, market price, edge,
  reasoning, and `polymarket_url`. Predictions are logged BEFORE outcomes
  are known.
- **`ALL_SIGNALS.md`** — chronological human-readable feed of every signal
  ever issued.
- **`resolutions.json`** — keyed by `(market_id, scan_date)`, holds the
  realized outcome and computed payoff for every signal whose market has
  closed.
- **`SUMMARY.md`** — cumulative PnL, hit rate, equity curve, recent
  resolved trades. **This is the live scoreboard.**
- **`summary.json`** — same data as `SUMMARY.md` but machine-readable.

## How it works

Two GitHub Actions workflows:

1. **`.github/workflows/daily-scan.yml`** — runs every day at 14:00 UTC,
   scans currently-open Polymarket markets, asks Claude Sonnet 4.6 for a
   calibrated probability, saves recommendations whose |edge| ≥ 0.10,
   commits the JSON to this directory.

2. **`.github/workflows/resolution-check.yml`** — runs daily at 16:00 UTC,
   walks every signal ever issued, checks if the market has now closed,
   fetches the resolution from Polymarket gamma API, computes paper PnL
   assuming $100K starting bankroll with 1.5% spread + 30 bps slippage,
   updates `SUMMARY.md`.

Both are free (GitHub Actions cron is free for public repos) and the
LLM API cost is ~$0.10 per daily scan ≈ $36/year.

## What we're measuring

After ~50 resolved signals, compare:

- **Realized hit rate** vs backtest 69% — are we actually right that often?
- **Realized PnL** vs backtest expected — within 30% margin = strategy is real
- **Realized vs LLM-predicted probability calibration** — does the LLM say
  70% on bets that resolve YES 70% of the time?

If realized lags backtest by > 50%, look-ahead was driving the backtest's
edge and this strategy doesn't deploy. If realized matches, deploy real
capital with the same parameters.

## Decision rule (when do we go live?)

```
Number of resolved trades  →  required realized return  →  go live?
       N < 20                       (insufficient data)         No
   N ∈ [20, 50)                   beat market echo + 5 pp        Maybe (small)
   N ≥ 50                         backtest within 50% margin     Yes (capped)
```

Real capital deployment is a separate decision tree. Paper proof comes first.
