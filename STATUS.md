# Clio status — 2026-04-28 (end of day)

## What's running autonomously

5 GitHub Actions workflows, all `active`:

| workflow | schedule | does |
|---|---|---|
| `daily-scan.yml` | every day 14:00 UTC | scan open markets, LLM-forecast, commit + email |
| `daily-scan.yml` (high-edge) | inline after scan | extra email if any \|edge\| ≥ 0.30 |
| `resolution-check.yml` | every day 16:00 UTC | check resolved markets, update PnL, email if changed |
| `weekly-recap.yml` | Monday 13:00 UTC | 7-day rollup email |
| `failure-alert.yml` | on failure | emails immediately if any of the above fails |

Email transport: Resend (`onboarding@resend.dev`) → `zhongdawei.ai@gmail.com`. Verified working at 09:04 UTC.

## What's logged so far

- **16 paper-trade signals** for 2026-04-28 in [`paper_trades/2026-04-28.json`](paper_trades/2026-04-28.json), URLs corrected to use Polymarket parent event slugs
- **0 resolved trades** yet — first resolutions expected 2026-04-30 (Bitcoin $80K April, US x Iran 4-30 meeting, etc)
- **710 news docs** for the v3 dataset cached in [`runs/live_iter/news_v3.jsonl`](runs/live_iter/news_v3.jsonl)
- **638 LLM forecasts** cached in [`runs/live_iter/llm_forecast_cache.json`](runs/live_iter/llm_forecast_cache.json) (~$5 worth of API)

## What you should see by tomorrow morning

Tomorrow ~14:00 UTC the cron auto-fires for the first time without me. Expected outcomes:

- New file: `paper_trades/2026-04-29.json` with the next day's signals
- New email: subject `[Clio] 2026-04-29 — N signals from K markets ($X.XX)`
- Possibly: high-edge alert email if the LLM finds a new ≥0.30 edge
- `paper_trades/ALL_SIGNALS.md` gets a new chronological section appended

By 2026-04-30 (Wed):
- First few markets resolve (the "by April 30" ones)
- `resolution-check.yml` populates `paper_trades/SUMMARY.md` + sends first PnL email

## Cost projection

```
Anthropic API:    ~$0.10/day  (Sonnet 4.6 on ~80 markets)  → ~$36/year
Resend:           free tier (100 emails/day, we send ~5/day)
GitHub Actions:   free for public repo
                  ──────
Total:            ~$36/year
```

## Auto-research directions for the next iteration

The whole project started from "what's autoresearch for prediction markets". Right now the closed loop is offline: backtest → tune → deploy. The next layer is making it **continuously self-improving in production**. Candidate next builds:

### 1. Calibration drift detector
After each resolution, score the LLM's prediction with Brier + log-loss. If realized calibration drifts from backtest expectation by > X over a rolling window, alert + auto-pause new bets.

### 2. Researcher agent
Weekly cron, runs after `weekly-recap.yml`. Reads the past 7 days of resolved trades, calls Claude to identify the **single most actionable parameter change** (e.g., "lower edge_threshold for `field` qtype because field hit rate is 80% on the bottom 25th percentile of edge"). Outputs a PR proposal with evidence.

### 3. A/B fork in production
Run 2-3 strategy variants concurrently with smaller allocations each. Allocate budget proportionally to recent live performance. The best variant wins capital share over time. This is real-time strategy selection, not offline hyperopt.

### 4. Self-modifying constitution
The hand-tuned `(λ, edge_threshold, kelly_fraction)` in the LLM strategy is currently locked. Treat them as agent-mutable: each Sunday, the researcher agent submits a PR with a proposed parameter delta + reasoning, the human merges (or auto-merge if backtest CI passes).

### 5. Cross-market arbitrage detector
Daily scan looks for markets that should be perfectly anti-correlated (e.g., "Trump wins Iowa" and "non-Trump wins Iowa") whose prices don't sum to ~1.00. These are free arb. Currently nothing in the system looks for these.

### 6. Position-aware decisions
Currently the scanner treats each market independently. A real trader would consider: "I already have 12% bankroll on geopolitical NO bets; should I take another 5% on a 13th?". Add correlation/concentration penalties.

### 7. Look-ahead bias audit
Once we have ~20 resolved trades, automatically compare LLM hit rate on:
- Markets where the answer was knowable from training cutoff (likely contaminated)
- Markets that resolved after training cutoff (clean signal)
If clean-signal hit rate < contaminated hit rate by > 10pp, the strategy is mostly leak.

## How to pick up tomorrow

```bash
cd /Users/daweizhong/projects/clio
git pull origin main
cat STATUS.md             # read this file
cat paper_trades/SUMMARY.md  # what's resolved so far
gh run list -R zhongdaweiai/clio --limit 10  # recent automated runs
```

Then ask me to develop one of the items in "Auto-research directions" above (or something else).

## What's clean

- 122 unit tests still pass
- `.env` gitignored, never committed
- All commits on `main` are pushed to GitHub
- Working tree clean
- All workflows `active` and recently confirmed working

You're good to step away. The system runs without anyone touching it.
