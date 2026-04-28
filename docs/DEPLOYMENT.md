# Deployment: live forward paper-trading

Run the strategy on currently-open Polymarket markets, log every signal to
the repo, automatically check resolutions and update PnL. Free hosting via
GitHub Actions.

The first signals are already in [`paper_trades/2026-04-28.json`](../paper_trades/2026-04-28.json) and visible in [`paper_trades/ALL_SIGNALS.md`](../paper_trades/ALL_SIGNALS.md). After the workflows are enabled, the next scan happens automatically tomorrow at 14:00 UTC.

## What this proves (or disproves)

The backtest in [`docs/LLM_RUN.md`](LLM_RUN.md) reports +259% CAGR. Honest
disclosure on that page: Sonnet 4.6 was trained on data through some
cutoff date, so when it forecast historical 2024-2025 markets it may have
already known the answer. **The realized live PnL is the only test that
can't be cheated by training-data leak.**

After ~50 resolved trades, compare:
- Live hit rate vs backtest 69%
- Live CAGR vs backtest 259%
- Live LLM calibration: do bets predicted at 70% YES actually resolve YES 70% of the time?

If realized matches backtest within ~50%, the strategy is real and you
can deploy real capital. If it lags by > 50%, look-ahead was driving the
backtest's edge.

## How it works (end to end)

```
                                 GitHub Actions cron (free)
                                          │
            ┌─────────────────────────────┼───────────────────────────┐
            │                             │                           │
            ▼ daily 14:00 UTC             ▼ daily 16:00 UTC          ▼ ad-hoc
       daily-scan.yml             resolution-check.yml         workflow_dispatch
            │                             │                     (manual run)
            │                             │
   scripts/paper_trade_scan.py    scripts/paper_trade_check.py
            │                             │
            │ 1. Fetch open Polymarket    │ 1. Walk every old daily file
            │    markets (vol≥200K, ≤60d) │ 2. For each rec where end_date passed,
            │ 2. Fetch news (HN+Wiki)     │    fetch resolution from Polymarket
            │ 3. LLM forecast (Sonnet 4.6)│ 3. Compute paper PnL with realistic costs
            │ 4. Save recs with edge≥0.10 │ 4. Update SUMMARY.md + summary.json
            │                             │
            ▼                             ▼
  paper_trades/YYYY-MM-DD.json   paper_trades/resolutions.json
  paper_trades/ALL_SIGNALS.md    paper_trades/SUMMARY.md
            │                             │
            └──────────┬──────────────────┘
                       ▼
                  git commit + push
                       │
                       ▼
            visible on GitHub for everyone
```

## Cost

```
GitHub Actions:               $0 (free tier on public repos)
Anthropic API per daily scan: ~$0.10
Anthropic API per year:       ~$36
GitHub repo storage:          $0 (signals are tiny JSON)
```

A year of fully-automated forward paper trading costs about a coffee.

## Setup (5 minutes)

### Step 1: Rotate the API key you sent me

The key you DM'd is in the chat log. Treat it as exposed. Go to
[console.anthropic.com](https://console.anthropic.com), revoke the old
key, create a new one. Don't share the new one with me.

### Step 2: Add the new key as a GitHub secret

1. Open https://github.com/zhongdaweiai/clio/settings/secrets/actions
2. Click **New repository secret**
3. Name: `ANTHROPIC_API_KEY`
4. Secret: paste your new Anthropic key
5. Click **Add secret**

The secret is encrypted at rest and only injected as an env var into
workflow runs. Never logged, never committed.

### Step 3: Enable workflows

If you forked the repo or this is the first time GitHub Actions runs on
this repo:

1. Open https://github.com/zhongdaweiai/clio/actions
2. If you see "Workflows aren't being run on this repository", click
   **I understand my workflows, go ahead and enable them**
3. In the left sidebar you'll see "Daily Polymarket Paper Trade Scan"
   and "Weekly Resolution Check + PnL Update"

### Step 3.5: Set up email notifications (optional but recommended)

Pick **one** of two paths:

**Path A — Resend (simplest, 30 seconds):**

1. Sign up free at https://resend.com (no credit card)
2. Go to API Keys → create one
3. Add to GitHub repo secrets:
   - `RESEND_API_KEY` = the resend key
   - `NOTIFY_EMAIL` = `zhongdawei.ai@gmail.com`

The workflow uses Resend's `onboarding@resend.dev` test sender (free 100/day,
no domain verification needed).

**Path B — Gmail SMTP (no third party signup):**

1. Go to https://myaccount.google.com/security → 2-Step Verification (turn on)
2. App passwords → "Mail" → "Other (clio)" → generate
3. Add to GitHub repo secrets:
   - `GMAIL_USER` = `zhongdawei.ai@gmail.com`
   - `GMAIL_APP_PASSWORD` = the 16-char app password
   - `NOTIFY_EMAIL` = `zhongdawei.ai@gmail.com`

Either path works. The workflow tries Resend first, then Gmail.

If you set up neither, the system still works fine — signals just stay
in the repo without an email summary. You'd check
[paper_trades/SUMMARY.md](../paper_trades/SUMMARY.md) on GitHub directly.

### Step 4: Trigger the first run manually (optional)

You don't have to wait for tomorrow's 14:00 UTC scheduled run:

1. Open https://github.com/zhongdaweiai/clio/actions/workflows/daily-scan.yml
2. Click **Run workflow** → **Run workflow** (green button on the right)
3. Refresh the page; you'll see a yellow dot turn into either green ✓ or red ✗
4. Click into the run to see logs

If it succeeds, [`paper_trades/`](../paper_trades/) gets a new commit with
today's signals.

### Step 5: Watch SUMMARY.md

Once a few markets resolve (some are within 2-3 days, some 30-60 days),
[`paper_trades/SUMMARY.md`](../paper_trades/SUMMARY.md) will populate
with cumulative PnL, hit rate, and an equity curve.

## Optional: render.com web dashboard

GitHub Actions handles the trading loop. If you want a live web URL that
shows the current signals + PnL, here's the render.com path:

1. Create a new render.com account, link to your GitHub
2. New → Web Service → connect this repo
3. Build command: `pip install -e ".[dev]" anthropic flask`
4. Start command: `python scripts/render_dashboard.py`  (would need to be
   built — small Flask app reading from `paper_trades/`)
5. Plan: free tier (sleeps after 15 min idle, ~30s cold start)

I haven't built `render_dashboard.py` yet because GitHub Actions covers
the actual trading + Markdown is good enough for monitoring. If you want
the web UI, ask and I'll build it (~30 min of work).

## Live trading actual money — the protocol

The system above is **paper** trading. To convert to real:

1. **Wait for ≥50 resolved paper trades.** Look at the live `SUMMARY.md`.
2. **If realized hit rate is within 5pp of backtest 69%**, and live PnL
   reaches at least 50% of backtest expected per-month rate, the edge is
   probably real.
3. **Polymarket account setup**: deposit $10K-100K USDC, set up the CLOB
   API auth (Polymarket has a Python client). The signals already include
   `polymarket_url` for direct manual execution.
4. **Sizing**: match the backtest config:
   - max position: 25% of bankroll
   - gross exposure: 150%
   - Kelly: 1.2× of computed
5. **Hard stop**: if drawdown exceeds 35%, stop trading and audit. The
   backtest had Max DD 35%; if live exceeds that, something is wrong.
6. **Capacity check**: at $1M+ bankroll, position sizes start moving
   markets. Cap deployed capital at $1M for the first 6 months.

## What if something breaks

**The daily scan fails:**
1. Check https://github.com/zhongdaweiai/clio/actions for the failing run
2. If `ANTHROPIC_API_KEY secret not set` — finish Step 2 above
3. If `503 from gamma-api.polymarket.com` — Polymarket is down, will retry tomorrow
4. If `RateLimitError from Anthropic` — your account hit a limit; check
   billing on console.anthropic.com

**The signals look wrong:**
- Click the `polymarket_url` in any rec, verify the question text matches
- Check the LLM `reasoning` field in the JSON — Sonnet 4.6 explains its
  call in one sentence
- If the LLM says 5% on something obviously close to 50%, it may be
  responding to stale news; check the news doc IDs in `news_doc_ids`

**You want to pause:**
- Disable the workflows: GitHub repo → Actions → workflow → ⋯ → Disable
- The repo state stays intact; predictions stop being added

## Telemetry: what to look at week-1, week-4, week-12

**Week 1:**
- Did any markets resolve? Was the LLM right or wrong?
- Are the signals diverse (not all in one regime)?

**Week 4:**
- ~20 resolved trades. Hit rate trending toward backtest 69% or below 50%?
- Any single market move > 30% your bankroll? (sizing too aggressive)

**Week 12 (~50-100 resolved):**
- Live CAGR within 50% of backtest 259% projection?
- Calibration plot: are 70% predictions resolving YES ~70% of the time?
- If yes → real edge, scale to real money cautiously.
- If no → look-ahead bias was the backtest signal; pivot strategy.

This is the only path to honestly knowing whether the strategy works.
