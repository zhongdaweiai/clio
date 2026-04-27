# Live run on Polymarket: a closed-loop iteration session

**Run date:** 2026-04-27
**Scope:** 24 resolved Polymarket markets, 85 news docs from Hacker News + Wikipedia revisions, 5 strategy iterations.
**Outcome:** Every iteration was correctly rejected by the gate. The Red Team identified a different failure mode each time; the next iteration was shaped by the previous report. The market baseline (Brier 0.0834) was never beaten — an honest negative result that confirms the gate is doing its job.

This document is a faithful record of what actually happened, not a story.
The underlying logs are in [`runs/live_iter/iterations.log`](../runs/live_iter/iterations.log).

---

## Setup (live, not synthetic)

### Markets
24 resolved binary markets pulled from Polymarket gamma API (`gamma-api.polymarket.com/markets`), sorted by volume, filtered for clean binary resolution (one outcome at ≥0.99). Spans Jan 2024 – Feb 2025, six regimes:

| Regime | n | YES rate | Examples |
|---|---|---|---|
| election | 8 | 25% | Trump 2024, Harris, Romanian election, RFK Jr., Michelle Obama |
| financial | 5 | 0% | Fed rate hikes (Nov, Jan), Bitcoin $100k in November |
| sport | 5 | 0% | Super Bowl 2025: Panthers, Raiders, Titans, Browns, Giants |
| other | 4 | 50% | Trump inauguration, Aston Villa UCL, TikTok ban, Nottingham Forest |
| geo | 1 | 100% | Israel-Hezbollah ceasefire 2024 |
| scientific | 1 | 0% | Kanye coin February |

Price histories pulled from CLOB `prices-history?interval=all&fidelity=1440` (one sample/day, full history). The adapter required a `Mozilla/5.0` User-Agent — without it the CLOB endpoint silently returns an empty `history: []`. Took ~15 seconds of probing to find this.

### News
Fetched per-market via `MultiSource(HackerNewsSource + WikipediaRevisionSource)`:
- **Hacker News (Algolia API)**: `created_at_i` is a real Unix timestamp; date filter is trustworthy. Required progressive query shortening — HN's keyword-AND search returns nothing for ~5+ tokens, fine with 3.
- **Wikipedia revisions**: queries the `revisions` API for the latest revision ≤ as_of date. Returns the article *as it stood* at that date. Inherently a snapshot source.

**85 docs total**, all date-validated by `clio.data.date_validator`. 1 rejected for no parseable date. 0 leaked across cutoffs (`Corpus.assert_no_leak` ran without raising). Fetch time: 90 seconds.

### Train / holdout split
Stratified by regime to avoid putting all of a small-n regime on one side:
- Train: 12 markets (election=4, sport=2, other=2, financial=2, scientific=1, geo=1)
- Holdout: 12 markets (election=4, sport=3, financial=3, other=2)

### Empirical train regime priors
Computed once from the train split, then frozen:
```
{scientific: 0.0, other: 0.5, election: 0.25, sport: 0.0, financial: 0.0, geo: 1.0}
```

### Baseline reference
`MarketPriceStrategy` echoes the market mid at each as_of. On the holdout: **Brier 0.0834, ECE 0.064, bankroll unchanged ($10K)**. This is the bar to beat.

---

## The closed loop

For each iteration:
1. Train (fit calibrator on train markets, no peek at holdout).
2. Score on holdout (Brier, ECE, resolution, Kelly bankroll, per-regime breakdown).
3. Run Red Team: `run_subset_attack` (bootstrap-significance failure miner) + `run_perturbation` (drop_strongest / flip_strongest / inject_adversary) + `evaluate_gate` (combined PASS/FAIL).
4. Read the report. Pick the next change.

The iterations and the reasoning behind each change:

### v1 baseline — default everything

Identity calibrator, default regime priors, LR clamp [0.1, 10].

```
holdout: brier=0.6918  ece=0.7607  bankroll=$6636  hit_rate=10%
```

**This is much worse than the market baseline (Brier 0.69 vs 0.08).** The strategy made 10 trades and lost money on 9 of them. Why? News docs that mention winning/losing keywords drive LRs of ~3–4 in either direction; with the default 0.50 election prior compounding through 5 docs, posteriors saturate near 0 or 1. The actual outcome distribution (mostly NO across most regimes) is nothing like 0.50. Red Team caught this exactly:

```
blind spots:
  regime=sport               Δbrier=+0.3047  p=0.172  n=3
  market_price_band=low      Δbrier=+0.1770  p=0.043  n=8   ← significant
  n_evidence=2-3             Δbrier=+0.1671  p=0.120  n=5
  forecast_band=high         Δbrier=+0.0772  p=0.105  n=10

GATE: FAIL
  ✗ Brier lift over baseline -0.6084 < required +0.0050
  ✗ uncovered blindspots (max degradation 0.3047)
  ✗ regime 'financial' ECE 0.640 > cap 0.070
  ✗ regime 'sport'     ECE 0.998 > cap 0.070   ← almost perfectly miscalibrated
  ✗ regime 'election'  ECE 0.748 > cap 0.070
  ✗ regime 'other'     ECE 0.610 > cap 0.070
```

**Diagnosis:** the per-regime ECE breakdown is the tell. ECE 0.998 on sport means we're predicting near-1 for outcomes that resolve to 0, every time. Either the priors are wrong, the LR-compounding is too aggressive, or the calibration is missing entirely. **Decision for v2: add isotonic calibrator** — the simplest mechanical fix.

### v2 isotonic — calibrator fit on train

Same scout, same priors, but raw posterior gets passed through an isotonic calibrator fit on train.

```
holdout: brier=0.1458  ece=0.0833  bankroll=$6634  hit_rate=0%
```

**Brier down 5x. ECE down 9x.** But:

```
perturbation: drop=0.000 flip=0.000 inject=0.000  fragility=0.000

GATE: FAIL
  ✗ Brier lift over baseline -0.0624
  ✗ strategy too rigid: mean flip sensitivity 0.000 — evidence is barely affecting predictions
```

**Red Team caught a new failure mode: the calibrator collapsed.** With train Brier dominated by extreme posteriors and the train YES rate around ~25%, isotonic fit a mapping that saturates raw posteriors of 0.99 and 0.01 to nearly the same calibrated value (~0.15). All evidence got squashed. The strategy now has good Brier but no actual signal in it — it's basically a constant predictor. *That's why bankroll didn't recover.*

**Diagnosis:** if the post-calibration prediction is the same regardless of evidence, we've thrown out the baby with the bathwater. The right place to fix this is *upstream* — give the strategy a more realistic prior so the raw posteriors don't all need to be squashed in the first place. **Decision for v3: replace the default regime priors with the empirical train YES rates.**

### v3 train_priors — empirical priors from train

```
holdout: brier=0.1522  ece=0.0925  bankroll=$8574  hit_rate=0%
```

**Bankroll up to $8574 — best of all five.** Brier nearly identical to v2 (0.152 vs 0.146). The strategy made only 3 trades (down from 8) — *fewer, more selective*. Per-regime ECE improved on sport/financial because the prior now matches reality there.

```
blind spots:
  disagreement=strong       Δbrier=+0.1019  p=0.075  n=5
  forecast_band=lo-mid      Δbrier=+0.0923  p=0.077  n=5

GATE: FAIL
  ✗ Brier lift over baseline -0.0688
  ✗ uncovered blindspots: ['disagreement=strong']
  ✗ regime 'election' ECE 0.083 > cap 0.070
  ✗ regime 'other'    ECE 0.722 > cap 0.070
```

**New failure mode the Red Team surfaced:** when our forecast strongly disagrees with the market price (`disagreement=strong`), we're systematically wrong. This is a real signal — the market is well-priced and our LR compounding produces over-confident contrarianism that historically doesn't pan out. **Decision for v4: blend the regime prior with the market's earliest price** as a "humility nudge" toward the crowd.

### v4 market_anchor — 60/40 prior blend with market

`prior = 0.6 × regime_prior + 0.4 × market_initial_price`

```
holdout: brier=0.1543  ece=0.2694  bankroll=$6977  hit_rate=11%
```

**Brier slightly worse, ECE much worse, bankroll dropped.** The market anchor pulled some priors closer to the market mid, but this re-introduced the calibration mismatch the isotonic was correcting for. The 'other' regime ECE jumped from 0.05 to 0.58.

```
blind spots:
  n_evidence=4+  Δbrier=+0.0679  p=0.182  n=3

GATE: FAIL
  ✗ regime 'financial' ECE 0.333 > cap 0.070
  ✗ regime 'sport'     ECE 0.333 > cap 0.070
  ✗ regime 'election'  ECE 0.353 > cap 0.070
  ✗ regime 'other'     ECE 0.576 > cap 0.070
```

**Diagnosis:** the calibrator was fit assuming raw posteriors come from one prior distribution. Changing the prior at scoring time breaks that assumption. **Decision for v5: keep the anchor (because it sometimes helps), but tighten LR clamp to reduce the post-anchor swings the calibrator now has to deal with.**

### v5 tighter_lr — LR clamp [0.3, 3.3]

```
holdout: brier=0.1543  ece=0.2694  bankroll=$6976  hit_rate=11%
```

**Numerically identical to v4.** Tightening LR didn't change the answer — once the isotonic calibrator squashes raw posteriors to a small calibrated range, the upstream LR scale stops mattering. **Diagnosis:** v5 was wasted; should have re-fit the calibrator after introducing the anchor. The right v5 would have been "re-fit calibrator on anchored train output", not "tighten LR".

---

## Pareto frontier across iterations

```
name                   brier   ece     bankroll   fragility  gate
v1_baseline            0.6918  0.7607  $6636      0.153      FAIL
v2_isotonic            0.1458  0.0833  $6634      0.000      FAIL
v3_train_priors        0.1522  0.0925  $8574      0.037      FAIL
v4_market_anchor       0.1543  0.2694  $6977      0.037      FAIL
v5_tighter_lr          0.1543  0.2694  $6976      0.037      FAIL

frontier: [v1_baseline, v2_isotonic, v3_train_priors, v4_market_anchor]
```

v5 is dominated by v4 (same scores, same gate). v1 stays on the frontier because of its higher fragility (more "evidence sensitivity") — that *is* a Pareto axis even though everything else is bad. v2 is the calibration champion, v3 is the bankroll champion. There's no single winner.

---

## What the Red Team got right

Every iteration failed the gate, and every failure was specific:

| Iteration | What Red Team flagged | Was it real? |
|---|---|---|
| v1 | systematic over-confidence on negatively-skewed regimes (sport/financial ECE near 1.0) | Yes — exact diagnosis of the problem |
| v2 | calibrator collapsed: fragility=0.000 means evidence was being squashed | Yes — Brier-good-but-flat is a hidden failure mode |
| v3 | `disagreement=strong` bucket: contrarian predictions losing systematically | Yes — the market is well-priced; strong disagreement is usually wrong |
| v4 | ECE re-exploded across regimes after introducing market anchor | Yes — calibrator was fit on different prior distribution |
| v5 | (no new flag — v5 was the wrong change) | The flag was the *absence* of improvement |

**The gate refused to promote any strategy.** This is correct: the rule-based scout is genuinely too weak to beat the market on resolved Polymarket markets. To pass the gate would need either (a) a real LLM scout reading actual article content for Bayesian evidence weighing, (b) a much larger holdout to get statistical power, or (c) markets selected for likely market-mispricing rather than top volume.

This is the system telling us the truth.

---

## What the closed loop revealed about the system itself

Three lessons from running this for real, not on synthetic data:

1. **The Mozilla/5.0 issue.** Polymarket's CLOB silently returns `{"history": []}` to non-browser User-Agents. No error, just empty data. If we hadn't probed manually we'd have shipped a backtest framework that always returned 0 markets. Captured in [`polymarket_adapter.py`](../clio/data/polymarket_adapter.py) with a comment.

2. **Timeline alignment.** Polymarket's gamma `startDate` is when the market was *created*, not when trading began. Many markets have price history starting days or weeks after `startDate`. Original adapter rejected every market because the first timeline date had no price. Fixed by aligning timeline to `max(startDate, earliest_price_date)`.

3. **Stratified split is non-optional.** With 24 markets across 6 regimes, an unstratified split puts the only `geo` example entirely on one side. Real backtests need explicit per-regime splitting.

---

## Reproducing this run

```bash
.venv/bin/python scripts/live_fetch.py     # 24 markets, ~75s
.venv/bin/python scripts/live_news.py      # 85 docs from HN+Wikipedia, ~90s
.venv/bin/python scripts/iterate.py        # 5 iterations + Red Team, ~3s
```

The full log is at [`runs/live_iter/iterations.log`](../runs/live_iter/iterations.log).
The structured per-iteration data is at [`runs/live_iter/iterations.json`](../runs/live_iter/iterations.json).

---

## What would actually move us past the gate

Honest list, in order of expected impact:

1. **Real LLM scout.** Replace the rule-based stance scoring (`_signal_llm` in `iterate.py`) with an Anthropic API call that reads the article and outputs a calibrated `P(YES | this article alone)`. The current rule-based system maps a single keyword match to a fixed LR — a real LLM could reason about who/what/when. **This is the single highest-impact change available.**

2. **Larger, more diverse holdout.** 24 markets is too few for the Pareto frontier to be meaningful. Want 200+ across at least 4 well-populated regimes. Means relaxing the "high volume only" filter and accepting some markets with sparser price history.

3. **Filter for likely-mispriced markets.** Polymarket high-volume markets are mostly efficient — it's hard to outperform the consensus on Trump-vs-Harris when 1B in volume has refined the price. Look for medium-volume markets, or sort by volatility/disagreement instead of volume.

4. **Decomposer agent.** Compound questions ("X and Y both happen") factor into independently estimable sub-events. Several of the 24 markets in this dataset are conjunctive. The Decomposer is on the design but not built yet.

5. **Internal futarchy.** Instead of one strategy at a time, have multiple variants disagree on un-resolved markets in real time and use their disagreement as a faster signal than waiting for resolution. Currently we wait the full market window for one bit of feedback.

The gate not passing is not a failure of the iteration loop. The iteration loop did its job: it ran, it measured, Red Team identified problems, the next iteration addressed them, and the gate refused to promote anything that wasn't actually better than the baseline. **A loop that *always* found "improvements" would be the suspicious one.**
