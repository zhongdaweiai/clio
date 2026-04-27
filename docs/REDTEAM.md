# Red Team

The Red Team is Clio's adversarial validation layer. Its purpose is to make
strategies *prove they survive* before any of them gets near real capital.

A strategy that passes a backtest is not a strategy that should be deployed.
Backtest performance can come from:

- Look-ahead leakage (despite our cutoff defenses)
- Spurious correlations in a small holdout
- Lucky alignment between the strategy's prior and the holdout's outcome distribution
- A regime that the strategy happens to handle, ignoring regimes it doesn't

The Red Team systematically attacks each of these.

## Three components

### 1. Subset attack — `clio.red_team.subset_attack`

Mines the holdout for **feature buckets** where the strategy systematically
underperforms. Buckets are computed across:

- regime (election / sport / financial / geo / scientific / other)
- forecast band (low / lo-mid / hi-mid / high)
- market price band (same)
- evidence volume (0 / 1 / 2-3 / 4+)
- agreement with market (tight / moderate / strong disagreement)

For each bucket with `n >= min_bucket_size`:
1. Compute bucket Brier and per-market P&L.
2. Bootstrap p-value: how often does a random same-size subset of the holdout
   degrade by at least this much? (one-sided, higher-Brier-is-worse)
3. Report buckets where degradation > threshold AND p < significance_level.

Output is a ranked list of `BlindSpot`s, each with sample market IDs so you
can drill down into specific failures.

### 2. Perturbation — `clio.red_team.perturbation`

For each market, replays the forecast under three counterfactual mutations:

| Perturbation | What it does | What we learn |
|---|---|---|
| `drop_strongest` | Remove the evidence doc with the largest \|log LR\| | Is the strategy actually using its signal? |
| `flip_strongest` | Replace strongest LR with `1/LR` | Is the strategy too anchored on prior, or too signal-dominated? |
| `inject_adversary` | Add a counter-evidence doc at the LR ceiling | Does the strategy degrade gracefully under noise? |

Aggregates: mean and max sensitivity across the holdout. The "fragility score"
is the mean flip sensitivity.

A healthy strategy lives in a sweet spot:

- **Too rigid** (`fragility < 0.02`): evidence has almost no effect — likely a
  base-rate-only forecaster in disguise, or the calibrator is squashing
  posterior variance.
- **Too volatile** (`fragility > 0.50`, or any single flip > 0.70): one bad
  document can move the posterior by more than half. Adversarial news will
  destroy this.
- **Healthy**: fragility 0.05 – 0.30. Evidence matters; no single doc owns
  the answer.

### 3. Gate — `clio.red_team.gate`

Combines subset-attack and perturbation results into a binary
**promotion decision**. Per the constitution, no strategy may move past T1
(paper-trade tier) without a passing gate.

Failure modes the gate catches:

- Brier lift over baseline below `min_brier_lift_vs_baseline` (default 0.005)
- Any blind spot above `max_blindspot_degradation` not covered by an
  `excluded_slots` declaration
- Fragility outside `[min_flip_sensitivity, max_flip_sensitivity]`
- Any single-doc influence above `max_single_doc_influence`
- Any per-regime ECE above `max_regime_ece`

The gate report enumerates **all** failures, not just the first. You see
everything that's broken at once.

## Usage

```bash
clio red-team --markets runs/2024_markets.json \
              --news runs/2024_news.jsonl \
              --train-size 100
```

Sample output (degenerate calibrator from N=1 train — Red Team correctly
rejects it):

```
=== Red Team report ===

holdout n=3  Brier=0.6667  PnL/market=-0.0311

No significant blind spots above the threshold.
Perturbation: drop=0.000  flip=0.000  inject=0.000  max_flip=0.000

GATE:  FAIL
  ✗ Brier lift over baseline -0.5952 < required +0.0050
  ✗ strategy too rigid: mean flip sensitivity 0.000 < 0.020 — evidence is barely affecting predictions
  ✗ regime 'financial' ECE 1.000 > cap 0.070
  ✗ regime 'scientific' ECE 1.000 > cap 0.070
```

## Programmatic API

```python
from clio.red_team import (
    run_subset_attack,
    run_perturbation,
    evaluate_gate,
    GateThresholds,
)

# Subset attack:
report = run_subset_attack(
    run=backtest_run,
    regimes=[m.regime for m in holdout_markets],
    n_evidence_per_market=[len(f.evidence_doc_ids) for f in run.forecasts],
    min_bucket_size=10,
    min_brier_degradation=0.03,
    significance_level=0.10,
)

# Perturbation:
pert = run_perturbation(strategy, holdout_markets, corpus)

# Gate:
decision = evaluate_gate(
    score=run.score,
    baseline_score=baseline_run.score,
    subset_attack=report,
    perturbation=pert,
    regime_scores=run.regime_breakdown,
    excluded_slots={"regime=geo"},  # opt out of known-bad regimes
    thresholds=GateThresholds(min_brier_lift_vs_baseline=0.01),
)
if not decision.passed:
    for f in decision.failures:
        log.warn(f)
```

## Why a separate "Red Team" instead of just better tests

Tests check **invariants** — properties that should always hold. The Red
Team checks **claims** — that this strategy beats the baseline, on this
holdout, in a way that's robust to perturbation. These are different things.

The Red Team's output is a *report*, not a pass/fail signal alone. The report
is meant to be read — by an iteration loop, by a human reviewer, by the
constitution amendment process. Failure modes become tickets for the next
iteration.

The right mental model: tests are the immune system; the Red Team is a
penetration tester you hire to attack your own system before someone else
does.

## Future work (designed but not built)

- **Code-aware Red Team**: have an LLM read the strategy code and propose
  weakness conjectures, then test them automatically. Currently the Red Team
  is feature-based, which limits it to weaknesses we already know how to
  bucket.
- **Cross-strategy futarchy**: have candidate strategies bet against each
  other on un-resolved live questions, producing a high-frequency relative
  signal that's faster than waiting for resolution.
- **Time-windowed regime drift**: detect if the strategy's edge is decaying
  over rolling time windows, not just the static holdout split.
