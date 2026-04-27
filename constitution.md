# Clio Constitution

The constitution defines the rules under which Clio's micro-agents may evolve.
Two layers:

- **Red lines** — only humans may edit. Any agent edit to this section is auto-reverted.
- **Methodology** — agents may submit amendments via PR; humans merge.

---

## Red Lines (human-only)

1. The frozen layer (`clio/frozen/`) is immutable to agents. Any commit modifying it
   without `[human]` in the message is rejected by CI.
2. Knowledge cutoff is enforced at the corpus layer. No agent may bypass it.
3. Live capital allocation requires explicit human approval per tier transition.
4. No strategy may be promoted past T1 (paper trade) without:
   - holdout Brier improvement over baseline ≥ 0.005
   - Red Team adversarial pass
   - calibration ECE < 0.07 across all regimes it claims to cover
5. Single-position cap: 5% of bankroll. No exceptions.
6. Daily loss circuit breaker: -3% bankroll halts all trading until human review.

## Methodology (agent-amendable)

These are the current best beliefs. Amend with evidence.

### M1. Base rate first
Every forecast must start from a reference-class base rate before incorporating
news. The Base-rater micro-agent is the source of truth for priors.

### M2. Bayesian updates with bounded LRs
News evidence is incorporated as likelihood ratios with magnitude clipped to
`[0.1, 10]` to avoid pathological compounding from over-confident NLI.

### M3. Calibrate before sizing
Raw posteriors must pass through the Calibrator before any Kelly sizing.
Empirical calibration error has historically been the single biggest failure mode.

### M4. Decompose when natural
Conjunctive questions (X and Y both happen) should be decomposed when sub-events
are conditionally independent enough to estimate separately.

### M5. Pareto, not best
The strategy roster is the Pareto frontier on the score vector. We do not pick
"the best strategy" — we pick the non-dominated set and route by regime.

### M6. Bankroll is the metric
Brier, ECE, and resolution are diagnostic. Simulated fractional-Kelly bankroll
on holdout is the final arbiter. A high-Brier strategy that loses money is dead.
