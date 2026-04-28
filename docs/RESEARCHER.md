# The Researcher Agent

The Researcher Agent is Clio's autoresearch loop applied to **production**, not just backtesting. Once a week it reads what the live system actually did, identifies the single most actionable parameter change with statistical evidence, and submits it as a **draft PR for human review**.

This is the Karpathy autoresearch pattern in two dimensions: a strategy that runs, an agent that proposes how to improve it, a human that gates each change. Closed loop without ever auto-merging anything.

## What it actually does

```
┌─────────────────────────────────────────────────────────────────────┐
│ Sunday 18:00 UTC — researcher.yml cron fires                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ scripts/researcher_agent.py                                         │
│                                                                     │
│   1. compute_live_metrics(paper_trades/)                            │
│      → slices: by qtype, edge band, LLM confidence,                 │
│                 side, days held bucket                              │
│   2. if n_resolved < 20: emit "hold" + email, exit                  │
│   3. _read_current_params() from scripts/paper_trade_scan.py        │
│   4. Call Claude Sonnet 4.6 with evidence-required prompt           │
│   5. parse_proposal() → tolerant JSON parser                        │
│   6. validate_proposal()                                            │
│      → whitelisted params only                                      │
│      → bounded ranges                                               │
│      → max delta per step ≤ 50% of range                            │
│      → hold proposals must be empty                                 │
│   7. if propose + valid: create branch, apply change, push, draft PR│
│   8. Email summary with diff + PR link                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Whitelist of allowed parameters

The agent can only mutate these. Anything outside this list is rejected at validation.

| param | range | default | description |
|---|---|---|---|
| `edge_threshold` | [0.05, 0.30] | 0.10 | min \|edge\| for a signal to be issued |
| `max_position_pct` | [0.05, 0.40] | 0.25 | hard cap on single position size |
| `kelly_fraction` | [0.5, 1.5] | 1.2 | Kelly multiplier |
| `notional_floor` | [0.02, 0.20] | 0.10 | min position size before Kelly |
| `min_volume` | [50_000, 1_000_000] | 200_000 | volume floor for inclusion |
| `max_days_remaining` | [14, 120] | 60 | resolution-horizon ceiling |

For each parameter, **one step ≤ 50% of allowed range**. So `edge_threshold` (range 0.25) can change at most 0.125 in a single proposal. Forces the agent to probe, not jump.

## Conservative by design

The agent's system prompt explicitly prefers `hold` over `propose`. Specific evidence requirements:

- **n ≥ 20 resolved trades** total (hard floor — under that, agent always holds)
- **Slice-level n ≥ 8** for any cited slice
- **Slice deviation from overall ≥ 10pp** before it counts as evidence
- **Output is JSON only** — no markdown, no ad-libbing

The validator enforces these as a second line of defense even if the LLM gets creative.

## Output: the proposal artifact

Every run writes `research_proposals/YYYY-MM-DD.json`:

```json
{
  "decision": "propose",
  "summary": "Raise edge_threshold from 0.10 to 0.20 to filter out underperforming mid-band signals.",
  "evidence": [
    "mid_0.10-0.20 band: n=15, hit_rate=6.7%, avg_per_trade_return=-5.44%",
    "high LLM confidence slice: n=8, hit_rate=0.0%",
    "YES side..."
  ],
  "parameter_changes": {
    "edge_threshold": {"old": 0.10, "new": 0.20}
  },
  "expected_impact": "+5pp hit rate, fewer trades but better quality",
  "confidence": "medium",
  "rollback": "revert if hit rate < 55% over next 14 days",
  "proposed_at": "2026-04-28T12:35:26+00:00",
  "branch": "clio-researcher/2026-04-28",
  "pr_url": "https://github.com/zhongdaweiai/clio/pull/N",
  "validation_errors": []
}
```

Plus a snapshot file `research_proposals/snapshot_YYYY-MM-DD.json` with the full live metrics that informed the proposal.

## Decision flow (in production)

```
Resolved trades count
        │
        ├─ < 20  → emit "hold" proposal, no LLM call, email user the count
        │
        └─ ≥ 20  → compute slices → call LLM → parse → validate
                                                          │
                                              ┌───────────┴───────────┐
                                              │                       │
                                          decision=hold          decision=propose
                                              │                       │
                                          email user            create branch
                                          (no PR)               apply diff
                                                                push
                                                                open draft PR
                                                                email user with PR link
```

The user **always gets an email**, whether it's "no change this week" or "I propose X". This is the heartbeat — even silent weeks confirm the agent is running.

## Validation flow (anti-overshoot)

```
LLM output → parse_proposal() → Proposal object
                                      │
                                      ▼
                              validate_proposal()
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
              whitelist?         in range?         delta ≤ 50%?
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      │
                              all pass? → execute
                              any fail? → downgrade to "hold" + log errors in artifact
```

If validation fails, the proposal is **downgraded to hold** and the email surfaces what went wrong. The agent can't sneak a bad change past the gate.

## Tests

`tests/test_researcher.py` covers all the safety properties:

- Strict JSON parsing
- JSON-in-markdown-fence parsing
- Garbage rejection
- Invalid-decision falls back to hold
- Hold-with-changes rejected
- Propose-without-evidence rejected
- Unknown-param rejected
- Out-of-range rejected
- Overshoot (> 50% of range) rejected
- Valid small-step proposal accepted
- `ALLOWED_PARAMS` self-consistency

11 tests, all passing.

## How to test the agent without 20 real trades

```bash
.venv/bin/python scripts/researcher_agent_simulate.py
```

This:
1. Backs up your real `paper_trades/`
2. Generates 60 synthetic resolved trades from the v3 historical dataset
3. Runs the agent in `--dry-run` mode (no git ops, no email)
4. Restores your real `paper_trades/`

Sample output from this morning's test:

```
fabricated 23 files in paper_trades
DRY RUN: skipping git ops, PR creation, and email
live metrics: resolved=21 pending=0 bankroll=$38458
LLM response: {
  "decision": "propose",
  "summary": "Raise edge_threshold from 0.10 to 0.20 to filter out the
   underperforming mid_0.10-0.20 band and concentrate on higher-edge signals.",
  "evidence": [
    "mid_0.10-0.20 band: n=15, hit_rate=6.7%, ...",
    "high LLM confidence slice: n=8, hit_rate=0.0%, ...",
    "YES side: n=..."
  ],
  ...
}
[dry-run] would create branch + PR for: {'edge_threshold': {'old': 0.1, 'new': 0.2}}
```

That's the agent doing its job — citing exact slice statistics, proposing one whitelisted change, staying within the 50% delta bound (0.10 → 0.20 = 0.10 < 0.125 max).

## Where the loop becomes auto-research

This isn't fully autonomous research yet. The PR is **draft**, the human still merges. But the agent now:

- Reads what the system actually did
- Compares it to backtest expectation
- Identifies the largest evidence-backed gap
- Proposes a concrete, bounded change
- Applies the change in code (via regex substitution against `paper_trade_scan.py`)
- Opens a PR with diff + reasoning

The next steps toward fuller autonomy (each one separately gated):

- **Auto-merge if backtest CI passes**: PR triggers a backtest with the new params on the holdout dataset; if it doesn't degrade Brier/PnL, auto-merge.
- **Multi-armed bandit**: instead of replacing params, run the proposed variant in parallel as 30% of capital; promote if it wins over 4 weeks.
- **Researcher dialogue**: agent reads the previous proposal's outcome (was it merged? did it work?) and adjusts confidence going forward.
- **Strategy mutation, not just parameter tuning**: agent proposes new strategy *classes* (e.g., "add a decomposer step for compound questions"), generates the code change, runs the backtest, opens PR.

## Cost

```
LLM call per week:  ~$0.02 (Sonnet 4.6, ~5K input + 600 output tokens)
GitHub Actions:     $0
Resend email:       $0
                    ──────
Total:              ~$1/year
```

## Schedule

Sundays 18:00 UTC. First fire: this Sunday after enough live data accumulates. Until then it'll just emit "hold" emails so you know it's running.

Manual trigger any time: [Actions → Researcher Agent → Run workflow](https://github.com/zhongdaweiai/clio/actions/workflows/researcher.yml).
