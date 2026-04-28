# Clio

A self-evolving prediction market forecasting system.

Clio is the MVP of an architecture I designed in response to the question:
"What does Karpathy's `autoresearch` look like, applied to prediction markets,
without copying its form?" The answer is not a single mutable file with a single
scalar metric. It is a multi-agent, multi-objective, time-evolving harness with
adversarial validation and bankroll-as-ground-truth.

This repo contains the MVP — the smallest version of that design that can run
end-to-end without external API access. The architecture is designed to scale up.

## What's here

```
clio/
  frozen/        # immutable evaluation layer (agents may not modify)
    corpus.py    # date-sharded document store with knowledge-cutoff enforcement
    harness.py   # backtest replay engine
    cost_model.py# spread/slippage/fees
    scoring.py   # Brier, calibration ECE, Kelly P&L, resolution decomposition
    oracle.py    # market resolution lookup
  agents/        # mutable micro-agents (the layer that evolves)
    base.py      # MicroAgent ABC + LLMClient protocol (mockable)
    base_rater.py# reference-class base rate estimation
    news_scout.py# evidence gathering with bounded likelihood ratios
    calibrator.py# post-hoc probability calibration
  memory/        # cross-question learning
    traces.py    # reasoning trace store
    failures.py  # failure-cluster taxonomy
  data/
    synthetic.py # synthetic market generator (used by tests + demo)
    polymarket_adapter.py # stub for real Polymarket archive
  pareto.py      # Pareto frontier computation
  strategy.py    # micro-agent composition
  cli.py         # entrypoint
constitution.md  # the rules under which agents may evolve
tests/           # pytest suite
examples/        # runnable demo
```

## Design

See the design doc in `docs/DESIGN.md`. In one sentence:

> Optimize "the ability to distinguish real edge from overfit fantasy in a
> low-SNR environment", not "predict accuracy".

Ten core moves:

1. **Skill decomposition** — eight independently-evolving micro-agents instead of one monolith.
2. **Pareto frontier** — score vector (Brier, ECE, resolution, Kelly P&L, max DD, regime breadth, ...) replaces a single scalar.
3. **Adversarial backtest (Red Team)** — strategies must survive an opponent that has read their code.
4. **Regime-conditional routing** — election / sport / financial / geopolitical strategies are not averaged.
5. **Internal futarchy** — strategies bet against each other on live un-resolved questions for high-frequency relative signal.
6. **Time-evolving backtest** — predictions are scored at multiple `as_of` times along the question timeline, not once.
7. **Hard knowledge-cutoff enforcement** — corpus layer blocks any document with `published_at >= as_of`. Look-ahead is the #1 self-deception.
8. **Constitution co-evolution** — agents propose methodology amendments; humans merge.
9. **Failure-mode taxonomy** — losses are clustered into named patterns; next iteration targets the largest cluster.
10. **Bankroll is ground truth** — simulated fractional-Kelly bankroll subsumes accuracy + calibration + sizing + risk.

## What's in vs out

**In:**
- Frozen layer: corpus with cutoff enforcement, harness, cost model, scoring.
- 3 micro-agents: Base-rater, News Scout, Calibrator.
- Strategy pipeline composing them.
- Pareto frontier.
- Memory layer (SQLite traces + failure clustering).
- Synthetic market generator (so tests + demo run with no API key).
- **Polymarket adapter** (gamma + clob), with disk cache and offline-replay path. See `docs/POLYMARKET.md`.
- **News pipeline**: Tavily + LocalJSONL sources, **strict published-date validator** (URL + HTML + claim cross-check), `build_corpus` that produces a cutoff-clean Corpus.
- **Red Team agent**: subset attack (feature-bucket failure miner with bootstrap p-values) + perturbation tests (drop/flip/inject) + promotion gate. See `docs/REDTEAM.md`.
- Pytest suite (116 tests) covering scoring math, cutoff enforcement, Pareto, calibration, memory, adapter, news pipeline, date validator, Red Team subset/perturbation/gate, end-to-end.
- CLI: `clio demo`, `clio fetch polymarket`, `clio fetch news`, `clio backtest`, `clio red-team`.

**Out of scope (designed in, not built):**
- Decomposer, Devil's Advocate, Sizer, Regime Classifier agents (interfaces sketched).
- Live trading execution.
- The four-layer (inner / middle / outer / meta) loop scheduler.
- Constitution PR workflow.

## Honest disclosure

The Polymarket and Tavily HTTP adapters are written to the public API contracts
but were not network-tested in the same session they were written. The unit
tests exercise them against recorded fixtures that mirror the real response
shape. **Smoke-test against live before relying on them** — see
`docs/POLYMARKET.md` for the recommended verification.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                # 116 tests, all pass without external services
python -m clio.cli demo               # synthetic end-to-end backtest

# Real Polymarket pipeline (requires network + Tavily key for news):
python -m clio.cli fetch polymarket --since 2024-01-01 --until 2024-04-01 \
                                    --max-markets 100 --out runs/markets.json
python -m clio.cli fetch news --markets runs/markets.json --source tavily \
                              --tavily-key $TAVILY_API_KEY \
                              --out runs/news.jsonl --strict
python -m clio.cli backtest --markets runs/markets.json --news runs/news.jsonl \
                            --train-size 50
python -m clio.cli red-team --markets runs/markets.json --news runs/news.jsonl \
                            --train-size 50

# Offline (replay from a recorded dump — no network):
python -m clio.cli fetch polymarket --from-dump tests/fixtures/polymarket_dump.json \
                                    --since 2023-01-01 --until 2025-01-01 \
                                    --out runs/markets.json
python -m clio.cli fetch news --markets runs/markets.json --source jsonl \
                              --jsonl-path tests/fixtures/news_sample.jsonl \
                              --out runs/news.jsonl
```

## Status

MVP + first real-data adapter + adversarial validation layer + first live closed-loop run.

Architecture-complete; live trading and additional micro-agents are the next two milestones.

## Live runs

**Two closed-loop iteration sessions on real Polymarket data are documented:**

### v1: high-volume markets — gate correctly refused promotion ([`docs/LIVE_RUN.md`](docs/LIVE_RUN.md))
24 high-volume resolved markets (Trump-Harris, Super Bowl, etc.), 5 strategy iterations. Red Team refused to promote any variant — they all underperformed the market baseline because top-volume markets are too efficient to beat with rule-based reasoning. Honest negative result.

### v2: mid-volume markets — real, robust edge demonstrated ([`docs/LIVE_RUN_V2.md`](docs/LIVE_RUN_V2.md))
94 mid-volume resolved markets ($200K – $50M USD volume), 8 strategy variants, 30-seed stability test, temporal-split Red Team. **v6_shrink_typed produces edge over market baseline:**
- ΔBrier −0.0026 (better) in **28/30 random seeds**
- Positive PnL in **26/30 random seeds**
- Under temporal split (train past, test future): bootstrap **95% CI on PnL = [+0.024, +1.324]**, strictly positive

The mechanism is interpretable: shrink event-type market prices toward the empirical event base rate (13% YES), trade only when |edge| > 5%. Fades longshots that the market over-prices. Wins on Yoon-out-of-office ✗, Polish-presidential-candidate ✗; loses on Mark-Carney-PM ✓ (market was right, we faded, it happened).

```bash
# v1 (high-volume markets):
.venv/bin/python scripts/live_fetch.py
.venv/bin/python scripts/live_news.py
.venv/bin/python scripts/iterate.py

# v2 (mid-volume, edge demonstrated):
.venv/bin/python scripts/live_fetch_v2.py
.venv/bin/python scripts/iterate_v2.py        # single seed
.venv/bin/python scripts/iterate_v3.py        # 30-seed stability
.venv/bin/python scripts/iterate_v4_redteam.py # temporal split + bootstrap
```
