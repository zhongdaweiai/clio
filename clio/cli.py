"""Command-line entrypoints.

Subcommands:
  clio demo                 — synthetic end-to-end backtest
  clio fetch polymarket     — fetch resolved Polymarket markets to a dump
  clio fetch news           — fetch news for a market dump via a NewsSource
  clio backtest             — run strategies on a saved markets+news dataset
  clio red-team             — run subset attack + perturbation + gate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from clio.agents.base import MockLLMClient
from clio.agents.base_rater import BaseRater
from clio.agents.calibrator import IdentityCalibrator, IsotonicCalibrator
from clio.agents.news_scout import NewsScout
from clio.data.news_pipeline import build_corpus, read_corpus_jsonl, write_corpus_jsonl
from clio.data.news_sources import LocalJSONLNewsSource, TavilyNewsSource
from clio.data.polymarket_adapter import PolymarketAdapter, adapter_from_recorded_dump
from clio.data.synthetic import SyntheticConfig, generate_synthetic_world
from clio.frozen.harness import BacktestHarness, Market
from clio.frozen.oracle import ResolutionOracle
from clio.pareto import pareto_frontier
from clio.red_team import evaluate_gate, run_perturbation, run_subset_attack
from clio.strategy import BayesianStrategy, MarketPriceStrategy


# ---------------------------- demo ----------------------------


def _seeded_mock_llm() -> MockLLMClient:
    llm = MockLLMClient()
    llm.register(r"document title:.*supports", "0.75")
    llm.register(r"document title:.*contradicts", "0.30")
    llm.register(r"document content:.*supports", "0.75")
    llm.register(r"document content:.*contradicts", "0.30")
    llm.set_default("0.55")
    return llm


def _make_calibrated_strategy(world, train_size: int):
    train_markets = world.markets[:train_size]
    llm = _seeded_mock_llm()
    base_rater = BaseRater(llm)
    scout = NewsScout(llm)
    raw = BayesianStrategy("bayes_raw_for_fit", base_rater, scout)
    h = BacktestHarness(world.corpus, world.oracle, train_markets)
    run = h.run(raw)
    cal = IsotonicCalibrator()
    cal.fit(run.final_probs, run.final_outcomes)
    return BayesianStrategy("bayes_isotonic", base_rater, scout, calibrator=cal)


def _print_score(label: str, score) -> None:
    if score is None:
        print(f"{label:<24}  (no score)")
        return
    e = score.extra
    print(
        f"{label:<24}  Brier={e['brier']:.4f}  ECE={e['ece']:.4f}  "
        f"Resolution={e['resolution']:.4f}  Bankroll=${e['bankroll_final']:.0f}  "
        f"Trades={int(e['n_trades'])}  Hit={e['hit_rate']:.2%}"
    )


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = SyntheticConfig(n_markets=args.n_markets, seed=args.seed, timeline_steps=4)
    world = generate_synthetic_world(cfg)
    holdout = world.markets[args.train_size:]
    if not holdout:
        print("error: train_size >= n_markets", file=sys.stderr)
        return 1

    llm = _seeded_mock_llm()
    base_rater = BaseRater(llm)
    scout = NewsScout(llm)
    strategies = [
        MarketPriceStrategy(),
        BayesianStrategy("bayes_identity", base_rater, scout, IdentityCalibrator()),
        _make_calibrated_strategy(world, args.train_size),
    ]
    harness = BacktestHarness(world.corpus, world.oracle, holdout)
    print(f"\n=== synthetic backtest: {len(holdout)} holdout markets ===\n")
    runs = []
    for s in strategies:
        run = harness.run(s)
        runs.append(run)
        _print_score(s.name, run.score)

    scores = [r.score for r in runs if r.score is not None]
    front = pareto_frontier(scores)
    print(f"\nPareto frontier: {[strategies[i].name for i in front]}")
    return 0


# ---------------------------- fetch polymarket ----------------------------


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_fetch_polymarket(args: argparse.Namespace) -> int:
    if args.from_dump:
        adapter = adapter_from_recorded_dump(args.from_dump)
    else:
        adapter = PolymarketAdapter(cache_dir=args.cache)
    since = _parse_date(args.since)
    until = _parse_date(args.until)
    print(f"fetching resolved Polymarket markets {since} → {until}", file=sys.stderr)
    markets = adapter.load_markets(since, until, max_markets=args.max_markets)
    oracle = adapter.load_resolutions(markets)
    print(f"got {len(markets)} markets, {len(oracle)} resolved", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "markets": [_market_to_dict(m) for m in markets],
        "resolutions": {m.market_id: oracle.lookup(m.market_id) for m in markets if m.market_id in oracle},
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {out}", file=sys.stderr)
    return 0


def _market_to_dict(m: Market) -> dict:
    return {
        "market_id": m.market_id,
        "question": m.question,
        "regime": m.regime,
        "observed_at": m.observed_at.isoformat(),
        "closes_at": m.closes_at.isoformat(),
        "timeline": [d.isoformat() for d in m.timeline],
        "market_prices": {d.isoformat(): p for d, p in m.market_prices.items()},
    }


def _market_from_dict(d: dict) -> Market:
    return Market(
        market_id=d["market_id"],
        question=d["question"],
        regime=d["regime"],
        observed_at=date.fromisoformat(d["observed_at"]),
        closes_at=date.fromisoformat(d["closes_at"]),
        timeline=tuple(date.fromisoformat(s) for s in d["timeline"]),
        market_prices={date.fromisoformat(k): float(v) for k, v in d["market_prices"].items()},
    )


def _load_markets_payload(path: str | Path) -> tuple[list[Market], ResolutionOracle]:
    with open(path) as f:
        payload = json.load(f)
    markets = [_market_from_dict(m) for m in payload["markets"]]
    oracle = ResolutionOracle()
    for mid, outcome in (payload.get("resolutions") or {}).items():
        oracle.record(mid, int(outcome))
    return markets, oracle


# ---------------------------- fetch news ----------------------------


def cmd_fetch_news(args: argparse.Namespace) -> int:
    markets, _oracle = _load_markets_payload(args.markets)
    if args.source == "tavily":
        src = TavilyNewsSource(api_key=args.tavily_key)
    elif args.source == "jsonl":
        if not args.jsonl_path:
            print("--source jsonl requires --jsonl-path", file=sys.stderr)
            return 1
        src = LocalJSONLNewsSource(args.jsonl_path)
    else:
        print(f"unknown source: {args.source}", file=sys.stderr)
        return 1

    corpus, stats = build_corpus(
        markets,
        src,
        per_market_limit=args.per_market,
        require_high_confidence=args.strict,
    )
    print(
        f"queried={stats.queried} fetched={stats.fetched} accepted={stats.accepted} "
        f"rejected_no_date={stats.rejected_no_date} rejected_post_close={stats.rejected_post_close} "
        f"rejected_low_confidence={stats.rejected_low_confidence} duplicates={stats.duplicates}",
        file=sys.stderr,
    )
    write_corpus_jsonl(corpus, args.out)
    print(f"wrote {len(corpus)} validated docs to {args.out}", file=sys.stderr)
    return 0


# ---------------------------- backtest ----------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    markets, oracle = _load_markets_payload(args.markets)
    corpus = read_corpus_jsonl(args.news) if args.news else None
    if corpus is None:
        print("warning: no --news provided, only baseline strategy will run", file=sys.stderr)
        from clio.frozen.corpus import Corpus

        corpus = Corpus()

    llm = _seeded_mock_llm()
    base_rater = BaseRater(llm)
    scout = NewsScout(llm)

    train_size = args.train_size
    train_markets = markets[:train_size]
    holdout_markets = markets[train_size:]

    raw = BayesianStrategy("bayes_raw", base_rater, scout, IdentityCalibrator())
    train_h = BacktestHarness(corpus, oracle, train_markets)
    train_run = train_h.run(raw)
    cal = IsotonicCalibrator()
    cal.fit(train_run.final_probs, train_run.final_outcomes)

    strategies = [
        MarketPriceStrategy(),
        raw,
        BayesianStrategy("bayes_isotonic", base_rater, scout, cal),
    ]

    holdout_h = BacktestHarness(corpus, oracle, holdout_markets)
    print(f"\n=== backtest: {len(holdout_markets)} holdout markets ===\n")
    for s in strategies:
        run = holdout_h.run(s)
        _print_score(s.name, run.score)
    return 0


# ---------------------------- red team ----------------------------


def cmd_red_team(args: argparse.Namespace) -> int:
    markets, oracle = _load_markets_payload(args.markets)
    corpus = read_corpus_jsonl(args.news) if args.news else None
    if corpus is None:
        print("--news is required for red-team", file=sys.stderr)
        return 1

    llm = _seeded_mock_llm()
    base_rater = BaseRater(llm)
    scout = NewsScout(llm)

    train_markets = markets[: args.train_size]
    holdout_markets = markets[args.train_size :]

    raw = BayesianStrategy("bayes_raw", base_rater, scout, IdentityCalibrator())
    train_h = BacktestHarness(corpus, oracle, train_markets)
    train_run = train_h.run(raw)
    cal = IsotonicCalibrator()
    cal.fit(train_run.final_probs, train_run.final_outcomes)
    strategy = BayesianStrategy("bayes_isotonic", base_rater, scout, cal)

    h = BacktestHarness(corpus, oracle, holdout_markets)
    strat_run = h.run(strategy)
    base_run = h.run(MarketPriceStrategy())

    n_evi = []
    regimes = []
    seen_ids: set[str] = set()
    for f in strat_run.forecasts:
        if f.market_id in seen_ids:
            continue
        seen_ids.add(f.market_id)
        n_evi.append(len(f.evidence_doc_ids))
    for m in holdout_markets:
        if m.market_id in seen_ids:
            regimes.append(m.regime)
    while len(n_evi) < len(strat_run.final_probs):
        n_evi.append(0)

    subset_report = run_subset_attack(strat_run, regimes, n_evi)
    pert_report = run_perturbation(strategy, holdout_markets, corpus)
    decision = evaluate_gate(
        score=strat_run.score,
        baseline_score=base_run.score,
        subset_attack=subset_report,
        perturbation=pert_report,
        regime_scores=strat_run.regime_breakdown,
    )

    print("\n=== Red Team report ===\n")
    print(f"holdout n={subset_report.n_holdout}  Brier={subset_report.overall_brier:.4f}  "
          f"PnL/market={subset_report.overall_pnl_per_market:+.4f}")
    if subset_report.blind_spots:
        print("\nBlind spots (Brier degradation, p-value, n):")
        for bs in subset_report.blind_spots[:8]:
            print(
                f"  {bs.slot!s:<26} +{bs.brier_degradation:.4f}  p={bs.bootstrap_p_value:.3f}  "
                f"n={bs.n}  pnl/mkt={bs.bucket_pnl_per_market:+.4f}"
            )
    else:
        print("\nNo significant blind spots above the threshold.")

    pert = pert_report.summary()
    print(
        f"\nPerturbation: drop={pert['mean_drop_sensitivity']:.3f}  "
        f"flip={pert['mean_flip_sensitivity']:.3f}  "
        f"inject={pert['mean_inject_sensitivity']:.3f}  "
        f"max_flip={pert['max_flip_sensitivity']:.3f}"
    )

    print("\nGATE: ", "PASS" if decision.passed else "FAIL")
    for f in decision.failures:
        print(f"  ✗ {f}")
    for n in decision.notes:
        print(f"  · {n}")

    return 0 if decision.passed else 2


# ---------------------------- main ----------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="synthetic end-to-end backtest")
    p_demo.add_argument("--n-markets", type=int, default=60)
    p_demo.add_argument("--train-size", type=int, default=30)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.set_defaults(func=cmd_demo)

    p_fetch = sub.add_parser("fetch", help="fetch external data")
    fetch_sub = p_fetch.add_subparsers(dest="source", required=True)

    p_pm = fetch_sub.add_parser("polymarket", help="fetch resolved Polymarket markets")
    p_pm.add_argument("--since", required=True, help="YYYY-MM-DD")
    p_pm.add_argument("--until", required=True, help="YYYY-MM-DD")
    p_pm.add_argument("--max-markets", type=int, default=200)
    p_pm.add_argument("--cache", default="data_cache/polymarket")
    p_pm.add_argument("--from-dump", default=None, help="offline JSON dump path")
    p_pm.add_argument("--out", default="runs/polymarket.json")
    p_pm.set_defaults(func=cmd_fetch_polymarket)

    p_news = fetch_sub.add_parser("news", help="fetch news for a markets dump")
    p_news.add_argument("--markets", required=True)
    p_news.add_argument("--source", choices=["tavily", "jsonl"], required=True)
    p_news.add_argument("--tavily-key", default=None)
    p_news.add_argument("--jsonl-path", default=None)
    p_news.add_argument("--per-market", type=int, default=8)
    p_news.add_argument("--strict", action="store_true",
                        help="reject any doc with non-high date confidence")
    p_news.add_argument("--out", default="runs/news.jsonl")
    p_news.set_defaults(func=cmd_fetch_news)

    p_bt = sub.add_parser("backtest", help="run strategies on saved data")
    p_bt.add_argument("--markets", required=True)
    p_bt.add_argument("--news", default=None)
    p_bt.add_argument("--train-size", type=int, default=50)
    p_bt.set_defaults(func=cmd_backtest)

    p_rt = sub.add_parser("red-team", help="run adversarial validation + gate")
    p_rt.add_argument("--markets", required=True)
    p_rt.add_argument("--news", required=True)
    p_rt.add_argument("--train-size", type=int, default=50)
    p_rt.set_defaults(func=cmd_red_team)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
