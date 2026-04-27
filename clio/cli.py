"""Command-line entrypoint: `clio demo` runs an end-to-end synthetic backtest."""

from __future__ import annotations

import argparse
import sys

from clio.agents.base import MockLLMClient
from clio.agents.base_rater import BaseRater
from clio.agents.calibrator import IdentityCalibrator, IsotonicCalibrator
from clio.agents.news_scout import NewsScout
from clio.data.synthetic import SyntheticConfig, generate_synthetic_world
from clio.frozen.harness import BacktestHarness
from clio.pareto import pareto_frontier
from clio.strategy import BayesianStrategy, MarketPriceStrategy


def _make_calibrated_strategy(world, train_size: int):
    """Train a calibrator on a held-out slice, return a strategy using it."""
    train_markets = world.markets[:train_size]
    train_world = type(world)(
        corpus=world.corpus,
        oracle=world.oracle,
        markets=train_markets,
        truths=world.truths,
    )

    llm = _seeded_mock_llm(world)
    base_rater = BaseRater(llm)
    scout = NewsScout(llm)
    raw_strategy = BayesianStrategy("bayes_raw_for_fit", base_rater, scout)
    harness = BacktestHarness(world.corpus, world.oracle, train_markets)
    run = harness.run(raw_strategy)

    calibrator = IsotonicCalibrator()
    calibrator.fit(run.final_probs, run.final_outcomes)

    return BayesianStrategy(
        "bayes_isotonic", base_rater, scout, calibrator=calibrator
    )


def _seeded_mock_llm(world) -> MockLLMClient:
    """A mock LLM that 'reads' synthetic doc content to produce supportive vs not signals.

    This is a stand-in for a real LLM — its only job is to make the scout's per-doc
    LR roughly aligned with the synthetic world's hidden signal so the demo
    actually shows the system improving.
    """
    llm = MockLLMClient()
    llm.register(r"document title:.*supports", "0.75")
    llm.register(r"document title:.*contradicts", "0.30")
    llm.register(r"document content:.*supports", "0.75")
    llm.register(r"document content:.*contradicts", "0.30")
    llm.set_default("0.55")
    return llm


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
    cfg = SyntheticConfig(
        n_markets=args.n_markets,
        seed=args.seed,
        timeline_steps=4,
    )
    world = generate_synthetic_world(cfg)

    holdout = world.markets[args.train_size:]
    if not holdout:
        print("error: train_size >= n_markets, no holdout left", file=sys.stderr)
        return 1

    llm = _seeded_mock_llm(world)
    base_rater = BaseRater(llm)
    scout = NewsScout(llm)

    strategies = [
        MarketPriceStrategy(),
        BayesianStrategy("bayes_identity", base_rater, scout, IdentityCalibrator()),
        _make_calibrated_strategy(world, args.train_size),
    ]

    harness = BacktestHarness(world.corpus, world.oracle, holdout)

    print(f"\n=== Clio synthetic backtest: {len(holdout)} holdout markets ===\n")
    runs = []
    for s in strategies:
        run = harness.run(s)
        runs.append(run)
        _print_score(s.name, run.score)

    scores = [r.score for r in runs if r.score is not None]
    front = pareto_frontier(scores)
    front_names = [strategies[i].name for i in front]
    print(f"\nPareto frontier: {front_names}")

    print("\n=== Per-regime breakdown for the calibrated strategy ===")
    for regime, sv in sorted(runs[-1].regime_breakdown.items()):
        e = sv.extra
        print(
            f"  {regime:<12} Brier={e['brier']:.4f}  ECE={e['ece']:.4f}  "
            f"n={int(e['n_trades']) + max(0, len(world.markets) - args.train_size)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Run a synthetic end-to-end backtest")
    p_demo.add_argument("--n-markets", type=int, default=60)
    p_demo.add_argument("--train-size", type=int, default=30)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
