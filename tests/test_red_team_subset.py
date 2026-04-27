"""Subset attack tests."""

from datetime import date

from clio.frozen.harness import BacktestRun, Forecast
from clio.frozen.scoring import ScoreVector
from clio.red_team.subset_attack import run_subset_attack, FeatureSlot


def _make_run_with(probs, outcomes, market_prices, regimes, market_ids=None):
    """Hand-build a BacktestRun for testing."""
    if market_ids is None:
        market_ids = [f"M{i:04d}" for i in range(len(probs))]
    forecasts = [
        Forecast(market_id=mid, as_of=date(2024, 1, 1), prob=p)
        for mid, p in zip(market_ids, probs)
    ]
    return BacktestRun(
        strategy_name="test",
        forecasts=forecasts,
        final_probs=list(probs),
        final_outcomes=list(outcomes),
        final_market_prices=list(market_prices),
    ), market_ids, list(regimes)


def test_finds_blind_spot_in_one_regime():
    # 30 markets: 20 well-predicted, 10 in 'geo' systematically wrong.
    probs = [0.9] * 20 + [0.95] * 10
    outcomes = [1] * 20 + [0] * 10  # geo predictions are very wrong
    prices = [0.5] * 30
    regimes = ["election"] * 20 + ["geo"] * 10
    n_evi = [3] * 30

    run, _, _ = _make_run_with(probs, outcomes, prices, regimes)
    report = run_subset_attack(
        run, regimes, n_evi,
        min_bucket_size=5, min_brier_degradation=0.1,
        bootstrap_resamples=200,
    )
    assert report.blind_spots
    worst = report.blind_spots[0]
    assert worst.slot.feature == "regime"
    assert worst.slot.bucket == "geo"
    assert worst.brier_degradation > 0.1


def test_no_blind_spots_when_uniform():
    # Outcomes perfectly distributed across the two regime buckets so they
    # have identical Brier — no regime should be a blind spot.
    probs = [0.7] * 20
    # election: 5 wins, 5 losses (Brier per regime = (5*0.09+5*0.49)/10 = 0.29)
    # sport:    5 wins, 5 losses (same)
    outcomes = ([1, 0] * 5) + ([1, 0] * 5)
    prices = [0.5] * 20
    regimes = ["election"] * 10 + ["sport"] * 10
    n_evi = [3] * 20

    run, _, _ = _make_run_with(probs, outcomes, prices, regimes)
    report = run_subset_attack(
        run, regimes, n_evi,
        min_bucket_size=5, min_brier_degradation=0.05,
        bootstrap_resamples=200,
    )
    # No regime bucket should appear as a significant blind spot.
    regime_bs = [bs for bs in report.blind_spots if bs.slot.feature == "regime"]
    assert regime_bs == []


def test_min_bucket_size_filters_small_buckets():
    probs = [0.9] * 18 + [0.95] * 2
    outcomes = [1] * 18 + [0] * 2
    prices = [0.5] * 20
    regimes = ["a"] * 18 + ["b"] * 2  # only 2 in 'b'
    n_evi = [3] * 20

    run, _, _ = _make_run_with(probs, outcomes, prices, regimes)
    report = run_subset_attack(run, regimes, n_evi, min_bucket_size=5)
    # 'b' bucket should be filtered out due to size.
    assert all(bs.slot.bucket != "b" for bs in report.blind_spots)


def test_empty_run_returns_empty_report():
    run = BacktestRun(strategy_name="empty")
    report = run_subset_attack(run, [], [])
    assert report.n_holdout == 0
    assert report.blind_spots == []


def test_blind_spots_sorted_by_degradation():
    # Two bad regimes with different degrees of badness.
    probs = [0.9] * 10 + [0.95] * 10 + [0.99] * 10
    outcomes = [1] * 10 + [0] * 10 + [0] * 10
    prices = [0.5] * 30
    regimes = ["good"] * 10 + ["bad"] * 10 + ["worse"] * 10
    n_evi = [3] * 30

    run, _, _ = _make_run_with(probs, outcomes, prices, regimes)
    report = run_subset_attack(
        run, regimes, n_evi,
        min_bucket_size=5, min_brier_degradation=0.05,
        bootstrap_resamples=300, significance_level=0.20,
    )
    if len(report.blind_spots) >= 2:
        regime_blindspots = [b for b in report.blind_spots if b.slot.feature == "regime"]
        if len(regime_blindspots) >= 2:
            # 'worse' should rank above 'bad' in Brier degradation.
            ranks = {b.slot.bucket: i for i, b in enumerate(regime_blindspots)}
            assert ranks["worse"] < ranks["bad"]
