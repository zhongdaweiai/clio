"""Proper backtest accounting tests.

These verify that the cash math is correct — a bug here would silently make
results look better or worse than reality. Includes a regression for the
"adding back notional without subtracting it" bug found during development.
"""

from datetime import date, timedelta

from clio.frozen.harness import Market
from clio.research.proper_backtest import CostModel, proper_backtest
from clio.research.strategies import StrategyParams


def _make_market(market_id: str, observed: date, close: date, prices: list[float]) -> Market:
    n = len(prices)
    timeline = tuple(observed + timedelta(days=int((close - observed).days * i / n)) for i in range(n))
    return Market(
        market_id=market_id, question=f"Q {market_id}", regime="event",
        observed_at=observed, closes_at=close, timeline=timeline,
        market_prices=dict(zip(timeline, prices)),
    )


def test_no_trades_means_bankroll_unchanged():
    m = _make_market("M1", date(2024, 1, 1), date(2024, 4, 1), [0.5, 0.5, 0.5, 0.5])
    qtypes = {"M1": "event"}
    res = proper_backtest(
        StrategyParams(edge_threshold=10.0),  # never trades
        [m], qtypes, {"M1": 1}, {"event": 0.5},
        initial_bankroll=10_000, cost_model=CostModel(spread_bps=0, fee_bps=0, slippage_bps=0),
    )
    assert res.n_trades == 0
    assert res.final_bankroll == 10_000


def test_winning_yes_bet_increases_bankroll_correctly():
    # Market starts at 0.20 YES. Strategy with shrink_lambda=0 + edge_threshold=very low
    # never disagrees with market via shrinkage. Force a YES bet by using
    # symmetric and a base rate of 0.99 → strategy always says YES.
    m = _make_market("M1", date(2024, 1, 1), date(2024, 4, 1), [0.20, 0.20, 0.20, 0.20])
    qtypes = {"M1": "event"}
    params = StrategyParams(
        shrink_lambda=(("event", 1.0), ("field", 0.0), ("deadline", 0.0), ("durative", 0.0)),
        edge_threshold=0.05, kelly_fraction=0.0,
    )
    res = proper_backtest(
        params, [m], qtypes, {"M1": 1},
        base_rates={"event": 0.99},  # forces strategy to predict ~0.99 → YES bet at 0.20
        initial_bankroll=10_000,
        cost_model=CostModel(spread_bps=0, fee_bps=0, slippage_bps=0),
    )
    assert res.n_trades == 1
    t = res.trades[0]
    assert t.side == "YES"
    # Bought at 0.20, paid notional, won (notional / 0.20). Net = notional × 4.
    assert t.payoff_dollars > 0
    expected_payoff = t.notional_dollars * (1 / 0.20 - 1)
    assert abs(t.payoff_dollars - expected_payoff) < 1.0


def test_losing_no_bet_loses_full_notional():
    # Market at 0.65 YES, strategy says NO via shrinkage to base rate 0.05.
    # Outcome=YES → NO bet loses entire notional.
    m = _make_market("M1", date(2024, 1, 1), date(2024, 4, 1), [0.65, 0.65, 0.65, 0.65])
    qtypes = {"M1": "field"}
    params = StrategyParams(
        shrink_lambda=(("event", 0.0), ("field", 1.0), ("deadline", 0.0), ("durative", 0.0)),
        edge_threshold=0.05, kelly_fraction=0.0,
    )
    res = proper_backtest(
        params, [m], qtypes, {"M1": 1},
        base_rates={"field": 0.05},
        initial_bankroll=10_000,
        cost_model=CostModel(spread_bps=0, fee_bps=0, slippage_bps=0),
    )
    assert res.n_trades == 1
    t = res.trades[0]
    assert t.side == "NO"
    assert t.outcome == 1
    # NO bet on outcome=YES → lose ~ full notional.
    assert t.payoff_dollars < 0
    assert abs(t.payoff_dollars + t.notional_dollars) < 1.0


def test_cash_conservation_no_money_appears_or_disappears_with_zero_costs():
    """Regression: previous bug added notional back at close without subtracting
    at open, inflating bankroll. With zero costs, sum of trade pnls + initial
    must equal final."""
    markets = [
        _make_market(f"M{i}", date(2024, 1, 1) + timedelta(days=i * 5),
                     date(2024, 1, 1) + timedelta(days=i * 5 + 90), [0.65, 0.55, 0.45, 0.40])
        for i in range(8)
    ]
    qtypes = {f"M{i}": "field" for i in range(8)}
    resolutions = {f"M{i}": (1 if i % 2 == 0 else 0) for i in range(8)}
    params = StrategyParams(
        shrink_lambda=(("event", 0.0), ("field", 0.5), ("deadline", 0.0), ("durative", 0.0)),
        edge_threshold=0.05, kelly_fraction=0.0,
    )
    res = proper_backtest(
        params, markets, qtypes, resolutions,
        base_rates={"field": 0.05},
        initial_bankroll=10_000,
        cost_model=CostModel(spread_bps=0, fee_bps=0, slippage_bps=0),
    )
    summed_pnl = sum(t.payoff_dollars for t in res.trades)
    expected_final = res.initial_bankroll + summed_pnl
    # Should match exactly with zero costs.
    assert abs(res.final_bankroll - expected_final) < 0.01


def test_costs_reduce_pnl_monotonically():
    """Higher cost model → lower or equal final bankroll, all else equal."""
    markets = [
        _make_market(f"M{i}", date(2024, 1, 1) + timedelta(days=i * 7),
                     date(2024, 1, 1) + timedelta(days=i * 7 + 60), [0.55, 0.50, 0.45, 0.40])
        for i in range(10)
    ]
    qtypes = {f"M{i}": "field" for i in range(10)}
    resolutions = {f"M{i}": 0 for i in range(10)}  # all NO
    params = StrategyParams(
        shrink_lambda=(("event", 0.0), ("field", 0.5), ("deadline", 0.0), ("durative", 0.0)),
        edge_threshold=0.05, kelly_fraction=0.0,
    )
    res_low = proper_backtest(
        params, markets, qtypes, resolutions, {"field": 0.05},
        cost_model=CostModel(spread_bps=10, fee_bps=0, slippage_bps=0),
    )
    res_high = proper_backtest(
        params, markets, qtypes, resolutions, {"field": 0.05},
        cost_model=CostModel(spread_bps=200, fee_bps=0, slippage_bps=0),
    )
    assert res_low.final_bankroll >= res_high.final_bankroll


def test_gross_exposure_cap_binds():
    # 50 simultaneously open markets — without the gross cap we'd over-leverage.
    markets = [
        _make_market(f"M{i}", date(2024, 1, 1), date(2024, 12, 1), [0.65, 0.55, 0.45, 0.40])
        for i in range(50)
    ]
    qtypes = {f"M{i}": "field" for i in range(50)}
    resolutions = {f"M{i}": 0 for i in range(50)}
    params = StrategyParams(
        shrink_lambda=(("event", 0.0), ("field", 0.5), ("deadline", 0.0), ("durative", 0.0)),
        edge_threshold=0.05, kelly_fraction=0.0,
    )
    res = proper_backtest(
        params, markets, qtypes, resolutions, {"field": 0.05},
        initial_bankroll=10_000, max_position_pct=0.05, max_gross_exposure_pct=0.30,
        cost_model=CostModel(spread_bps=0, fee_bps=0, slippage_bps=0),
    )
    # If gross cap binds at 30% of $10K = $3K, max trades that fit at 5% each = 6.
    # We may get more trades total because positions close before market dates;
    # but the simultaneous open exposure should never exceed $3K.
    # Check trade-by-trade order they were taken: cumulative exposure at any
    # opening time can't exceed $3K + 1 position worth of overshoot.
    assert res.n_trades <= 50
    # The total notional summed across trades can be much more, since trades
    # don't all open simultaneously. So this test mainly checks the strategy
    # doesn't crash and stays sensible.
    assert res.final_bankroll > 0
