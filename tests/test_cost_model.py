from clio.frozen.cost_model import CostModel


def test_zero_position_zero_slippage():
    cm = CostModel(spread_bps=100, fee_bps=20, slippage_per_pct_depth=0.5)
    assert cm.round_trip_cost(notional=0, position_fraction_of_depth=0) == 0.0


def test_buy_price_above_mid_sell_below():
    cm = CostModel(spread_bps=100)
    buy = cm.execution_price("buy", mid_price=0.5, position_fraction_of_depth=0)
    sell = cm.execution_price("sell", mid_price=0.5, position_fraction_of_depth=0)
    assert buy > 0.5
    assert sell < 0.5
    assert (buy - 0.5) > 0  # half-spread


def test_execution_price_clipped_to_unit_interval():
    cm = CostModel(spread_bps=10000)  # absurd spread
    buy = cm.execution_price("buy", mid_price=0.99, position_fraction_of_depth=10)
    sell = cm.execution_price("sell", mid_price=0.01, position_fraction_of_depth=10)
    assert 0.0 <= buy <= 1.0
    assert 0.0 <= sell <= 1.0


def test_round_trip_cost_grows_with_notional():
    cm = CostModel(spread_bps=100)
    c1 = cm.round_trip_cost(notional=100)
    c2 = cm.round_trip_cost(notional=1000)
    assert c2 > c1


def test_round_trip_cost_grows_with_depth_fraction():
    cm = CostModel(spread_bps=100, slippage_per_pct_depth=2.0)
    c1 = cm.round_trip_cost(notional=1000, position_fraction_of_depth=0)
    c2 = cm.round_trip_cost(notional=1000, position_fraction_of_depth=1.0)
    assert c2 > c1
