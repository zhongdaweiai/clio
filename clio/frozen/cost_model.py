"""Trading cost model.

Real prediction markets charge in three ways:
- Bid-ask spread (your entry price is worse than mid)
- Maker/taker fees (Polymarket waives, Kalshi charges)
- Slippage (a fraction of the order book depth shifts against you)

We model all three. The model is intentionally pessimistic — backtest costs
should bound real costs, not match them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    spread_bps: float = 100.0
    fee_bps: float = 0.0
    slippage_per_pct_depth: float = 0.5

    def execution_price(
        self,
        side: str,
        mid_price: float,
        position_fraction_of_depth: float,
    ) -> float:
        """Price you actually pay/receive.

        side: "buy" or "sell"
        mid_price: current mid in [0,1]
        position_fraction_of_depth: order size as fraction of book depth (0..1+)
        """
        assert side in ("buy", "sell"), side
        assert 0.0 <= mid_price <= 1.0, mid_price
        depth_frac = max(0.0, position_fraction_of_depth)

        spread_half = self.spread_bps / 2 / 10_000
        slip = self.slippage_per_pct_depth * depth_frac / 100.0
        adverse = spread_half + slip
        if side == "buy":
            px = mid_price + adverse
        else:
            px = mid_price - adverse
        return min(1.0, max(0.0, px))

    def round_trip_cost(
        self,
        notional: float,
        position_fraction_of_depth: float = 0.0,
    ) -> float:
        """Approximate dollar cost of a full enter+exit round trip."""
        spread_cost = notional * self.spread_bps / 10_000
        fee_cost = notional * self.fee_bps / 10_000 * 2
        slip_cost = (
            notional * self.slippage_per_pct_depth * position_fraction_of_depth / 100.0
        )
        return spread_cost + fee_cost + slip_cost
