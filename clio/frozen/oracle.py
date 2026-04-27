"""Resolution oracle.

The single source of truth for "what actually happened" on each market.
Agents may not write to this — they may only read after a backtest is over.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Resolution:
    market_id: str
    outcome: int  # 0 or 1


class ResolutionOracle:
    def __init__(self) -> None:
        self._resolutions: dict[str, Resolution] = {}

    def record(self, market_id: str, outcome: int) -> None:
        if outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {outcome}")
        if market_id in self._resolutions:
            raise ValueError(f"market {market_id} already resolved")
        self._resolutions[market_id] = Resolution(market_id, outcome)

    def lookup(self, market_id: str) -> int:
        if market_id not in self._resolutions:
            raise KeyError(f"market {market_id} not resolved")
        return self._resolutions[market_id].outcome

    def __contains__(self, market_id: str) -> bool:
        return market_id in self._resolutions

    def __len__(self) -> int:
        return len(self._resolutions)
