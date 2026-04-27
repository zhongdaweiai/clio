"""Data adapters.

- `synthetic`: generates a self-consistent toy world with markets, news docs,
  and resolutions. Used by tests and the CLI demo.
- `polymarket_adapter`: stub interface for the real Polymarket archive. The MVP
  intentionally does not implement live fetching — it ships the contract.
"""

from clio.data.synthetic import generate_synthetic_world, SyntheticConfig

__all__ = ["generate_synthetic_world", "SyntheticConfig"]
