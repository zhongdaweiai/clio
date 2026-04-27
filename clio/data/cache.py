"""Disk cache for adapter responses.

Why we need this:
- Polymarket API responses are stable for resolved markets — cache forever.
- Re-fetching during development burns rate limit and slows iteration.
- Backtests must be reproducible across environments.

The cache is a flat directory of JSON files keyed by SHA-256 of the request
descriptor. Reads check freshness against a per-key TTL written into the value.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _hash_key(parts: tuple[Any, ...]) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


@dataclass
class CacheEntry:
    fetched_at: float
    ttl_seconds: float | None
    value: Any

    def is_fresh(self, now: float | None = None) -> bool:
        if self.ttl_seconds is None:
            return True
        now = now or time.time()
        return (now - self.fetched_at) < self.ttl_seconds


class DiskCache:
    """A keyed JSON cache. Thread-unsafe; one process per cache dir."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key_parts: tuple[Any, ...]) -> Any | None:
        key = _hash_key(key_parts)
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        entry = CacheEntry(**obj)
        if not entry.is_fresh():
            return None
        return entry.value

    def put(
        self,
        key_parts: tuple[Any, ...],
        value: Any,
        ttl_seconds: float | None = None,
    ) -> None:
        key = _hash_key(key_parts)
        p = self._path(key)
        entry = CacheEntry(fetched_at=time.time(), ttl_seconds=ttl_seconds, value=value)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(entry.__dict__, f)
        os.replace(tmp, p)

    def __len__(self) -> int:
        return sum(1 for p in self.root.glob("*.json"))

    def clear(self) -> None:
        for p in self.root.glob("*.json"):
            p.unlink()
