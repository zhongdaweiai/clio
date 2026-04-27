"""Trace store. Append-only. SQLite-backed.

Why SQLite: zero ops cost, transactional, queryable. The schema is
intentionally simple — agents may query by (market_id, as_of, strategy)
or by date range to look at recent failures.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Trace:
    strategy_name: str
    market_id: str
    regime: str
    as_of: date
    forecast: float
    market_price: float
    rationale: str
    evidence_doc_ids: tuple[str, ...]
    extra: dict


class TraceStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_name TEXT NOT NULL,
        market_id TEXT NOT NULL,
        regime TEXT NOT NULL,
        as_of TEXT NOT NULL,
        forecast REAL NOT NULL,
        market_price REAL NOT NULL,
        rationale TEXT,
        evidence_doc_ids TEXT,
        extra TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_traces_market ON traces(market_id, as_of);
    CREATE INDEX IF NOT EXISTS idx_traces_strategy ON traces(strategy_name);
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        finally:
            cur.close()

    def write(self, trace: Trace) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO traces (strategy_name, market_id, regime, as_of, "
                "forecast, market_price, rationale, evidence_doc_ids, extra, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.strategy_name,
                    trace.market_id,
                    trace.regime,
                    trace.as_of.isoformat(),
                    trace.forecast,
                    trace.market_price,
                    trace.rationale,
                    json.dumps(list(trace.evidence_doc_ids)),
                    json.dumps(trace.extra),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def read_by_strategy(self, strategy_name: str) -> list[Trace]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT strategy_name, market_id, regime, as_of, forecast, "
                "market_price, rationale, evidence_doc_ids, extra "
                "FROM traces WHERE strategy_name = ? ORDER BY id",
                (strategy_name,),
            )
            rows = cur.fetchall()
        return [
            Trace(
                strategy_name=r[0],
                market_id=r[1],
                regime=r[2],
                as_of=date.fromisoformat(r[3]),
                forecast=r[4],
                market_price=r[5],
                rationale=r[6] or "",
                evidence_doc_ids=tuple(json.loads(r[7] or "[]")),
                extra=json.loads(r[8] or "{}"),
            )
            for r in rows
        ]

    def __len__(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM traces")
            return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
