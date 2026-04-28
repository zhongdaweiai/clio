"""Compute live performance slices for the researcher agent.

Slice the resolved trades by qtype, edge band, LLM confidence, days_held,
side. The agent uses these slices to identify which parameter change has
the most evidence behind it.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path


@dataclass
class SliceStats:
    n: int = 0
    n_wins: int = 0
    avg_edge: float = 0.0
    avg_return_pct: float = 0.0
    total_pnl_pct: float = 0.0  # cumulative bankroll return contribution

    @property
    def hit_rate(self) -> float:
        return self.n_wins / self.n if self.n else 0.0


@dataclass
class LiveMetrics:
    """Snapshot of live paper-trade performance, sliced multiple ways."""
    n_resolved: int = 0
    n_pending: int = 0
    overall: SliceStats = field(default_factory=SliceStats)
    by_qtype: dict[str, SliceStats] = field(default_factory=dict)
    by_edge_band: dict[str, SliceStats] = field(default_factory=dict)
    by_confidence: dict[str, SliceStats] = field(default_factory=dict)
    by_side: dict[str, SliceStats] = field(default_factory=dict)
    by_days_held_bucket: dict[str, SliceStats] = field(default_factory=dict)
    bankroll_initial: float = 100_000.0
    bankroll_current: float = 100_000.0
    max_drawdown_pct: float = 0.0
    backtest_expectations: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "n_resolved": self.n_resolved,
            "n_pending": self.n_pending,
            "overall": asdict(self.overall),
            "by_qtype": {k: asdict(v) for k, v in self.by_qtype.items()},
            "by_edge_band": {k: asdict(v) for k, v in self.by_edge_band.items()},
            "by_confidence": {k: asdict(v) for k, v in self.by_confidence.items()},
            "by_side": {k: asdict(v) for k, v in self.by_side.items()},
            "by_days_held_bucket": {k: asdict(v) for k, v in self.by_days_held_bucket.items()},
            "bankroll_initial": self.bankroll_initial,
            "bankroll_current": self.bankroll_current,
            "max_drawdown_pct": self.max_drawdown_pct,
            "backtest_expectations": self.backtest_expectations,
        }


def _edge_band(edge: float) -> str:
    a = abs(edge)
    if a < 0.10:
        return "low<0.10"
    if a < 0.20:
        return "mid_0.10-0.20"
    if a < 0.30:
        return "high_0.20-0.30"
    return "very_high>=0.30"


def _days_bucket(days: int) -> str:
    if days <= 7:
        return "0-7d"
    if days <= 30:
        return "8-30d"
    if days <= 60:
        return "31-60d"
    return "60d+"


def _add_to(stats: SliceStats, edge: float, ret: float, pnl_pct: float, won: bool) -> None:
    n = stats.n
    stats.avg_edge = (stats.avg_edge * n + abs(edge)) / (n + 1)
    stats.avg_return_pct = (stats.avg_return_pct * n + ret) / (n + 1)
    stats.total_pnl_pct += pnl_pct
    stats.n += 1
    if won:
        stats.n_wins += 1


def compute_live_metrics(
    paper_trades_dir: Path,
    backtest_expectations: dict | None = None,
) -> LiveMetrics:
    """Aggregate paper_trades/ into a LiveMetrics object.

    Reads:
      - paper_trades/resolutions.json  (resolved trades with payoffs)
      - paper_trades/summary.json      (cumulative bankroll snapshot)
      - paper_trades/YYYY-MM-DD.json   (pending counts)
    """
    metrics = LiveMetrics()
    metrics.backtest_expectations = backtest_expectations or {
        "hit_rate": 0.69,
        "monthly_compound": 0.1125,
        "cagr": 2.591,
        "profit_factor": 3.26,
        "max_dd": 0.346,
        "sharpe_annual": 3.12,
        "best_edge_threshold": 0.20,
    }

    # Load summary if present.
    summary_path = paper_trades_dir / "summary.json"
    if summary_path.exists():
        try:
            s = json.loads(summary_path.read_text())
            metrics.bankroll_initial = s.get("initial_bankroll", 100_000)
            metrics.bankroll_current = s.get("current_bankroll", 100_000)
            metrics.max_drawdown_pct = s.get("max_drawdown_pct", 0)
        except json.JSONDecodeError:
            pass

    # Load resolved trades.
    res_path = paper_trades_dir / "resolutions.json"
    if not res_path.exists():
        # No resolved data yet — count pendings only.
        for f in sorted(paper_trades_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")):
            try:
                scan = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            metrics.n_pending += len(scan.get("recommendations", []))
        return metrics

    try:
        resolved = json.loads(res_path.read_text())
    except json.JSONDecodeError:
        resolved = {}
    resolved_only = [r for r in resolved.values() if r.get("status") == "resolved"]
    metrics.n_resolved = len(resolved_only)

    # Pending = total signals minus resolved.
    n_total_signals = 0
    for f in sorted(paper_trades_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")):
        try:
            scan = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        n_total_signals += len(scan.get("recommendations", []))
    metrics.n_pending = max(0, n_total_signals - len(resolved_only))

    for r in resolved_only:
        edge = r.get("edge", 0)
        ret = r.get("ret_pct_of_position", 0)
        pnl_pct = r.get("bankroll_pct_change", 0)
        won = ret > 0

        # We need confidence from the original signal — look up by market_id+scan_date.
        # Skip for now; confidence isn't preserved in resolutions.json.
        # We'll add it after a re-fetch if needed.

        _add_to(metrics.overall, edge, ret, pnl_pct, won)
        qt = r.get("qtype", "unknown")
        _add_to(metrics.by_qtype.setdefault(qt, SliceStats()), edge, ret, pnl_pct, won)

        band = _edge_band(edge)
        _add_to(metrics.by_edge_band.setdefault(band, SliceStats()), edge, ret, pnl_pct, won)

        side = r.get("side", "unknown").replace("BUY ", "")
        _add_to(metrics.by_side.setdefault(side, SliceStats()), edge, ret, pnl_pct, won)

        end_d = r.get("end_date")
        scan_d = r.get("scan_date")
        if end_d and scan_d:
            try:
                days = (date.fromisoformat(end_d) - date.fromisoformat(scan_d)).days
                _add_to(metrics.by_days_held_bucket.setdefault(_days_bucket(days), SliceStats()),
                        edge, ret, pnl_pct, won)
            except ValueError:
                pass

    # Best-effort confidence join from signal files.
    confs: dict[str, list] = {}
    for f in sorted(paper_trades_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")):
        try:
            scan = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for rec in scan.get("recommendations", []):
            confs[(rec["market_id"], rec["scan_date"])] = rec.get("llm_confidence", "unknown")

    for r in resolved_only:
        key = (r.get("market_id"), r.get("scan_date"))
        conf = confs.get(key, "unknown")
        _add_to(
            metrics.by_confidence.setdefault(conf, SliceStats()),
            r.get("edge", 0),
            r.get("ret_pct_of_position", 0),
            r.get("bankroll_pct_change", 0),
            r.get("ret_pct_of_position", 0) > 0,
        )

    return metrics
