"""Local-only simulation: fabricate resolved trades from the v3 backtest
dataset to prove the researcher agent can produce valid proposals when
given enough data.

NOT run in production. Used only to test the agent in development.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, "scripts")
from researcher_agent import main as researcher_main  # noqa


def _fabricate_paper_trades(target: Path, n_trades: int = 50, seed: int = 7) -> None:
    """Generate fake but well-typed paper_trades/ data from v3 historical
    markets, with realistic edge/outcome correlation."""
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    v3 = json.loads(Path("runs/live_iter/markets_v3.json").read_text())
    markets_with_qtype = [m for m in v3["markets"] if m.get("qtype") in ("event", "field", "deadline")]
    rng.shuffle(markets_with_qtype)
    sampled = markets_with_qtype[:n_trades]
    resolutions_data = v3["resolutions"]

    today = date(2026, 4, 28)
    earliest_scan = today - timedelta(days=60)
    daily: dict[str, list] = {}
    resolutions: dict[str, dict] = {}

    for i, m in enumerate(sampled):
        if m["market_id"] not in resolutions_data:
            continue
        outcome = int(resolutions_data[m["market_id"]])
        scan_d = earliest_scan + timedelta(days=i * (60 // n_trades))
        end_d = scan_d + timedelta(days=rng.randint(7, 45))
        if end_d > today - timedelta(days=1):
            end_d = today - timedelta(days=1)
        # Synthesize an LLM prediction with a slight edge over market by qtype
        first_t = m["timeline"][0]
        market_price = m["market_prices"].get(first_t, 0.5)
        # event/field type — simulate "fade favorites" edge
        if m["qtype"] in ("event", "field") and market_price > 0.5:
            llm_pred = max(0.05, market_price - 0.30)
        elif m["qtype"] in ("event",) and market_price < 0.10:
            llm_pred = min(0.95, market_price + 0.10)
        else:
            llm_pred = market_price + rng.uniform(-0.10, 0.10)
        edge = llm_pred - market_price

        if abs(edge) < 0.10:
            continue  # below threshold

        side = "BUY YES" if edge > 0 else "BUY NO"
        size_frac = 0.10 + 1.2 * 0.05  # mock Kelly
        size_frac = min(size_frac, 0.25)

        rec = {
            "market_id": m["market_id"],
            "question": m["question"],
            "qtype": m["qtype"],
            "scan_date": scan_d.isoformat(),
            "current_price": market_price,
            "llm_pred": llm_pred,
            "llm_confidence": rng.choice(["low", "medium", "high"]),
            "edge": edge,
            "side": side,
            "size_pct_of_bankroll": round(size_frac * 100, 2),
            "days_remaining": (end_d - scan_d).days,
            "end_date": end_d.isoformat(),
            "polymarket_url": "https://polymarket.com/event/sim",
        }
        daily.setdefault(scan_d.isoformat(), []).append(rec)

        # Compute resolution payoff
        spread_pct = 0.0105
        if side == "BUY YES":
            fill = min(0.99, market_price + spread_pct)
            ret = (1 - fill) / fill if outcome == 1 else -1
        else:
            fill = max(0.01, market_price - spread_pct)
            ret = fill / (1 - fill) if outcome == 0 else -1

        resolutions[f"{m['market_id']}::{scan_d.isoformat()}"] = {
            "market_id": m["market_id"],
            "scan_date": scan_d.isoformat(),
            "end_date": end_d.isoformat(),
            "status": "resolved",
            "question": m["question"],
            "qtype": m["qtype"],
            "side": side,
            "current_price": market_price,
            "llm_pred": llm_pred,
            "edge": edge,
            "size_pct": size_frac,
            "outcome": outcome,
            "fill_price": fill,
            "ret_pct_of_position": ret,
            "bankroll_pct_change": ret * size_frac,
            "polymarket_url": "https://polymarket.com/event/sim",
        }

    # Write daily files
    for d, recs in daily.items():
        (target / f"{d}.json").write_text(json.dumps({
            "scan_date": d,
            "scan_iso_utc": d + "T00:00:00+00:00",
            "model": "claude-sonnet-4-6",
            "n_open_scanned": 100,
            "n_recommendations": len(recs),
            "skipped": {},
            "llm_usage": {"calls": len(recs), "tokens_in": 0, "tokens_out": 0,
                          "estimated_cost_usd": 0},
            "recommendations": recs,
        }, indent=2))

    (target / "resolutions.json").write_text(json.dumps(resolutions, indent=2))
    # Mock summary
    bankroll = 100_000
    for r in sorted(resolutions.values(), key=lambda x: x["scan_date"]):
        bankroll *= 1 + r["bankroll_pct_change"]
    (target / "summary.json").write_text(json.dumps({
        "as_of": today.isoformat(),
        "n_trades": len(resolutions),
        "n_wins": sum(1 for r in resolutions.values() if r["bankroll_pct_change"] > 0),
        "hit_rate": sum(1 for r in resolutions.values() if r["bankroll_pct_change"] > 0) / max(1, len(resolutions)),
        "initial_bankroll": 100_000,
        "current_bankroll": bankroll,
        "total_return_pct": (bankroll - 100_000) / 100_000,
        "max_drawdown_pct": 0.20,
        "profit_factor": 2.0,
        "avg_win": 1500,
        "avg_loss": -800,
        "sharpe_per_trade": 0.5,
        "sharpe_annual_rough": 1.5,
        "equity_curve": [],
    }, indent=2))


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required for simulation")
        return 1

    # Backup real paper_trades and replace with simulated.
    real = Path("paper_trades")
    backup = Path("paper_trades_backup_for_sim")
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(real, backup)

    # Wipe and refabricate
    shutil.rmtree(real)
    real.mkdir()
    _fabricate_paper_trades(real, n_trades=60, seed=11)
    print(f"fabricated {len(list(real.glob('*.json')))} files in {real}")

    # Run researcher (won't actually open PR because we're not on a clean branch; but
    # we'll see the LLM proposal and validation result).
    try:
        # Use dry-run mode so the agent skips git ops + PR + email.
        sys.argv = ["researcher_agent_simulate", "--dry-run"]
        rc = researcher_main()
    finally:
        # Restore real paper_trades
        shutil.rmtree(real)
        shutil.copytree(backup, real)
        shutil.rmtree(backup)

    return rc


if __name__ == "__main__":
    sys.exit(main())
