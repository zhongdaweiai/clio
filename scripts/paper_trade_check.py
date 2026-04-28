"""Resolution checker for paper trades.

Walks all `paper_trades/YYYY-MM-DD.json` files, finds recommendations whose
market end_date has passed, fetches resolution from Polymarket gamma API,
computes paper PnL (assuming $100K bankroll, fixed % size per recommendation,
realistic 1.5% spread + 30bps slippage round-trip cost).

Outputs:
- `paper_trades/resolutions.json`     — per-trade resolution record
- `paper_trades/SUMMARY.md`           — human-readable cumulative stats
- `paper_trades/equity_curve.json`    — chronological bankroll history

Idempotent: if a trade's resolution is already recorded, skips re-fetching.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, stdev

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("paper_check")

_HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}

INITIAL_BANKROLL = 100_000.0
SPREAD_HALF_BPS = 75    # 0.75% one-way spread cost
FEE_BPS = 0
SLIPPAGE_BPS = 30


def _http(url, timeout=20):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers=_HEADERS), timeout=timeout
    ).read().decode())


def fetch_resolution(market_id: str) -> int | None:
    """Returns 1 if YES resolved, 0 if NO, None if unresolved or unknown."""
    try:
        m = _http(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=15)
    except Exception:
        return None
    if not m.get("closed"):
        return None
    op = m.get("outcomePrices")
    if isinstance(op, str):
        try:
            op = json.loads(op)
        except json.JSONDecodeError:
            return None
    if not op:
        return None
    try:
        prices = [float(x) for x in op]
    except (TypeError, ValueError):
        return None
    outcomes = m.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = None
    yes_idx = 0
    if outcomes:
        for j, o in enumerate(outcomes):
            if str(o).strip().lower() == "yes":
                yes_idx = j
                break
    if prices[yes_idx] >= 0.99:
        return 1
    if prices[yes_idx] <= 0.01:
        return 0
    return None


def compute_payoff(rec: dict, outcome: int) -> dict:
    """Realistic payoff: notional × b on win, -notional on loss, half-spread cost."""
    cur = rec["current_price"]
    side = rec["side"]
    size_pct = rec["size_pct_of_bankroll"] / 100
    spread_pct = (SPREAD_HALF_BPS + SLIPPAGE_BPS) / 10_000
    if side == "BUY YES":
        fill = min(0.99, cur + spread_pct)
        if outcome == 1:
            ret_per_dollar = (1 - fill) / fill
        else:
            ret_per_dollar = -1
    else:
        fill = max(0.01, cur - spread_pct)
        if outcome == 0:
            ret_per_dollar = fill / (1 - fill)
        else:
            ret_per_dollar = -1
    return {
        "outcome": outcome,
        "fill_price": fill,
        "size_pct": size_pct,
        "return_pct_of_position": ret_per_dollar,
        "bankroll_pct_change": ret_per_dollar * size_pct,
    }


def main() -> int:
    pt_dir = Path("paper_trades")
    if not pt_dir.exists():
        log.warning("paper_trades/ dir does not exist; nothing to check")
        return 0

    res_file = pt_dir / "resolutions.json"
    resolutions: dict[str, dict] = {}
    if res_file.exists():
        try:
            resolutions = json.loads(res_file.read_text())
        except json.JSONDecodeError:
            resolutions = {}

    # Walk daily files chronologically.
    daily_files = sorted(pt_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    log.info("found %d daily scan files; %d resolutions cached",
             len(daily_files), len(resolutions))

    today = datetime.now(tz=timezone.utc).date()
    n_new = 0
    n_unresolved = 0

    for f in daily_files:
        try:
            scan = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        scan_date = date.fromisoformat(scan["scan_date"])
        for rec in scan["recommendations"]:
            mid = rec["market_id"]
            try:
                end_d = date.fromisoformat(rec["end_date"])
            except (KeyError, ValueError):
                continue
            # Key by (market_id, scan_date) so multiple recs on the same market
            # across different days are tracked independently.
            res_key = f"{mid}::{scan_date}"
            if res_key in resolutions:
                continue
            if end_d > today:
                n_unresolved += 1
                continue
            outcome = fetch_resolution(mid)
            time.sleep(0.10)
            if outcome is None:
                # Closed but not yet binary-resolved (e.g., outcomes invalidated).
                resolutions[res_key] = {
                    "market_id": mid, "scan_date": scan_date.isoformat(),
                    "end_date": end_d.isoformat(),
                    "status": "unknown", "outcome": None,
                    "question": rec["question"],
                }
                continue
            payoff = compute_payoff(rec, outcome)
            resolutions[res_key] = {
                "market_id": mid,
                "scan_date": scan_date.isoformat(),
                "end_date": end_d.isoformat(),
                "status": "resolved",
                "question": rec["question"],
                "qtype": rec["qtype"],
                "side": rec["side"],
                "current_price": rec["current_price"],
                "llm_pred": rec["llm_pred"],
                "edge": rec["edge"],
                "size_pct": rec["size_pct_of_bankroll"] / 100,
                "outcome": outcome,
                "fill_price": payoff["fill_price"],
                "ret_pct_of_position": payoff["return_pct_of_position"],
                "bankroll_pct_change": payoff["bankroll_pct_change"],
                "polymarket_url": rec["polymarket_url"],
            }
            n_new += 1
            if n_new % 5 == 0:
                res_file.write_text(json.dumps(resolutions, indent=2))

    res_file.write_text(json.dumps(resolutions, indent=2))
    log.info(f"new resolutions: {n_new}, unresolved waiting: {n_unresolved}, "
             f"total tracked: {len(resolutions)}")

    # ---- compute cumulative stats ----
    resolved_only = [r for r in resolutions.values() if r.get("status") == "resolved"]
    resolved_only.sort(key=lambda r: r["scan_date"])

    if not resolved_only:
        log.info("no resolved trades yet — writing placeholder SUMMARY.md")
        # Count pending signals so the user has something to look at.
        n_pending = 0
        n_total_signals = 0
        next_close = None
        for f in daily_files:
            try:
                scan = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            n_total_signals += len(scan["recommendations"])
            for rec in scan["recommendations"]:
                try:
                    end_d = date.fromisoformat(rec["end_date"])
                except (KeyError, ValueError):
                    continue
                if end_d > today:
                    n_pending += 1
                    if next_close is None or end_d < next_close:
                        next_close = end_d

        days_to_first = (next_close - today).days if next_close else None
        md = [
            "# Clio Paper Trading — Live Forward Performance",
            "",
            f"**As of:** {today.isoformat()}",
            f"**Initial bankroll:** ${INITIAL_BANKROLL:,.0f}",
            "",
            "## No resolved trades yet",
            "",
            f"- **{n_total_signals}** signals issued so far across **{len(daily_files)}** daily scans",
            f"- **{n_pending}** signals waiting for their markets to close",
        ]
        if next_close is not None:
            md.append(f"- Earliest pending resolution: **{next_close.isoformat()}** "
                     f"({days_to_first} day{'s' if days_to_first != 1 else ''} from now)")
        md += [
            "",
            "Realized PnL, hit rate, and equity curve will populate this file",
            "as soon as the first market closes. Until then, see:",
            "",
            "- [`ALL_SIGNALS.md`](ALL_SIGNALS.md) — every signal logged so far",
            f"- [`{today.isoformat()}.json`](" + f"{today.isoformat()}.json) — today's raw output (if today's scan ran)",
            "",
            "Backtest expectation (from `docs/LLM_RUN.md`): **+259% CAGR / +11.25%/mo, hit rate 69%**.",
            "Live results will be compared after ≥50 resolved trades.",
            "",
        ]
        (pt_dir / "SUMMARY.md").write_text("\n".join(md) + "\n")
        return 0

    bankroll = INITIAL_BANKROLL
    equity = [(min(r["scan_date"] for r in resolved_only), bankroll)]
    n_wins = 0
    pnls = []
    for r in resolved_only:
        size_dollars = bankroll * r["size_pct"]
        ret_dollars = size_dollars * r["ret_pct_of_position"]
        bankroll += ret_dollars
        pnls.append(ret_dollars)
        if r["ret_pct_of_position"] > 0:
            n_wins += 1
        equity.append((r["end_date"], bankroll))

    total_return_pct = (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL
    n_trades = len(resolved_only)
    hit_rate = n_wins / n_trades if n_trades else 0
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]
    profit_factor = (sum(win_pnls) / abs(sum(loss_pnls))
                     if loss_pnls and abs(sum(loss_pnls)) > 0 else float("inf"))
    avg_win = mean(win_pnls) if win_pnls else 0
    avg_loss = mean(loss_pnls) if loss_pnls else 0

    # Per-position returns for Sharpe (rough — assumes one bet per period).
    rets = [r["bankroll_pct_change"] for r in resolved_only]
    if len(rets) > 1:
        mu = mean(rets)
        sd = stdev(rets)
        sharpe_per_trade = mu / sd if sd > 0 else 0
        # Annualize assuming ~5 trades/week (rough)
        sharpe_annual = sharpe_per_trade * math.sqrt(250)
    else:
        sharpe_annual = 0

    peak = INITIAL_BANKROLL
    max_dd = 0
    for _, e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    summary = {
        "as_of": today.isoformat(),
        "n_trades": n_trades,
        "n_wins": n_wins,
        "hit_rate": hit_rate,
        "initial_bankroll": INITIAL_BANKROLL,
        "current_bankroll": bankroll,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe_per_trade": sharpe_per_trade if len(rets) > 1 else 0,
        "sharpe_annual_rough": sharpe_annual,
        "equity_curve": equity,
    }
    (pt_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Human-readable Markdown summary
    md = ["# Clio Paper Trading — Live Forward Performance", ""]
    md.append(f"**As of:** {today.isoformat()}")
    md.append(f"**Initial bankroll:** ${INITIAL_BANKROLL:,.0f}")
    md.append(f"**Current bankroll:** ${bankroll:,.0f}  ({100 * total_return_pct:+.2f}%)")
    md.append(f"**Trades resolved:** {n_trades} (hit rate {100*hit_rate:.1f}%)")
    md.append(f"**Profit factor:** {profit_factor:.2f}")
    md.append(f"**Max drawdown:** {100*max_dd:.1f}%")
    md.append(f"**Sharpe (rough annual):** {sharpe_annual:.2f}")
    md.append(f"**Avg win / Avg loss:** ${avg_win:+,.0f} / ${avg_loss:+,.0f}")
    md.append("")
    md.append("## Recent resolved trades")
    md.append("")
    md.append("| date | side | qtype | mkt→LLM | edge | outcome | $ pnl | bankroll |")
    md.append("|---|---|---|---|---|---|---|---|")
    cur_bank = INITIAL_BANKROLL
    for r in resolved_only[-30:]:
        size_dollars = cur_bank * r["size_pct"]
        pnl_dollars = size_dollars * r["ret_pct_of_position"]
        cur_bank += pnl_dollars
        side_emoji = "🟢" if r["side"] == "BUY YES" else "🔴"
        outcome_emoji = "✅" if r["ret_pct_of_position"] > 0 else "❌"
        q_short = r["question"][:60].replace("|", "\\|")
        md.append(
            f"| {r['end_date']} | {side_emoji} {r['side'].split()[1]} | {r['qtype']} | "
            f"{r['current_price']:.2f}→{r['llm_pred']:.2f} | {r['edge']:+.2f} | "
            f"{outcome_emoji} {r['outcome']} | ${pnl_dollars:+,.0f} | ${cur_bank:,.0f} |"
        )
    md.append("")
    md.append("## Equity curve (snapshot at each resolution)")
    md.append("```")
    for d, e in equity[-30:]:
        bar_len = int(40 * (e / INITIAL_BANKROLL))
        bar = "█" * min(bar_len, 60)
        md.append(f"{d}  ${e:>10,.0f}  {bar}")
    md.append("```")
    (pt_dir / "SUMMARY.md").write_text("\n".join(md) + "\n")
    log.info(f"updated SUMMARY.md: {n_trades} trades, ${bankroll:,.0f} bankroll, "
             f"hit {100*hit_rate:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
