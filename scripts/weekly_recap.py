"""Weekly recap email.

Aggregates the past 7 days of signals + resolutions into a single email.
Runs Monday mornings.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("recap")

sys.path.insert(0, str(Path(__file__).parent))
from notify import send_email  # noqa


REPO_URL = "https://github.com/zhongdaweiai/clio"
INITIAL_BANKROLL = 100_000.0


def main() -> int:
    to = os.environ.get("NOTIFY_EMAIL")
    if not to:
        log.warning("NOTIFY_EMAIL not set, skipping")
        return 0

    pt = Path("paper_trades")
    if not pt.exists():
        log.warning("paper_trades/ not found")
        return 0

    today = datetime.now(tz=timezone.utc).date()
    week_ago = today - timedelta(days=7)

    # Collect last 7 days of scans
    weekly_scans = []
    weekly_signals = 0
    high_edge_signals = 0
    for f in sorted(pt.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")):
        try:
            scan = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        scan_d = date.fromisoformat(scan["scan_date"])
        if scan_d < week_ago:
            continue
        weekly_scans.append(scan)
        weekly_signals += len(scan["recommendations"])
        high_edge_signals += sum(1 for r in scan["recommendations"] if abs(r["edge"]) >= 0.20)

    # Collect last 7 days of resolutions
    res_path = pt / "resolutions.json"
    weekly_resolved = []
    if res_path.exists():
        all_res = json.loads(res_path.read_text())
        for r in all_res.values():
            if r.get("status") != "resolved":
                continue
            try:
                end_d = date.fromisoformat(r["end_date"])
            except (KeyError, ValueError):
                continue
            if end_d >= week_ago:
                weekly_resolved.append(r)
    weekly_resolved.sort(key=lambda r: r["end_date"], reverse=True)

    # Bankroll snapshot from summary.json
    summary_path = pt / "summary.json"
    bankroll = INITIAL_BANKROLL
    cum_trades = 0
    cum_hit = 0
    cum_ret = 0
    if summary_path.exists():
        try:
            s = json.loads(summary_path.read_text())
            bankroll = s.get("current_bankroll", INITIAL_BANKROLL)
            cum_trades = s.get("n_trades", 0)
            cum_hit = s.get("hit_rate", 0)
            cum_ret = s.get("total_return_pct", 0)
        except json.JSONDecodeError:
            pass

    # Compose email
    subj = f"[Clio Weekly] {week_ago} → {today}: {weekly_signals} signals, " \
           f"{len(weekly_resolved)} resolved, ${bankroll:,.0f} bankroll"

    weekly_pnl_dollars = sum(
        bankroll * r.get("size_pct", 0) * r.get("ret_pct_of_position", 0)
        for r in weekly_resolved
    )
    weekly_wins = sum(1 for r in weekly_resolved if r.get("ret_pct_of_position", 0) > 0)
    weekly_hit = weekly_wins / len(weekly_resolved) if weekly_resolved else 0

    rows_resolved = []
    for r in weekly_resolved[:15]:
        side_em = "🟢" if r["side"] == "BUY YES" else "🔴"
        outcome_em = "✅" if r.get("ret_pct_of_position", 0) > 0 else "❌"
        ret_pct = r.get("ret_pct_of_position", 0) * 100
        rows_resolved.append(
            f'<tr><td>{r["end_date"]}</td><td>{side_em}</td><td>{r["qtype"]}</td>'
            f'<td>{r["current_price"]:.2f}→{r["llm_pred"]:.2f}</td>'
            f'<td>{outcome_em} outcome={r.get("outcome")}</td>'
            f'<td>{ret_pct:+.0f}%</td>'
            f'<td>{r["question"][:60]}</td></tr>'
        )
    table_resolved = "".join(rows_resolved) if rows_resolved else \
        '<tr><td colspan="7"><em>No trades resolved this week.</em></td></tr>'

    html = f"""
    <html><body style="font-family:-apple-system,sans-serif;max-width:900px;margin:auto;color:#222">
    <h2>Clio Weekly Recap — {week_ago} → {today}</h2>

    <h3>Bankroll</h3>
    <table cellpadding="8" style="font-size:14px;border-collapse:collapse;width:100%">
        <tr style="background:#f4f4f4">
            <td><strong>Current bankroll</strong></td>
            <td><strong>${bankroll:,.0f}</strong>
                <span style="color:{'#0a7' if cum_ret>=0 else '#c00'}">({100*cum_ret:+.2f}% all-time)</span>
            </td>
        </tr>
        <tr><td>Cumulative resolved trades</td><td>{cum_trades} (hit rate {100*cum_hit:.1f}%)</td></tr>
    </table>

    <h3>This week</h3>
    <ul>
        <li><strong>{len(weekly_scans)}</strong> daily scans</li>
        <li><strong>{weekly_signals}</strong> signals issued</li>
        <li><strong>{high_edge_signals}</strong> high-conviction (|edge| ≥ 0.20)</li>
        <li><strong>{len(weekly_resolved)}</strong> trades resolved
            (hit rate {100*weekly_hit:.0f}%, ${weekly_pnl_dollars:+,.0f} this week)</li>
    </ul>

    <h3>Resolutions this week</h3>
    <table cellpadding="6" style="font-size:13px;border-collapse:collapse;width:100%">
        <thead style="background:#f4f4f4">
            <tr><th align="left">Date</th><th>Side</th><th>Type</th><th>Mkt→LLM</th>
                <th>Outcome</th><th>Return</th><th align="left">Question</th></tr>
        </thead>
        <tbody>{table_resolved}</tbody>
    </table>

    <p style="margin-top:24px">
        <a href="{REPO_URL}/blob/main/paper_trades/SUMMARY.md"
           style="background:#0066cc;color:white;padding:8px 12px;text-decoration:none;border-radius:4px">
            Full PnL + equity curve
        </a>
        &nbsp;
        <a href="{REPO_URL}/blob/main/paper_trades/ALL_SIGNALS.md"
           style="background:#0a7;color:white;padding:8px 12px;text-decoration:none;border-radius:4px">
            Every signal ever issued
        </a>
    </p>

    <p style="color:#666;font-size:12px;margin-top:24px">
        Forward paper trade. Backtest expectation: +259% CAGR / +11.25%/mo / hit rate 69%.
        Will compare to live performance after ≥50 resolved trades.
    </p>
    </body></html>
    """

    if send_email(to, subj, html):
        log.info(f"weekly recap sent to {to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
