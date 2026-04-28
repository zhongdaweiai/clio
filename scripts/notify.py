"""Send a paper-trading summary email.

Reads the most recent daily scan + cumulative summary, formats as HTML,
sends via Resend (preferred) or Gmail SMTP (fallback).

Env vars (set as GitHub repo secrets):
  Either:
    RESEND_API_KEY   — sign up free at resend.com
    NOTIFY_EMAIL     — the recipient
  Or:
    GMAIL_USER       — your gmail address
    GMAIL_APP_PASSWORD — Google App Password (myaccount.google.com → Security → 2-step → App passwords)
    NOTIFY_EMAIL     — recipient (can be same as GMAIL_USER)

Usage:
    python scripts/notify.py scan
    python scripts/notify.py resolutions
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("notify")


REPO_URL = "https://github.com/zhongdaweiai/clio"


# ---------------- Email senders ----------------


def send_resend(to: str, subject: str, html: str, sender: str | None = None) -> bool:
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        return False
    sender = sender or os.environ.get("RESEND_FROM", "Clio Bot <onboarding@resend.dev>")
    body = json.dumps({
        "from": sender,
        "to": [to],
        "subject": subject,
        "html": html,
    }).encode()
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Resend's Cloudflare WAF blocks the default Python urllib UA.
            # A browser-like UA gets through.
            "User-Agent": "Mozilla/5.0 (clio-bot; +https://github.com/zhongdaweiai/clio)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            log.info("resend: %s", r.read().decode()[:200])
        return True
    except urllib.error.HTTPError as exc:
        log.error("resend HTTP %s: %s", exc.code, exc.read().decode()[:300])
        return False
    except Exception as exc:
        log.error("resend failed: %s", exc)
        return False


def send_gmail(to: str, subject: str, html: str) -> bool:
    user = os.environ.get("GMAIL_USER")
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    if not user or not pwd:
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
            s.login(user, pwd)
            s.sendmail(user, [to], msg.as_string())
        log.info("gmail SMTP delivered")
        return True
    except Exception as exc:
        log.error("gmail SMTP failed: %s", exc)
        return False


def send_email(to: str, subject: str, html: str) -> bool:
    if send_resend(to, subject, html):
        return True
    if send_gmail(to, subject, html):
        return True
    log.warning("no working email transport; printing to log instead")
    log.info("\nTO: %s\nSUBJECT: %s\n\n%s", to, subject, html[:5000])
    return False


# ---------------- Content builders ----------------


def latest_scan() -> tuple[Path, dict] | None:
    pt = Path("paper_trades")
    files = sorted(pt.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    if not files:
        return None
    f = files[-1]
    return f, json.loads(f.read_text())


def render_scan_email(scan: dict) -> tuple[str, str]:
    today = scan["scan_date"]
    n_recs = scan["n_recommendations"]
    cost = scan["llm_usage"]["estimated_cost_usd"]
    n_scanned = scan["n_open_scanned"]

    subj = f"[Clio] {today} — {n_recs} signals from {n_scanned} markets (${cost:.2f})"

    rows = []
    for r in scan["recommendations"][:25]:
        side = "🟢 YES" if r["side"] == "BUY YES" else "🔴 NO"
        conf_color = {"high": "#0a7", "medium": "#888", "low": "#aaa"}.get(r["llm_confidence"], "#aaa")
        rows.append(f"""
        <tr>
            <td><strong>{side}</strong></td>
            <td>{r["qtype"]}</td>
            <td>{r["current_price"]:.2f} → <strong>{r["llm_pred"]:.2f}</strong></td>
            <td><strong>{r["edge"]:+.2f}</strong></td>
            <td>{r["days_remaining"]}d</td>
            <td><span style="color:{conf_color}">{r["llm_confidence"]}</span></td>
            <td>{r["size_pct_of_bankroll"]:.1f}%</td>
            <td><a href="{r["polymarket_url"]}" style="color:#0066cc">{r["question"][:75]}</a></td>
        </tr>
        """)

    table_rows = "".join(rows) if rows else '<tr><td colspan="8"><em>No signals at edge ≥ 0.10 today.</em></td></tr>'

    html = f"""
    <html><body style="font-family:-apple-system,sans-serif;max-width:900px;margin:auto;color:#222">
    <h2>Clio paper trade — {today}</h2>
    <p style="color:#666">
        Scanned <strong>{n_scanned}</strong> currently-open Polymarket markets ·
        <strong>{n_recs}</strong> signals at |edge| ≥ 0.10 ·
        Cost: ${cost:.2f}
    </p>

    <p>
        <a href="{REPO_URL}/blob/main/paper_trades/{today}.json"
           style="background:#0066cc;color:white;padding:8px 12px;text-decoration:none;border-radius:4px">
            View today's full JSON
        </a>
        &nbsp;&nbsp;
        <a href="{REPO_URL}/blob/main/paper_trades/SUMMARY.md"
           style="background:#0a7;color:white;padding:8px 12px;text-decoration:none;border-radius:4px">
            Cumulative PnL
        </a>
    </p>

    <h3>Top {min(25, n_recs)} signals (sorted by |edge|)</h3>
    <table cellpadding="6" style="border-collapse:collapse;width:100%;font-size:13px">
        <thead style="background:#f4f4f4">
            <tr>
                <th align="left">Side</th>
                <th align="left">Type</th>
                <th align="left">Mkt → LLM</th>
                <th align="left">Edge</th>
                <th align="left">Days</th>
                <th align="left">Conf</th>
                <th align="left">Size%</th>
                <th align="left">Question</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <p style="color:#666;font-size:12px;margin-top:24px">
        Forward paper trade. Predictions logged BEFORE outcomes are known.
        Realized PnL appears in
        <a href="{REPO_URL}/blob/main/paper_trades/SUMMARY.md">SUMMARY.md</a>
        as markets close.
    </p>
    </body></html>
    """
    return subj, html


def render_high_edge_alert(scan: dict, edge_threshold: float = 0.30) -> tuple[str, str] | None:
    """Build an alert email for the highest-edge signals (edge ≥ 0.30).

    Returns None if no signal qualifies.
    """
    high = [r for r in scan["recommendations"] if abs(r["edge"]) >= edge_threshold]
    if not high:
        return None
    today = scan["scan_date"]
    subj = f"⚡ [Clio HIGH EDGE] {len(high)} signals at |edge| ≥ {edge_threshold:.2f} on {today}"
    rows = []
    for r in sorted(high, key=lambda x: -abs(x["edge"])):
        side = "🟢 YES" if r["side"] == "BUY YES" else "🔴 NO"
        rows.append(f"""
        <tr>
            <td><strong>{side}</strong></td>
            <td>{r["qtype"]}</td>
            <td>{r["current_price"]:.2f} → <strong>{r["llm_pred"]:.2f}</strong></td>
            <td><strong style="color:#c00">{r["edge"]:+.2f}</strong></td>
            <td>{r["days_remaining"]}d</td>
            <td>{r["llm_confidence"]}</td>
            <td>{r["size_pct_of_bankroll"]:.1f}%</td>
            <td>
                <a href="{r["polymarket_url"]}">{r["question"][:80]}</a>
                <br><span style="color:#666;font-size:11px">{r["llm_reasoning"]}</span>
            </td>
        </tr>
        """)
    html = f"""
    <html><body style="font-family:-apple-system,sans-serif;max-width:900px;margin:auto;color:#222">
    <h2 style="color:#c00">⚡ HIGH-EDGE ALERT — {today}</h2>
    <p>The LLM is flagging {len(high)} signals with |edge| ≥ {edge_threshold:.2f}. These are the strongest mispricings detected today.</p>
    <table cellpadding="8" style="border-collapse:collapse;width:100%;font-size:13px">
        <thead style="background:#f4f4f4">
            <tr>
                <th>Side</th><th>Type</th><th>Mkt → LLM</th><th>Edge</th>
                <th>Days</th><th>Conf</th><th>Size</th><th>Question + reasoning</th>
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    <p style="color:#666;font-size:12px;margin-top:20px">
        Auto-generated by .github/workflows/daily-scan.yml. Threshold |edge| ≥ {edge_threshold:.2f}.
    </p>
    </body></html>
    """
    return subj, html


def render_resolution_email() -> tuple[str, str]:
    pt = Path("paper_trades")
    summary_path = pt / "summary.json"
    if not summary_path.exists():
        return "[Clio] Resolution check — no resolved trades yet", \
               "<p>No paper trades have resolved yet. Check back after some markets close.</p>"
    s = json.loads(summary_path.read_text())
    n = s["n_trades"]
    bankroll = s["current_bankroll"]
    initial = s["initial_bankroll"]
    ret_pct = 100 * s["total_return_pct"]
    hit = 100 * s["hit_rate"]
    dd = 100 * s["max_drawdown_pct"]
    sharpe = s.get("sharpe_annual_rough", 0)
    pf = s.get("profit_factor", 0)
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

    subj = f"[Clio] PnL update: ${bankroll:,.0f} ({ret_pct:+.1f}%) — {n} trades, {hit:.0f}% hit"

    html = f"""
    <html><body style="font-family:-apple-system,sans-serif;max-width:800px;margin:auto;color:#222">
    <h2>Clio paper trade — resolution update</h2>
    <p style="color:#666">As of {s["as_of"]}</p>

    <table cellpadding="10" style="font-size:14px;border-collapse:collapse;width:100%">
        <tr style="background:#f4f4f4">
            <td><strong>Bankroll</strong></td>
            <td>${initial:,.0f} → <strong>${bankroll:,.0f}</strong>
                <span style="color:{'#0a7' if ret_pct>=0 else '#c00'}">({ret_pct:+.2f}%)</span></td>
        </tr>
        <tr><td>Resolved trades</td><td>{n}</td></tr>
        <tr style="background:#f4f4f4"><td>Hit rate</td><td>{hit:.1f}%</td></tr>
        <tr><td>Profit factor</td><td>{pf_str}</td></tr>
        <tr style="background:#f4f4f4"><td>Sharpe (annual, rough)</td><td>{sharpe:.2f}</td></tr>
        <tr><td>Max drawdown</td><td>{dd:.1f}%</td></tr>
    </table>

    <p style="margin-top:20px">
        <a href="{REPO_URL}/blob/main/paper_trades/SUMMARY.md"
           style="background:#0066cc;color:white;padding:8px 12px;text-decoration:none;border-radius:4px">
            Full equity curve + recent trades
        </a>
    </p>

    <p style="color:#888;font-size:12px;margin-top:24px">
        Backtest expectation: +259% CAGR, +11.25%/mo. After ≥50 resolved trades,
        compare live to backtest. If realized hit rate ≥ 60% and live PnL ≥ 50%
        of backtest expectation, the strategy is real and not a training-cutoff
        artifact.
    </p>
    </body></html>
    """
    return subj, html


# ---------------- Main ----------------


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("scan", "resolutions", "high_edge", "welcome"):
        log.error("usage: notify.py scan | resolutions | high_edge | welcome")
        return 2

    to = os.environ.get("NOTIFY_EMAIL")
    if not to:
        log.warning("NOTIFY_EMAIL not set; skipping email")
        return 0

    cmd = sys.argv[1]

    if cmd == "scan":
        latest = latest_scan()
        if not latest:
            log.warning("no scan files yet")
            return 0
        path, scan = latest
        today = datetime.now(tz=timezone.utc).date().isoformat()
        if scan["scan_date"] != today:
            log.info("latest scan %s isn't today's; skipping", scan["scan_date"])
            return 0
        subj, html = render_scan_email(scan)

    elif cmd == "high_edge":
        latest = latest_scan()
        if not latest:
            return 0
        path, scan = latest
        result = render_high_edge_alert(scan)
        if result is None:
            log.info("no high-edge signals today, skipping alert")
            return 0
        subj, html = result

    elif cmd == "welcome":
        subj = "[Clio] Deployment confirmed — paper trading is live"
        html = """
        <html><body style="font-family:-apple-system,sans-serif;max-width:700px;margin:auto;color:#222">
        <h2>✅ Clio is live</h2>
        <p>Paper-trading deployment confirmed. From now on you'll receive:</p>
        <ul>
            <li><strong>Daily signal email</strong> — every day at ~14:00 UTC after the scan completes</li>
            <li><strong>High-edge alert</strong> — same scan, but a separate email if any signal has |edge| ≥ 0.30</li>
            <li><strong>Resolution + PnL update</strong> — every day at ~16:00 UTC if any market resolved that day</li>
            <li><strong>Weekly recap</strong> — Mondays at 13:00 UTC, summarizing the past 7 days</li>
        </ul>
        <p>All signals are also visible at any time on GitHub:</p>
        <ul>
            <li><a href="https://github.com/zhongdaweiai/clio/blob/main/paper_trades/ALL_SIGNALS.md">All signals (chronological)</a></li>
            <li><a href="https://github.com/zhongdaweiai/clio/blob/main/paper_trades/SUMMARY.md">PnL summary (live)</a></li>
            <li><a href="https://github.com/zhongdaweiai/clio/actions">Workflow runs (audit log)</a></li>
        </ul>
        <p>First trades expected to resolve: <strong>2026-04-30</strong> (Bitcoin $80K April, US x Iran 4-30 meeting, etc).</p>
        <p style="color:#666;font-size:12px;margin-top:30px">
            If a workflow fails GitHub also emails you (via your account-level settings).
            To pause everything, disable the workflows under
            <a href="https://github.com/zhongdaweiai/clio/actions">Actions</a>.
        </p>
        </body></html>
        """
    else:  # resolutions
        subj, html = render_resolution_email()

    if send_email(to, subj, html):
        log.info("email sent to %s", to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
