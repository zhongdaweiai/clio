"""Daily paper-trading scan.

Runs the LLM scanner on currently-OPEN Polymarket markets, saves
recommendations + reasoning + provenance to `paper_trades/YYYY-MM-DD.json`.

Idempotent: if today's file already exists, exits without re-running.
Resilient: if API fails on individual markets, skips them and logs.

Designed to be run by GitHub Actions cron once per day.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from clio.agents.llm_anthropic import AnthropicLLMClient, LLMForecaster
from clio.data.live_news import HackerNewsSource, MultiSource, WikipediaRevisionSource
from clio.data.date_validator import validate_published_date
from clio.frozen.corpus import Corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("paper_scan")

_HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}


def _http(url, timeout=30):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers=_HEADERS), timeout=timeout
    ).read().decode())


def fetch_open_markets(min_volume=200_000, max_days=60, limit=120):
    today = datetime.now(tz=timezone.utc).date()
    out = []
    offset = 0
    page = 100
    while len(out) < limit:
        url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode({
            "closed": "false", "active": "true",
            "limit": str(page), "offset": str(offset),
            "order": "volumeNum", "ascending": "false",
        })
        try:
            payload = _http(url)
        except Exception as exc:
            log.warning("page failed: %s", exc)
            break
        if not payload:
            break
        for m in payload:
            try:
                vol = float(m.get("volumeNum") or 0)
            except (TypeError, ValueError):
                continue
            if vol < min_volume:
                continue
            end_str = m.get("endDate") or m.get("closedTime")
            if not end_str:
                continue
            try:
                end_d = datetime.fromisoformat(end_str.replace("Z", "+00:00")).date()
            except ValueError:
                continue
            if end_d <= today:
                continue
            if (end_d - today).days > max_days:
                continue
            out.append(m)
        if len(payload) < page:
            break
        offset += page
        time.sleep(0.15)
        if len(out) >= limit:
            break
    return out[:limit]


def fetch_current_price(token_id):
    url = ("https://clob.polymarket.com/prices-history?"
           + urllib.parse.urlencode({"market": token_id, "interval": "all", "fidelity": "1440"}))
    try:
        p = _http(url, timeout=15)
    except Exception:
        return None
    h = p.get("history") or []
    if not h:
        return None
    try:
        return float(h[-1].get("p"))
    except (TypeError, ValueError):
        return None


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY env var required")
        return 1

    today = datetime.now(tz=timezone.utc).date()
    out_dir = Path("paper_trades")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{today.isoformat()}.json"
    if out_file.exists():
        log.info(f"{out_file} already exists, skipping today's scan")
        return 0

    log.info("daily paper trade scan: %s", today.isoformat())

    open_markets = fetch_open_markets(min_volume=200_000, max_days=60, limit=120)
    log.info("got %d open markets (vol>=200K, ≤60 days)", len(open_markets))

    news_src = MultiSource([
        HackerNewsSource(sleep_between=0.10),
        WikipediaRevisionSource(sleep_between=0.10, max_articles=2),
    ])

    sys.path.insert(0, str(Path(__file__).parent))
    from live_fetch_v2 import classify_question_type  # noqa

    base_rates = {"event": 0.204, "field": 0.085, "deadline": 0.324, "other": 0.20}
    llm = AnthropicLLMClient(model="claude-sonnet-4-6")
    forecaster = LLMForecaster(llm)

    recommendations = []
    skipped = {"no_price": 0, "no_outcomes": 0, "duplicates": 0, "errors": 0}
    seen_cids: set[str] = set()

    for i, m in enumerate(open_markets, 1):
        cid = m.get("conditionId") or m.get("id")
        if cid in seen_cids:
            skipped["duplicates"] += 1
            continue
        seen_cids.add(cid)
        question = (m.get("question") or "").strip()
        if not question:
            skipped["no_outcomes"] += 1
            continue

        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                skipped["no_outcomes"] += 1
                continue
        if not outcomes or len(outcomes) != 2:
            skipped["no_outcomes"] += 1
            continue

        token_ids = m.get("clobTokenIds")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except json.JSONDecodeError:
                skipped["no_outcomes"] += 1
                continue
        if not token_ids or len(token_ids) != 2:
            skipped["no_outcomes"] += 1
            continue

        end_d = datetime.fromisoformat((m.get("endDate") or "").replace("Z", "+00:00")).date()
        days_rem = (end_d - today).days
        yes_idx = next((j for j, o in enumerate(outcomes) if str(o).lower() == "yes"), 0)

        cur = fetch_current_price(token_ids[yes_idx])
        time.sleep(0.10)
        if cur is None or not (0.02 < cur < 0.98):
            skipped["no_price"] += 1
            continue

        qtype = classify_question_type(question)
        if qtype == "durative":
            continue

        # News
        try:
            arts = news_src.search(
                query=question,
                published_after=today - timedelta(days=21),
                published_before=end_d,
                max_results=8,
            )
        except Exception as exc:
            arts = []

        docs = []
        for art in arts:
            v = validate_published_date(art.published_at, art.url, art.raw_html)
            if v.confidence == "rejected" or v.consensus_date is None:
                continue
            docs.append(art.to_document(consensus_date=v.consensus_date, market_id="paper"))

        from clio.frozen.harness import Market

        synthetic = Market(
            market_id=cid, question=question, regime=qtype,
            observed_at=today, closes_at=end_d,
            timeline=(today,), market_prices={today: cur},
        )
        live_corpus = Corpus()
        live_corpus.add_many(docs)

        try:
            result = forecaster.forecast(
                synthetic, today, live_corpus,
                base_rate_hint=base_rates.get(qtype, 0.30),
            )
        except Exception as exc:
            skipped["errors"] += 1
            continue

        edge = result.p_yes - cur
        if abs(edge) < 0.10:
            continue  # below threshold, no recommendation

        # Position sizing — fixed-fraction Kelly with hard caps for paper
        # trading (matches the LLM_edge>=0.20 backtest tier).
        if edge > 0:
            b = (1 - cur) / cur
            f_star = max(0.0, (result.p_yes * b - (1 - result.p_yes)) / b)
        else:
            b = cur / (1 - cur)
            f_star = max(0.0, ((1 - result.p_yes) * b - result.p_yes) / b)
        size_frac = 0.10 + 1.2 * f_star * 0.10
        size_frac = min(size_frac, 0.25)

        # URL: prefer the parent event slug for multi-outcome markets (negRisk
        # events like FOMC rate brackets, sport O/U variants). Fall back to
        # market slug for true binary standalones.
        events = m.get("events") or []
        event_slug = events[0].get("slug") if events and isinstance(events[0], dict) else None
        url_slug = event_slug or m.get("slug") or cid

        rec = {
            "market_id": cid,
            "question": question,
            "qtype": qtype,
            "scan_date": today.isoformat(),
            "scan_iso": datetime.now(tz=timezone.utc).isoformat(),
            "current_price": cur,
            "llm_pred": result.p_yes,
            "llm_confidence": result.confidence,
            "llm_reasoning": result.reasoning,
            "edge": edge,
            "side": "BUY YES" if edge > 0 else "BUY NO",
            "size_pct_of_bankroll": round(size_frac * 100, 2),
            "days_remaining": days_rem,
            "end_date": end_d.isoformat(),
            "polymarket_url": f"https://polymarket.com/event/{url_slug}",
            "volume": m.get("volumeNum"),
            "n_news_docs": len(docs),
            "news_doc_ids": [d.doc_id for d in docs[:3]],
            "model": llm.model,
        }
        recommendations.append(rec)
        if i % 20 == 0:
            u = llm.usage_summary()
            log.info(f"[{i}/{len(open_markets)}] {len(recommendations)} recs, "
                     f"~${u['estimated_cost_usd']:.3f} spent")

    recommendations.sort(key=lambda r: -abs(r["edge"]))
    usage = llm.usage_summary()

    payload = {
        "scan_date": today.isoformat(),
        "scan_iso_utc": datetime.now(tz=timezone.utc).isoformat(),
        "model": llm.model,
        "n_open_scanned": len(open_markets),
        "n_recommendations": len(recommendations),
        "skipped": skipped,
        "llm_usage": usage,
        "recommendations": recommendations,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    log.info(f"wrote {out_file}")
    log.info(f"   {len(recommendations)} recs, ${usage['estimated_cost_usd']:.3f} spent")

    # Append to chronological summary file
    summary_path = out_dir / "ALL_SIGNALS.md"
    _append_signals_summary(summary_path, payload)
    return 0


def _append_signals_summary(path: Path, payload: dict) -> None:
    """Maintain a human-readable chronological log of all signals issued."""
    lines = []
    if path.exists():
        lines = path.read_text().splitlines()
    if not lines:
        lines = [
            "# Clio Paper Trading — All Signals",
            "",
            "Auto-generated. Each daily scan appends here. Predictions logged",
            "BEFORE outcomes are known (forward paper trade). Realized PnL",
            "computed in `paper_trades/RESOLVED.md` after markets close.",
            "",
        ]
    lines.append(f"\n## {payload['scan_date']}  ({len(payload['recommendations'])} signals, ${payload['llm_usage']['estimated_cost_usd']:.3f} cost)\n")
    if not payload["recommendations"]:
        lines.append("_No signals at edge ≥ 0.10 today._")
    else:
        lines.append("| side | qtype | mkt → LLM | edge | days | conf | size% | question |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in payload["recommendations"][:25]:
            side = "🟢 YES" if r["side"] == "BUY YES" else "🔴 NO"
            q_short = r["question"][:80].replace("|", "\\|")
            lines.append(
                f"| {side} | {r['qtype']} | {r['current_price']:.2f}→{r['llm_pred']:.2f} | "
                f"{r['edge']:+.2f} | {r['days_remaining']}d | {r['llm_confidence']} | "
                f"{r['size_pct_of_bankroll']}% | [{q_short}]({r['polymarket_url']}) |"
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
