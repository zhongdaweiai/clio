"""LLM-powered live scanner.

Pull currently-open Polymarket markets, fetch contemporaneous news, ask
Claude for a calibrated probability per market, identify mispricings.

Output: ranked recommendation list with the LLM's reasoning attached.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from clio.agents.llm_anthropic import AnthropicLLMClient, LLMForecaster
from clio.data.live_news import HackerNewsSource, MultiSource, WikipediaRevisionSource
from clio.frozen.corpus import Corpus, Document


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("llm_scan")

_HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}


def _http_get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_open_markets(limit: int = 200, min_volume: float = 100_000,
                        max_days_remaining: int = 60):
    """Pull open markets, filtering for short-horizon."""
    today = datetime.now(tz=timezone.utc).date()
    out = []
    offset = 0
    page = 100
    while len(out) < limit:
        params = {
            "closed": "false", "active": "true",
            "limit": str(page), "offset": str(offset),
            "order": "volumeNum", "ascending": "false",
        }
        url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode(params)
        try:
            payload = _http_get(url)
        except Exception:
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
            days_rem = (end_d - today).days
            if days_rem > max_days_remaining:
                continue
            out.append(m)
        if len(payload) < page:
            break
        offset += page
        time.sleep(0.15)
        if len(out) >= limit:
            break
    return out[:limit]


def fetch_current_price(token_id: str):
    url = (
        "https://clob.polymarket.com/prices-history?"
        + urllib.parse.urlencode({"market": token_id, "interval": "all", "fidelity": "1440"})
    )
    try:
        p = _http_get(url, timeout=15)
    except Exception:
        return None
    h = p.get("history") or []
    if not h:
        return None
    last = h[-1]
    try:
        return float(last.get("p"))
    except (TypeError, ValueError):
        return None


def _news_for_market(question: str, observed: date, closes: date, news_src):
    from datetime import timedelta
    from clio.data.date_validator import validate_published_date

    arts = news_src.search(
        query=question,
        published_after=observed - timedelta(days=21),
        published_before=closes,
        max_results=8,
    )
    docs = []
    for art in arts:
        v = validate_published_date(art.published_at, art.url, art.raw_html)
        if v.confidence == "rejected" or v.consensus_date is None:
            continue
        d = art.to_document(consensus_date=v.consensus_date, market_id="live")
        docs.append(d)
    return docs


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set")
        return 1

    today = datetime.now(tz=timezone.utc).date()
    log.info("scanning Polymarket for currently-open short-horizon markets...")
    open_markets = fetch_open_markets(limit=80, min_volume=200_000, max_days_remaining=45)
    log.info("got %d open markets (vol>=200K, ≤45 days remaining)", len(open_markets))

    news_src = MultiSource([
        HackerNewsSource(sleep_between=0.10),
        WikipediaRevisionSource(sleep_between=0.10, max_articles=2),
    ])

    llm = AnthropicLLMClient(model="claude-sonnet-4-6")
    forecaster = LLMForecaster(llm)

    base_rates = {"event": 0.204, "field": 0.085, "deadline": 0.324}
    sys.path.insert(0, str(Path(__file__).parent))
    from live_fetch_v2 import classify_question_type  # noqa

    recommendations = []
    for i, m in enumerate(open_markets, 1):
        cid = m.get("conditionId") or m.get("id")
        question = (m.get("question") or "").strip()
        if not question:
            continue

        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if not outcomes or len(outcomes) != 2:
            continue
        token_ids = m.get("clobTokenIds")
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if not token_ids or len(token_ids) != 2:
            continue

        end_d = datetime.fromisoformat((m.get("endDate") or "").replace("Z", "+00:00")).date()
        days_rem = (end_d - today).days
        yes_idx = next((j for j, o in enumerate(outcomes) if str(o).lower() == "yes"), 0)

        cur = fetch_current_price(token_ids[yes_idx])
        time.sleep(0.10)
        if cur is None or not (0.02 < cur < 0.98):
            continue

        qtype = classify_question_type(question)
        if qtype == "durative":
            continue

        # News
        docs = _news_for_market(question, today, end_d, news_src)

        # Synthesize a Market for the forecaster.
        from clio.frozen.harness import Market

        synthetic = Market(
            market_id=cid, question=question, regime=qtype,
            observed_at=today, closes_at=end_d,
            timeline=(today,), market_prices={today: cur},
        )
        live_corpus = Corpus()
        live_corpus.add_many(docs)

        result = forecaster.forecast(synthetic, today, live_corpus,
                                      base_rate_hint=base_rates.get(qtype, 0.30))
        edge = result.p_yes - cur

        rec = {
            "market_id": cid, "question": question, "qtype": qtype,
            "current_price": cur, "llm_pred": result.p_yes,
            "llm_confidence": result.confidence,
            "llm_reasoning": result.reasoning,
            "edge": edge,
            "side": "BUY YES" if edge > 0 else "BUY NO",
            "days_remaining": days_rem,
            "n_news_docs": len(docs),
            "polymarket_url": (
                f"https://polymarket.com/event/"
                f"{(m.get('events') or [{}])[0].get('slug') or m.get('slug') or cid}"
            ),
            "volume": m.get("volumeNum"),
        }
        if abs(edge) >= 0.10:
            recommendations.append(rec)
        if i % 10 == 0:
            u = llm.usage_summary()
            log.info(f"[{i}/{len(open_markets)}] {len(recommendations)} recs so far, "
                     f"~${u['estimated_cost_usd']:.3f} spent")

    recommendations.sort(key=lambda r: -abs(r["edge"]))

    log.info("\n=== TOP 25 RECOMMENDATIONS BY EDGE (LLM-augmented) ===")
    log.info(f"{'side':<4} {'qtype':<8} {'price→pred':<14} {'edge':>7} {'days':>4} {'conf':>6}  question")
    log.info("-" * 130)
    for r in recommendations[:25]:
        side = "YES" if r["side"] == "BUY YES" else "NO"
        log.info(
            f"{side:<4} {r['qtype']:<8} {r['current_price']:.3f}→{r['llm_pred']:.3f}  "
            f"{r['edge']:+.3f} {r['days_remaining']:>3}d {r['llm_confidence']:>6}  {r['question'][:65]}"
        )

    out_dir = Path("runs/live_iter/scanner")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"llm_recommendations_{today.isoformat()}.json"
    out.write_text(json.dumps({
        "scan_date": str(today), "model": llm.model,
        "n_open_markets_scanned": len(open_markets),
        "n_recommendations": len(recommendations),
        "llm_usage": llm.usage_summary(),
        "recommendations": recommendations,
    }, indent=2))
    log.info(f"wrote {out}")

    u = llm.usage_summary()
    log.info(f"\nLLM cost this run: ${u['estimated_cost_usd']:.3f} ({u['calls']} calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
