"""Live market scanner.

Scan currently-OPEN Polymarket markets, identify ones where the trained
strategy (from iterate_v5_master) recommends a bet RIGHT NOW. Output a
ranked recommendation list with edge size, position sizing, and rationale.

This is the production-trader's morning report: "today's actionable mispricings".

Trains on the full v3 dataset (638 resolved markets), then scans live open
markets. The strategy parameters used are the ones that emerged from genetic
evolution + Pareto filtering.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from clio.frozen.harness import Market
from clio.research.evolve import evolve
from clio.research.strategies import StrategyParams, predict, simulate
from clio.research.tuner import grid_search
from clio.research.walk_forward import compute_base_rates


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("scanner")

_HEADERS = {"User-Agent": "Mozilla/5.0 (clio-research)", "Accept": "application/json"}


def _http_get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_v3():
    sys.path.insert(0, str(Path(__file__).parent))
    from live_fetch_v2 import classify_question_type  # noqa: E402

    with open("runs/live_iter/markets_v3.json") as f:
        payload = json.load(f)
    markets, qtypes = [], {}
    for d in payload["markets"]:
        m = Market(
            market_id=d["market_id"], question=d["question"], regime=d["regime"],
            observed_at=date.fromisoformat(d["observed_at"]),
            closes_at=date.fromisoformat(d["closes_at"]),
            timeline=tuple(date.fromisoformat(s) for s in d["timeline"]),
            market_prices={date.fromisoformat(k): float(v) for k, v in d["market_prices"].items()},
        )
        markets.append(m)
        qtypes[m.market_id] = d.get("qtype", "event")
    resolutions = {k: int(v) for k, v in payload["resolutions"].items()}
    return markets, qtypes, resolutions


def fetch_open_markets(limit: int = 200, min_volume: float = 100_000) -> list[dict]:
    """Pull currently-open markets from gamma."""
    log.info("fetching open Polymarket markets (volume >= %.0fK)...", min_volume / 1000)
    out: list[dict] = []
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
        except Exception as exc:
            log.warning("fetch failed offset=%d: %s", offset, exc)
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
            out.append(m)
        if len(payload) < page:
            break
        offset += page
        time.sleep(0.15)
        if len(out) >= limit:
            break
    return out[:limit]


def fetch_current_price(token_id: str) -> float | None:
    """Get the most recent CLOB mid price for this token."""
    url = (
        "https://clob.polymarket.com/prices-history?"
        + urllib.parse.urlencode({"market": token_id, "interval": "all", "fidelity": "1440"})
    )
    try:
        payload = _http_get(url, timeout=15)
    except Exception:
        return None
    history = payload.get("history") or []
    if not history:
        return None
    last = history[-1]
    try:
        return float(last.get("p"))
    except (TypeError, ValueError):
        return None


def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from live_fetch_v2 import classify_question_type  # noqa: E402

    out_dir = Path("runs/live_iter/scanner")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train the strategy on the full v3 dataset.
    log.info("loading v3 dataset for training...")
    markets, qtypes, resolutions = load_v3()
    base_rates = compute_base_rates(markets, qtypes, resolutions)
    log.info("base rates: %s", {k: round(v, 3) for k, v in base_rates.items()})

    # Quick grid for params; for production we'd snapshot the result of
    # iterate_v5_master, but here we re-tune on the full corpus to be honest.
    log.info("tuning strategy on full v3 corpus...")
    candidates = grid_search(
        markets, qtypes, resolutions, base_rates,
        objective="pnl",
        lambda_grid={
            "event":    [0.2, 0.3],
            "field":    [0.3, 0.4],
            "deadline": [0.0, 0.1],
            "durative": [0.0],
        },
        trend_alphas=[-0.25, 0.0, 0.25],
        time_decays=[0.0, 1.0],
        edge_thresholds=[0.05, 0.07],
        kelly_fractions=[0.0, 0.5],
        symmetric_options=[False],
    )
    chosen, train_res = candidates[0]
    log.info("chosen strategy:")
    log.info("  λ=%s trend=%.2f td=%.1f et=%.2f kelly=%.1f sym=%s",
             dict(chosen.shrink_lambda), chosen.trend_alpha, chosen.time_decay,
             chosen.edge_threshold, chosen.kelly_fraction, chosen.symmetric)
    log.info("  in-sample brier=%.4f pnl=%+0.3f trades=%d (NB: in-sample, not held-out)",
             train_res["brier_overall"], train_res["pnl"], train_res["n_trades"])

    log.info("\nscanning live open markets...")
    open_markets = fetch_open_markets(limit=300, min_volume=200_000)
    log.info("got %d open markets above volume floor", len(open_markets))

    today = datetime.now(tz=timezone.utc).date()
    recommendations: list[dict] = []
    seen_cids: set[str] = set()

    for i, m in enumerate(open_markets, 1):
        cid = m.get("conditionId") or m.get("id")
        if cid in seen_cids:
            continue
        seen_cids.add(cid)

        question = (m.get("question") or "").strip()
        if not question:
            continue

        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                continue
        if not outcomes or len(outcomes) != 2:
            continue

        token_ids = m.get("clobTokenIds")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except json.JSONDecodeError:
                continue
        if not token_ids or len(token_ids) != 2:
            continue

        end_str = m.get("endDate") or m.get("closedTime")
        if not end_str:
            continue
        try:
            end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00")).date()
        except ValueError:
            continue
        if end_date <= today:
            continue
        days_remaining = (end_date - today).days
        if days_remaining > 365:
            continue  # very long horizon: too much can change

        # YES is conventionally first; double-check.
        yes_idx = 0
        for j, o in enumerate(outcomes):
            if str(o).strip().lower() == "yes":
                yes_idx = j
                break
        yes_token = token_ids[yes_idx]

        # Get current price.
        cur_price = fetch_current_price(yes_token)
        time.sleep(0.10)
        if cur_price is None or not (0.02 < cur_price < 0.98):
            continue

        qtype = classify_question_type(question)
        if qtype == "durative":
            continue  # we don't have data on this type yet

        # Apply the strategy. We construct a one-step "synthetic Market" since
        # the strategy expects a Market object. The trend term needs prev price;
        # we don't have it for live, so set step=0.
        synthetic_market = Market(
            market_id=cid, question=question, regime=qtype,
            observed_at=today, closes_at=end_date,
            timeline=(today,),
            market_prices={today: cur_price},
        )
        pred = predict(chosen, synthetic_market, 0, qtype, base_rates)
        edge = pred - cur_price

        if abs(edge) < chosen.edge_threshold:
            continue

        side = "BUY YES" if edge > 0 else "BUY NO"
        # Confidence-weighted size
        if edge > 0:
            b = (1 - cur_price) / cur_price
            f_star = max(0.0, (pred * b - (1 - pred)) / b)
        else:
            b = cur_price / (1 - cur_price)
            f_star = max(0.0, ((1 - pred) * b - pred) / b)
        size_frac = chosen.notional + chosen.kelly_fraction * f_star * chosen.notional
        size_frac = min(size_frac, 0.10)

        recommendations.append({
            "market_id": cid,
            "question": question,
            "qtype": qtype,
            "current_price": cur_price,
            "strategy_pred": pred,
            "edge": edge,
            "side": side,
            "size_pct_of_bankroll": round(size_frac * 100, 2),
            "days_remaining": days_remaining,
            "end_date": str(end_date),
            "polymarket_url": f"https://polymarket.com/event/{m.get('slug') or cid}",
            "volume": m.get("volumeNum"),
        })

    recommendations.sort(key=lambda r: -abs(r["edge"]))

    log.info("\n%d recommendations from %d scanned markets", len(recommendations), len(open_markets))
    log.info("\n=== top 20 by edge ===")
    log.info("%-5s %-8s %-12s %-7s %-7s %-7s %-6s   %s",
             "side", "qtype", "price→pred", "edge", "size%", "days", "vol(K)", "question")
    for r in recommendations[:20]:
        vol_k = (r["volume"] or 0) / 1000
        log.info(
            "%-5s %-8s %.3f→%.3f  %+0.3f  %5.2f  %4d  %6.0fK  %s",
            "YES" if r["side"] == "BUY YES" else "NO ",
            r["qtype"][:8],
            r["current_price"], r["strategy_pred"],
            r["edge"], r["size_pct_of_bankroll"], r["days_remaining"],
            vol_k,
            r["question"][:60],
        )

    out = out_dir / f"recommendations_{today.isoformat()}.json"
    out.write_text(json.dumps({
        "scan_date": str(today),
        "strategy": chosen.to_dict(),
        "n_open_markets_scanned": len(open_markets),
        "n_recommendations": len(recommendations),
        "recommendations": recommendations,
    }, indent=2))
    log.info("\nwrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
