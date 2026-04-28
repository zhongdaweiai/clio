"""Parallelize the LLM forecast precompute step (8 concurrent calls).

Reads the existing cache and only fills missing entries. Safe to run while
the serial llm_run.py is also writing — last-writer-wins on cache, but most
markets will be uniquely fetched by one or the other.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from clio.agents.llm_anthropic import AnthropicLLMClient, LLMForecaster
from clio.cli import _load_markets_payload
from clio.data.news_pipeline import read_corpus_jsonl
from clio.research.llm_backtest import ForecastCache
from clio.research.walk_forward import compute_base_rates


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("par")


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return 1

    markets, _ = _load_markets_payload("runs/live_iter/markets_v3.json")
    corpus = read_corpus_jsonl("runs/live_iter/news_v3.jsonl")
    qtypes = {}
    with open("runs/live_iter/markets_v3.json") as f:
        payload = json.load(f)
    for d in payload["markets"]:
        qtypes[d["market_id"]] = d.get("qtype", "event")
    resolutions = {k: int(v) for k, v in payload["resolutions"].items()}
    base_rates = compute_base_rates(markets, qtypes, resolutions)
    log.info("loaded %d markets, %d news, base_rates=%s",
             len(markets), len(corpus), {k: round(v, 3) for k, v in base_rates.items()})

    cache = ForecastCache(Path("runs/live_iter/llm_forecast_cache.json"))
    log.info("cache entries: %d", len(cache))

    model = "claude-sonnet-4-6"
    llm = AnthropicLLMClient(model=model)
    forecaster = LLMForecaster(llm)

    # Find missing markets (first as_of only).
    missing = []
    for m in markets:
        as_of = m.timeline[0]
        if as_of >= m.closes_at:
            continue
        if cache.get(m.market_id, as_of, model) is None:
            missing.append((m, as_of))
    log.info("missing forecasts: %d", len(missing))

    if not missing:
        log.info("nothing to do")
        return 0

    cache_lock = threading.Lock()
    save_lock = threading.Lock()
    counter = {"done": 0, "errors": 0}

    def _one(args):
        m, as_of = args
        qt = qtypes[m.market_id]
        br = base_rates.get(qt, 0.30)
        try:
            res = forecaster.forecast(m, as_of, corpus, base_rate_hint=br)
            with cache_lock:
                cache.put(m.market_id, as_of, model, {
                    "p_yes": res.p_yes, "confidence": res.confidence,
                    "reasoning": res.reasoning,
                })
                counter["done"] += 1
                if counter["done"] % 20 == 0:
                    with save_lock:
                        cache.save()
                    u = llm.usage_summary()
                    log.info(f"  done={counter['done']}/{len(missing)}  "
                             f"~${u['estimated_cost_usd']:.2f}")
        except Exception as exc:
            counter["errors"] += 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(_one, missing):
            pass
    cache.save()

    u = llm.usage_summary()
    log.info(f"done: {counter['done']} new, {counter['errors']} errors in {time.time()-t0:.0f}s")
    log.info(f"usage: {u['calls']} calls, {u['tokens_in']} in, {u['tokens_out']} out, "
             f"~${u['estimated_cost_usd']:.2f}")
    log.info(f"cache total: {len(cache)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
