"""Real LLM forecaster using Anthropic Claude.

Replaces the rule-based scout for the most ambitious experiments.
Reads the question, the available news (cutoff-enforced), and produces
a calibrated probability with reasoning.

The forecaster has two modes:
- `score_doc`: per-document Bayesian likelihood ratio (fits the existing
  NewsScout interface)
- `forecast`: end-to-end, takes question + all news + market price and
  outputs P(YES). This is what we use for the production strategy.

The prompt is engineered for calibration over confidence: the LLM is
explicitly told that being well-calibrated matters more than being
"right" on any single question. Outputs structured JSON for parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date

from clio.agents.base import LLMClient, MicroAgent
from clio.frozen.corpus import Corpus, Document
from clio.frozen.harness import Market


log = logging.getLogger(__name__)


_DEFAULT_MODEL = "claude-sonnet-4-6"  # quality/cost balance
_HAIKU = "claude-haiku-4-5-20251001"
_OPUS = "claude-opus-4-7"


_FORECAST_SYSTEM = """You are a calibrated probabilistic forecaster for prediction markets.

Your goal is CALIBRATION over CONFIDENCE. Saying 80% means it should happen 80% of the time, not 100%. If you're unsure, say so by giving a probability close to 50% or to the empirical base rate. Overconfidence is the most expensive failure mode.

You do not see the market price for the question — predict only based on the question text and the news evidence provided. Use evidence chronologically and only the dates shown.

Output format: respond with a single JSON object on one line:
  {"p_yes": <float 0..1>, "confidence": "low"|"medium"|"high", "reasoning": "<one short sentence>"}

No other text. No markdown. JSON only."""


@dataclass
class ForecastResult:
    p_yes: float
    confidence: str
    reasoning: str
    raw: str = ""


class AnthropicLLMClient:
    """Implements LLMClient protocol using Anthropic SDK."""

    def __init__(self, model: str = _DEFAULT_MODEL, max_retries: int = 2) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("pip install anthropic to use AnthropicLLMClient") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY env var must be set")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.calls += 1
                self.tokens_in += r.usage.input_tokens
                self.tokens_out += r.usage.output_tokens
                return r.content[0].text
            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                log.warning("LLM call failed after %d retries: %s", self.max_retries, exc)
                return ""
        return ""

    def usage_summary(self) -> dict:
        return {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            # Sonnet 4.6 pricing (USD per 1M tokens)
            "estimated_cost_usd": (self.tokens_in / 1_000_000 * 3) + (self.tokens_out / 1_000_000 * 15),
        }


class LLMForecaster:
    """End-to-end forecaster: question + news → calibrated P(YES).

    This replaces the (BaseRater, NewsScout, BayesianStrategy) pipeline for
    the LLM-augmented experiments. The LLM is the entire forecaster.
    """

    def __init__(self, llm: AnthropicLLMClient, max_news_chars: int = 800,
                 max_news_per_market: int = 6) -> None:
        self.llm = llm
        self.max_news_chars = max_news_chars
        self.max_news_per_market = max_news_per_market

    def forecast(
        self,
        market: Market,
        as_of: date,
        corpus: Corpus,
        base_rate_hint: float | None = None,
    ) -> ForecastResult:
        docs = corpus.search(market.question, as_of=as_of, limit=self.max_news_per_market)
        corpus.assert_no_leak(docs, as_of=as_of)

        # Build the news block.
        if docs:
            news_block_parts = []
            for d in docs[: self.max_news_per_market]:
                content = d.content[: self.max_news_chars]
                news_block_parts.append(
                    f"[{d.published_at.isoformat()}] {d.title}\n{content}"
                )
            news_block = "\n\n---\n\n".join(news_block_parts)
        else:
            news_block = "(no relevant news found before the cutoff date)"

        hint_line = (
            f"\nReference: in our historical dataset, questions like this one have resolved YES at a base rate of approximately {base_rate_hint:.0%}. "
            f"This is a prior, not a fact about the specific question."
            if base_rate_hint is not None else ""
        )

        prompt = (
            f"QUESTION: {market.question}\n"
            f"OBSERVATION DATE (cutoff): {as_of.isoformat()}\n"
            f"MARKET CLOSES AT: {market.closes_at.isoformat()}\n"
            f"DAYS REMAINING: {(market.closes_at - as_of).days}"
            f"{hint_line}\n\n"
            f"NEWS AVAILABLE BEFORE CUTOFF (most recent first):\n{news_block}\n\n"
            f"What is your calibrated probability that this question resolves YES?"
        )

        raw = self.llm.complete(
            f"{_FORECAST_SYSTEM}\n\n{prompt}",
            max_tokens=200,
            temperature=0.0,
        )
        return _parse_forecast(raw)


def _parse_forecast(raw: str) -> ForecastResult:
    """Tolerant JSON parser. Falls back to extracting a probability number."""
    if not raw:
        return ForecastResult(p_yes=0.5, confidence="low", reasoning="", raw="")
    raw = raw.strip()

    # Try strict JSON first.
    try:
        obj = json.loads(raw)
        p = float(obj.get("p_yes", 0.5))
        return ForecastResult(
            p_yes=min(0.99, max(0.01, p)),
            confidence=obj.get("confidence", "medium"),
            reasoning=obj.get("reasoning", ""),
            raw=raw,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Fallback: find a {...} substring
    m = re.search(r"\{[^{}]*\"p_yes\"[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            p = float(obj.get("p_yes", 0.5))
            return ForecastResult(
                p_yes=min(0.99, max(0.01, p)),
                confidence=obj.get("confidence", "medium"),
                reasoning=obj.get("reasoning", ""),
                raw=raw,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    # Last resort: any number that looks like a probability
    m2 = re.search(r"(0?\.\d+|0|1)", raw)
    if m2:
        try:
            p = float(m2.group(1))
            return ForecastResult(
                p_yes=min(0.99, max(0.01, p)),
                confidence="low",
                reasoning="(parser fallback)",
                raw=raw,
            )
        except ValueError:
            pass

    return ForecastResult(p_yes=0.5, confidence="low", reasoning="(unparseable)", raw=raw)
