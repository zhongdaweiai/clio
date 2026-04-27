"""Base-rater micro-agent.

Reference-class forecasting only. Does not look at news. Estimates the prior
probability that the event of interest happens given a coarse class label.

Why this exists as its own agent: most forecasting failures are not "I missed
recent news" — they are "I never asked what fraction of similar past events
actually resolved YES". Base rates are unreasonably effective; isolating them
forces the rest of the pipeline to *update from a prior*, not invent from scratch.
"""

from __future__ import annotations

from datetime import date
from typing import Mapping

from clio.agents.base import MicroAgent, LLMClient, parse_probability
from clio.frozen.harness import Market


_DEFAULT_BASE_RATES: dict[str, float] = {
    "election": 0.50,
    "sport": 0.50,
    "financial": 0.40,
    "geo": 0.20,
    "scientific": 0.30,
    "corporate": 0.35,
    "weather": 0.50,
    "cultural": 0.50,
}


class BaseRater(MicroAgent):
    """Returns P(event) under the reference class only.

    Two paths:
    - If `regime_priors` covers the market's regime, use that as the canonical
      reference class. This is the *audit-friendly* path — every base rate is
      traceable to a written-down number.
    - Otherwise, ask the LLM for a base rate using a strict no-news prompt.
    """

    name = "base_rater"

    def __init__(
        self,
        llm: LLMClient,
        version: str = "v1",
        regime_priors: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(llm, version)
        self.regime_priors = dict(regime_priors or _DEFAULT_BASE_RATES)

    def __call__(self, market: Market, as_of: date) -> float:
        if market.regime in self.regime_priors:
            return self.regime_priors[market.regime]

        prompt = (
            "You are a strict base-rate forecaster.\n"
            "Do NOT use any news, polls, or recent events.\n"
            "Estimate the prior probability of this event purely from its "
            "reference class.\n"
            f"Question: {market.question}\n"
            f"Regime hint: {market.regime}\n"
            "Reply with a single decimal number between 0 and 1.\n"
        )
        raw = self.llm.complete(prompt, max_tokens=16, temperature=0.0)
        return parse_probability(raw, default=0.5)
