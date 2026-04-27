"""News Scout micro-agent.

Gathers evidence under the cutoff and converts each piece into a likelihood
ratio (LR) for Bayesian update. LR magnitudes are clipped per Constitution M2
to avoid pathological compounding from over-confident NLI.

The scout does not produce a final probability — that's the strategy
composer's job. It produces (Document, LR, rationale) triples.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from clio.agents.base import MicroAgent, LLMClient, parse_probability
from clio.frozen.corpus import Corpus, Document
from clio.frozen.harness import Market


@dataclass(frozen=True)
class Evidence:
    doc: Document
    lr: float  # likelihood ratio P(doc | YES) / P(doc | NO)
    rationale: str


class NewsScout(MicroAgent):
    name = "news_scout"

    def __init__(
        self,
        llm: LLMClient,
        version: str = "v1",
        max_docs: int = 5,
        lr_floor: float = 0.1,
        lr_ceil: float = 10.0,
    ) -> None:
        super().__init__(llm, version)
        self.max_docs = max_docs
        self.lr_floor = lr_floor
        self.lr_ceil = lr_ceil

    def __call__(
        self,
        market: Market,
        as_of: date,
        corpus: Corpus,
    ) -> list[Evidence]:
        docs = corpus.search(market.question, as_of=as_of, limit=self.max_docs)
        # Defense in depth: harness already promises this, but cheap to recheck.
        corpus.assert_no_leak(docs, as_of=as_of)

        evidence: list[Evidence] = []
        for doc in docs:
            lr = self._score_doc(market.question, doc)
            evidence.append(Evidence(doc=doc, lr=lr, rationale=""))
        return evidence

    def _score_doc(self, question: str, doc: Document) -> float:
        # NOTE: the instruction text below intentionally avoids the tokens
        # used as evidence stance markers ("supports", "contradicts", "favors",
        # "opposes"). Including them here would leak into prompt-matching
        # mock LLMs and produce false-positive evidence scoring.
        prompt = (
            "You are a Bayesian evidence assessor.\n"
            f"Question: {question}\n"
            f"Document title: {doc.title}\n"
            f"Document content: {doc.content[:1500]}\n"
            "Estimate the conditional likelihood ratio of this document under "
            "the YES vs NO hypothesis. "
            "Output a single decimal between 0 and 1 representing P(YES | this document alone).\n"
        )
        raw = self.llm.complete(prompt, max_tokens=16, temperature=0.0)
        p = parse_probability(raw, default=0.5)
        # Convert per-document P(YES|doc) to a likelihood ratio assuming flat prior.
        # P/(1-P) is the LR vs an uninformative prior of 0.5.
        if p <= 0.0:
            lr = self.lr_floor
        elif p >= 1.0:
            lr = self.lr_ceil
        else:
            lr = p / (1 - p)
        return min(self.lr_ceil, max(self.lr_floor, lr))
