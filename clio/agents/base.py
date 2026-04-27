"""Common types for the micro-agent layer.

Two things matter here:

1. `LLMClient` is a Protocol. Tests use `MockLLMClient`. Production can plug in
   `AnthropicLLMClient` (in clio.agents.llm_anthropic, optional).
2. `MicroAgent` is the interface that the strategy composer talks to. Concrete
   agents subclass it, declare their `name` and `version`, and implement
   `__call__` with whatever signature makes sense for that agent's job.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str: ...


class MockLLMClient:
    """A deterministic, test-friendly LLM stand-in.

    Behavior is driven by simple prompt-pattern rules. Tests can register
    additional rules. The mock is intentionally dumb — its job is to make the
    plumbing testable, not to simulate real LLM behavior.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[re.Pattern, str]] = []
        self._default = "0.5"

    def register(self, pattern: str, response: str) -> None:
        self._rules.append((re.compile(pattern, re.IGNORECASE | re.DOTALL), response))

    def set_default(self, response: str) -> None:
        self._default = response

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        for pat, resp in self._rules:
            if pat.search(prompt):
                return resp
        return self._default


@dataclass(frozen=True)
class AgentMeta:
    name: str
    version: str


class MicroAgent(abc.ABC):
    """Base for all micro-agents. Concrete subclasses define their own __call__."""

    def __init__(self, llm: LLMClient, version: str = "v1") -> None:
        self.llm = llm
        self.version = version

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def meta(self) -> AgentMeta:
        return AgentMeta(name=self.name, version=self.version)


def parse_probability(text: str, default: float = 0.5) -> float:
    """Tolerant parser for LLM-style numeric outputs.

    Accepts: "0.62", "62%", "probability: 0.62", "Answer: 62 percent".
    Falls back to `default` if nothing parseable is found.
    """
    text = text.strip()
    pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    if pct:
        v = float(pct.group(1)) / 100
        return min(1.0, max(0.0, v))
    num = re.search(r"(?<![\d\.])(0?\.\d+|1\.0+|0|1)(?![\d])", text)
    if num:
        v = float(num.group(1))
        return min(1.0, max(0.0, v))
    return default
