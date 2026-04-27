"""Mutable micro-agent layer.

Each micro-agent has one job and one evaluation surface. They evolve
independently. The MVP ships with three: Base-rater, News Scout, Calibrator.

Designed-but-not-implemented: Decomposer, Devil's Advocate, Sizer, Regime
Classifier, Red Team. These are the next agents to add.
"""

from clio.agents.base import LLMClient, MockLLMClient, MicroAgent
from clio.agents.base_rater import BaseRater
from clio.agents.news_scout import NewsScout, Evidence
from clio.agents.calibrator import Calibrator

__all__ = [
    "LLMClient",
    "MockLLMClient",
    "MicroAgent",
    "BaseRater",
    "NewsScout",
    "Evidence",
    "Calibrator",
]
