"""Calibrator micro-agent.

Post-processes raw probabilities so that, on the holdout, predictions in bin k
empirically resolve YES at frequency ≈ k. Two modes:

- `IdentityCalibrator`: passthrough. Useful as a baseline.
- `IsotonicCalibrator`: learns a monotone non-decreasing mapping from raw probs
  to calibrated probs by pool-adjacent-violators. No external dependency.
- `TemperatureCalibrator`: simpler one-parameter fit (logit scaling).

The calibrator is fit on a training split of resolved markets and frozen at fit
time. Refitting on holdout is forbidden.
"""

from __future__ import annotations

from typing import Sequence

from clio.agents.base import MicroAgent, LLMClient


class Calibrator(MicroAgent):
    """Base class. Subclasses implement `_transform`."""

    name = "calibrator"

    def __init__(self, llm: LLMClient | None = None, version: str = "v1") -> None:
        # LLM is optional for calibrators; the default ones are pure math.
        if llm is None:
            llm = _NullLLM()
        super().__init__(llm, version)
        self._fitted = False

    def fit(self, raw_probs: Sequence[float], outcomes: Sequence[int]) -> None:
        self._fit(raw_probs, outcomes)
        self._fitted = True

    def _fit(self, raw_probs: Sequence[float], outcomes: Sequence[int]) -> None:
        pass

    def __call__(self, raw_prob: float) -> float:
        return self._transform(raw_prob)

    def _transform(self, raw_prob: float) -> float:
        return raw_prob


class IdentityCalibrator(Calibrator):
    pass


class TemperatureCalibrator(Calibrator):
    """Logit scaling: p' = sigmoid(T * logit(p)).

    T > 1 sharpens (pushes probs toward extremes), T < 1 softens. Fitted by
    a small grid search on Brier score.
    """

    def __init__(self, llm: LLMClient | None = None, version: str = "v1") -> None:
        super().__init__(llm, version)
        self.T = 1.0

    def _fit(self, raw_probs: Sequence[float], outcomes: Sequence[int]) -> None:
        import math

        def logit(p: float) -> float:
            p = min(1 - 1e-6, max(1e-6, p))
            return math.log(p / (1 - p))

        def sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        best_T, best_loss = 1.0, float("inf")
        for k in range(1, 41):
            T = k / 10
            loss = 0.0
            for p, o in zip(raw_probs, outcomes):
                p2 = sigmoid(T * logit(p))
                loss += (p2 - o) ** 2
            if loss < best_loss:
                best_loss = loss
                best_T = T
        self.T = best_T

    def _transform(self, raw_prob: float) -> float:
        import math

        p = min(1 - 1e-6, max(1e-6, raw_prob))
        x = math.log(p / (1 - p))
        return 1 / (1 + math.exp(-self.T * x))


class IsotonicCalibrator(Calibrator):
    """Pool-adjacent-violators fit. No sklearn dependency.

    After fitting, prediction is piecewise-constant on the sorted training probs.
    """

    def __init__(self, llm: LLMClient | None = None, version: str = "v1") -> None:
        super().__init__(llm, version)
        self._knots_x: list[float] = [0.0, 1.0]
        self._knots_y: list[float] = [0.0, 1.0]

    def _fit(self, raw_probs: Sequence[float], outcomes: Sequence[int]) -> None:
        if not raw_probs:
            return
        pairs = sorted(zip(raw_probs, outcomes), key=lambda x: x[0])
        xs = [float(p) for p, _ in pairs]
        ys = [float(o) for _, o in pairs]
        weights = [1.0] * len(ys)

        # PAV
        i = 0
        while i < len(ys) - 1:
            if ys[i] > ys[i + 1]:
                w = weights[i] + weights[i + 1]
                merged = (ys[i] * weights[i] + ys[i + 1] * weights[i + 1]) / w
                ys[i] = merged
                weights[i] = w
                del ys[i + 1]
                del xs[i + 1]
                del weights[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Force endpoints for safe extrapolation.
        if xs[0] > 0.0:
            xs.insert(0, 0.0)
            ys.insert(0, ys[0])
        if xs[-1] < 1.0:
            xs.append(1.0)
            ys.append(ys[-1])

        self._knots_x = xs
        self._knots_y = ys

    def _transform(self, raw_prob: float) -> float:
        x = min(1.0, max(0.0, raw_prob))
        # Linear interpolation between knots.
        xs, ys = self._knots_x, self._knots_y
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                if xs[i + 1] == xs[i]:
                    return ys[i]
                t = (x - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]


class _NullLLM:
    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return ""
