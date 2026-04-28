"""Researcher proposal schema and validation.

The agent outputs a structured `Proposal`. Anything outside the allowed
parameter set or out-of-range values is rejected before being applied
or PR'd.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal


# Whitelist of parameters the researcher is allowed to mutate, and the
# bounded range each must stay within. Any proposal outside these bounds
# is rejected before being applied or PR'd.
ALLOWED_PARAMS: dict[str, dict] = {
    "edge_threshold": {
        "min": 0.05, "max": 0.30, "default": 0.10,
        "file": "scripts/paper_trade_scan.py",
        "description": "Minimum |edge| for a signal to be issued. Higher = fewer, more confident signals.",
    },
    "max_position_pct": {
        "min": 0.05, "max": 0.40, "default": 0.25,
        "file": "scripts/paper_trade_scan.py",
        "description": "Hard cap on single-position size as fraction of bankroll.",
    },
    "kelly_fraction": {
        "min": 0.5, "max": 1.5, "default": 1.2,
        "file": "scripts/paper_trade_scan.py",
        "description": "Kelly multiplier — 1.0 = full Kelly, <1.0 = fractional.",
    },
    "notional_floor": {
        "min": 0.02, "max": 0.20, "default": 0.10,
        "file": "scripts/paper_trade_scan.py",
        "description": "Minimum position size before Kelly multiplier is applied.",
    },
    "min_volume": {
        "min": 50_000, "max": 1_000_000, "default": 200_000,
        "file": "scripts/paper_trade_scan.py",
        "description": "Skip markets with lifetime volume below this. Higher = better liquidity, fewer markets.",
    },
    "max_days_remaining": {
        "min": 14, "max": 120, "default": 60,
        "file": "scripts/paper_trade_scan.py",
        "description": "Skip markets with more than this many days to resolution.",
    },
}


@dataclass
class Proposal:
    decision: Literal["propose", "hold"]
    summary: str
    evidence: list[str]
    parameter_changes: dict[str, dict]  # {"param": {"old": ..., "new": ...}}
    expected_impact: str
    confidence: Literal["low", "medium", "high"]
    rollback: str
    raw: str = ""

    def to_json(self) -> dict:
        return {
            "decision": self.decision,
            "summary": self.summary,
            "evidence": self.evidence,
            "parameter_changes": self.parameter_changes,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "rollback": self.rollback,
        }


def parse_proposal(raw: str) -> Proposal | None:
    """Tolerant parser. Falls back to extracting JSON from any markdown block."""
    if not raw:
        return None
    raw = raw.strip()

    # Try strict JSON.
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fence.
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    if obj is None:
        # Last resort: find first { ... } block.
        m = re.search(r"\{[^{}]*\"decision\"[^{}]*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    if obj is None:
        return None

    decision = obj.get("decision", "hold")
    if decision not in ("propose", "hold"):
        decision = "hold"
    confidence = obj.get("confidence", "low")
    if confidence not in ("low", "medium", "high"):
        confidence = "low"

    return Proposal(
        decision=decision,
        summary=str(obj.get("summary", ""))[:300],
        evidence=[str(e)[:300] for e in (obj.get("evidence") or [])][:6],
        parameter_changes=dict(obj.get("parameter_changes") or {}),
        expected_impact=str(obj.get("expected_impact", ""))[:300],
        confidence=confidence,
        rollback=str(obj.get("rollback", ""))[:300],
        raw=raw,
    )


def validate_proposal(p: Proposal) -> tuple[bool, list[str]]:
    """Return (is_valid, list_of_errors).

    Strict guards:
    - hold proposals must have empty parameter_changes
    - propose must have ≥1 evidence item
    - all parameter names must be in ALLOWED_PARAMS
    - all proposed values must be within range
    - delta from current must not exceed 50% of the allowed range (anti-overshoot)
    """
    errors: list[str] = []

    if p.decision == "hold":
        if p.parameter_changes:
            errors.append("`hold` decision must have empty parameter_changes")
        return (len(errors) == 0, errors)

    if not p.evidence:
        errors.append("`propose` decision requires at least one evidence claim")

    for name, change in p.parameter_changes.items():
        if name not in ALLOWED_PARAMS:
            errors.append(f"parameter '{name}' is not in the whitelist")
            continue
        bounds = ALLOWED_PARAMS[name]
        try:
            old = float(change.get("old"))
            new = float(change.get("new"))
        except (TypeError, ValueError):
            errors.append(f"'{name}' missing or invalid old/new values")
            continue
        if not (bounds["min"] <= new <= bounds["max"]):
            errors.append(
                f"'{name}' new value {new} out of range [{bounds['min']}, {bounds['max']}]"
            )
            continue
        # Anti-overshoot: change ≤ 50% of allowed range
        max_delta = (bounds["max"] - bounds["min"]) * 0.5
        if abs(new - old) > max_delta:
            errors.append(
                f"'{name}' delta |{old} → {new}| = {abs(new-old):.4f} > 50% of range "
                f"({max_delta:.4f}) — too aggressive in one step"
            )

    return (len(errors) == 0, errors)
