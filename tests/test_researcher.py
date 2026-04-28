"""Researcher agent: parser + validator tests.

These guard the safety properties:
- whitelist enforcement
- bounded ranges
- max-delta-per-step
- hold proposals can't carry parameter changes
"""

import json

import pytest

from clio.research.agent.proposal import (
    ALLOWED_PARAMS,
    parse_proposal,
    validate_proposal,
)


def test_parse_strict_json():
    raw = json.dumps({
        "decision": "propose",
        "summary": "test",
        "evidence": ["e1"],
        "parameter_changes": {"edge_threshold": {"old": 0.10, "new": 0.12}},
        "expected_impact": "x",
        "confidence": "medium",
        "rollback": "revert if",
    })
    p = parse_proposal(raw)
    assert p.decision == "propose"
    assert p.parameter_changes == {"edge_threshold": {"old": 0.10, "new": 0.12}}
    assert p.confidence == "medium"


def test_parse_json_in_markdown_fence():
    raw = '```json\n{"decision":"hold","summary":"no","evidence":[],"parameter_changes":{},"expected_impact":"","confidence":"low","rollback":""}\n```'
    p = parse_proposal(raw)
    assert p.decision == "hold"


def test_parse_garbage_returns_none():
    assert parse_proposal("hello world") is None
    assert parse_proposal("") is None


def test_parse_invalid_decision_falls_back_to_hold():
    raw = json.dumps({
        "decision": "go_yolo",
        "summary": "x", "evidence": [], "parameter_changes": {},
        "expected_impact": "", "confidence": "high", "rollback": "",
    })
    p = parse_proposal(raw)
    assert p.decision == "hold"


def test_validate_hold_with_changes_fails():
    raw = json.dumps({
        "decision": "hold", "summary": "x",
        "evidence": [], "parameter_changes": {"edge_threshold": {"old": 0.1, "new": 0.15}},
        "expected_impact": "", "confidence": "low", "rollback": "",
    })
    p = parse_proposal(raw)
    ok, errs = validate_proposal(p)
    assert not ok
    assert any("hold" in e for e in errs)


def test_validate_propose_without_evidence_fails():
    raw = json.dumps({
        "decision": "propose", "summary": "x",
        "evidence": [],
        "parameter_changes": {"edge_threshold": {"old": 0.1, "new": 0.12}},
        "expected_impact": "", "confidence": "medium", "rollback": "",
    })
    p = parse_proposal(raw)
    ok, errs = validate_proposal(p)
    assert not ok
    assert any("evidence" in e for e in errs)


def test_validate_unknown_param_rejected():
    raw = json.dumps({
        "decision": "propose", "summary": "x",
        "evidence": ["e"],
        "parameter_changes": {"yolo_factor": {"old": 1, "new": 2}},
        "expected_impact": "", "confidence": "medium", "rollback": "",
    })
    p = parse_proposal(raw)
    ok, errs = validate_proposal(p)
    assert not ok
    assert any("whitelist" in e for e in errs)


def test_validate_out_of_range_rejected():
    # edge_threshold range is [0.05, 0.30]
    raw = json.dumps({
        "decision": "propose", "summary": "x",
        "evidence": ["e"],
        "parameter_changes": {"edge_threshold": {"old": 0.10, "new": 0.50}},
        "expected_impact": "", "confidence": "medium", "rollback": "",
    })
    p = parse_proposal(raw)
    ok, errs = validate_proposal(p)
    assert not ok
    assert any("out of range" in e for e in errs)


def test_validate_overshoot_rejected():
    # edge_threshold range = [0.05, 0.30] = 0.25 wide. Max delta per step = 50% = 0.125.
    # 0.10 → 0.30 = delta 0.20 > 0.125 → rejected.
    raw = json.dumps({
        "decision": "propose", "summary": "x",
        "evidence": ["e"],
        "parameter_changes": {"edge_threshold": {"old": 0.10, "new": 0.30}},
        "expected_impact": "", "confidence": "high", "rollback": "",
    })
    p = parse_proposal(raw)
    ok, errs = validate_proposal(p)
    assert not ok
    assert any("too aggressive" in e for e in errs)


def test_validate_small_propose_passes():
    raw = json.dumps({
        "decision": "propose",
        "summary": "Lower edge threshold by 0.02",
        "evidence": ["8 mid-band trades have 75% hit vs 60% overall"],
        "parameter_changes": {"edge_threshold": {"old": 0.10, "new": 0.08}},
        "expected_impact": "+5 trades/week",
        "confidence": "medium",
        "rollback": "revert if hit rate < 55% after 14 days",
    })
    p = parse_proposal(raw)
    ok, errs = validate_proposal(p)
    assert ok, errs


def test_allowed_params_self_consistent():
    """Every allowed param has min < default < max and a description."""
    for name, bounds in ALLOWED_PARAMS.items():
        assert bounds["min"] < bounds["default"] <= bounds["max"], name
        assert bounds["description"], name
        assert bounds["file"], name
