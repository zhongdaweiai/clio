"""Calibrator tests."""

import random

from clio.agents.calibrator import (
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
)
from clio.frozen.scoring import calibration_ece


def _make_overconfident_data(n: int = 400, seed: int = 0):
    """Synthesize a forecaster who says 0.9 but is right 70% of the time, etc."""
    rng = random.Random(seed)
    raw_probs, outcomes = [], []
    miscalibration_pairs = [
        (0.1, 0.30),
        (0.3, 0.40),
        (0.5, 0.55),
        (0.7, 0.60),
        (0.9, 0.70),
    ]
    for raw, true_freq in miscalibration_pairs:
        for _ in range(n // len(miscalibration_pairs)):
            raw_probs.append(raw)
            outcomes.append(1 if rng.random() < true_freq else 0)
    return raw_probs, outcomes


def test_identity_passthrough():
    cal = IdentityCalibrator()
    cal.fit([0.1, 0.5, 0.9], [0, 1, 1])
    assert cal(0.42) == 0.42


def test_isotonic_improves_ece_on_miscalibrated_data():
    raw, out = _make_overconfident_data()
    raw_ece = calibration_ece(raw, out)

    cal = IsotonicCalibrator()
    cal.fit(raw, out)
    calibrated = [cal(p) for p in raw]
    cal_ece = calibration_ece(calibrated, out)

    # The isotonic fit should reduce ECE on the same data substantially.
    assert cal_ece < raw_ece - 0.05


def test_isotonic_is_monotone():
    raw, out = _make_overconfident_data()
    cal = IsotonicCalibrator()
    cal.fit(raw, out)

    xs = [i / 100 for i in range(101)]
    ys = [cal(x) for x in xs]
    for i in range(len(ys) - 1):
        assert ys[i] <= ys[i + 1] + 1e-9


def test_isotonic_outputs_in_unit_interval():
    raw, out = _make_overconfident_data()
    cal = IsotonicCalibrator()
    cal.fit(raw, out)
    for x in [0.0, 0.1, 0.5, 0.9, 1.0]:
        y = cal(x)
        assert 0.0 <= y <= 1.0


def test_temperature_calibrator_runs_and_clips():
    raw, out = _make_overconfident_data()
    cal = TemperatureCalibrator()
    cal.fit(raw, out)
    for x in [0.0, 0.5, 1.0]:
        y = cal(x)
        assert 0.0 <= y <= 1.0
