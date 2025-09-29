import math

import pytest

from rev_cam.trailer_leveling import (
    LevelingSettings,
    OrientationAngles,
    RampSpecification,
    TrailerGeometry,
    compute_hitched_leveling,
    compute_unhitched_leveling,
    evaluate_leveling,
)


def test_hitched_leveling_with_small_roll():
    settings = LevelingSettings(
        geometry=TrailerGeometry(axle_width_m=2.4, hitch_to_axle_m=4.5, length_m=7.2),
        ramp=RampSpecification(length_m=0.8, height_m=0.12),
    )
    orientation = OrientationAngles(roll=4.0, pitch=0.0, yaw=0.0)
    result = compute_hitched_leveling(orientation, settings)
    assert result["side_to_raise"] == "left"
    assert pytest.approx(result["required_raise_m"], rel=1e-3) == math.sin(math.radians(4.0)) * 2.4 / 2
    assert result["within_ramp_limits"] is True
    assert result["ramp"]["limited"] is False


def test_hitched_leveling_respects_ramp_limits():
    settings = LevelingSettings(
        geometry=TrailerGeometry(axle_width_m=2.4, hitch_to_axle_m=4.5, length_m=7.2),
        ramp=RampSpecification(length_m=0.6, height_m=0.08),
    )
    orientation = OrientationAngles(roll=-10.0, pitch=0.0, yaw=0.0)
    result = compute_hitched_leveling(orientation, settings)
    assert result["side_to_raise"] == "right"
    assert result["within_ramp_limits"] is False
    assert result["ramp"]["limited"] is True


def test_unhitched_leveling_pitch_direction():
    settings = LevelingSettings(
        geometry=TrailerGeometry(axle_width_m=2.5, hitch_to_axle_m=5.0, length_m=8.0),
        ramp=RampSpecification(length_m=0.7, height_m=0.1),
    )
    orientation = OrientationAngles(roll=0.0, pitch=3.0, yaw=0.0)
    result = compute_unhitched_leveling(orientation, settings)
    assert result["hitch_direction"] == "lower"
    assert pytest.approx(result["hitch_adjustment_m"], rel=1e-3) == math.tan(math.radians(3.0)) * 5.0


def test_evaluate_leveling_returns_both_modes():
    settings = LevelingSettings()
    orientation = OrientationAngles(roll=0.0, pitch=0.0, yaw=195.0)
    result = evaluate_leveling(orientation, settings)
    assert "hitched" in result
    assert "unhitched" in result
    assert result["orientation"]["yaw"] == pytest.approx(-165.0)
