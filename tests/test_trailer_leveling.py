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
    orientation = OrientationAngles(roll=4.0, pitch=0.0)
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
    orientation = OrientationAngles(roll=-10.0, pitch=0.0)
    result = compute_hitched_leveling(orientation, settings)
    assert result["side_to_raise"] == "right"
    assert result["within_ramp_limits"] is False
    assert result["ramp"]["limited"] is True


def test_unhitched_leveling_includes_hitch_guidance():
    settings = LevelingSettings(
        geometry=TrailerGeometry(axle_width_m=2.5, hitch_to_axle_m=5.0, length_m=8.0),
        ramp=RampSpecification(length_m=0.7, height_m=0.1),
    )
    orientation = OrientationAngles(roll=0.0, pitch=3.0)
    hitched = compute_hitched_leveling(orientation, settings)
    unhitched = compute_unhitched_leveling(orientation, settings)
    expected_adjustment = math.tan(math.radians(3.0)) * 5.0
    assert unhitched["hitch_direction"] == "lower"
    assert unhitched["hitch_adjustment_m"] == pytest.approx(expected_adjustment)
    assert unhitched["side_to_raise"] == hitched["side_to_raise"]
    assert unhitched["required_raise_m"] == pytest.approx(hitched["required_raise_m"])
    assert unhitched["guidance_message"] == hitched["message"]
    assert unhitched["hitch_message"] == "Lower the hitch by 26.2 cm."
    assert unhitched["hitch_notice"] == "Hitch guidance assumes the trailer stays hitched."
    assert unhitched["message"].startswith(hitched["message"])  # combined guidance
    assert "Lower the hitch" in unhitched["message"]


def test_unhitched_leveling_reports_level_when_pitch_small():
    settings = LevelingSettings(
        geometry=TrailerGeometry(axle_width_m=2.4, hitch_to_axle_m=4.2, length_m=7.0),
        ramp=RampSpecification(length_m=0.8, height_m=0.12),
    )
    orientation = OrientationAngles(roll=1.0, pitch=0.05)
    unhitched = compute_unhitched_leveling(orientation, settings)
    assert unhitched["hitch_direction"] == "level"
    assert unhitched["hitch_adjustment_m"] == pytest.approx(0.0)
    assert unhitched["hitch_message"] == "Hitch is level relative to the axle."
    assert unhitched["hitch_notice"] == "Hitch guidance assumes the trailer stays hitched."


def test_evaluate_leveling_returns_both_modes():
    settings = LevelingSettings()
    orientation = OrientationAngles(roll=0.0, pitch=195.0)
    result = evaluate_leveling(orientation, settings)
    assert "hitched" in result
    assert "unhitched" in result
    assert set(result["orientation"].keys()) == {"roll", "pitch"}
    assert set(result["raw_orientation"].keys()) == {"roll", "pitch"}
    assert result["orientation"]["pitch"] == pytest.approx(-165.0)
    assert result["raw_orientation"]["pitch"] == pytest.approx(-165.0)


def test_evaluate_leveling_applies_reference():
    reference = OrientationAngles(roll=1.5, pitch=-0.5)
    settings = LevelingSettings(reference=reference)
    orientation = OrientationAngles(roll=4.5, pitch=1.0)
    result = evaluate_leveling(orientation, settings)
    assert result["orientation"]["roll"] == pytest.approx(3.0)
    assert result["orientation"]["pitch"] == pytest.approx(2.0)
    assert result["raw_orientation"]["roll"] == pytest.approx(4.0)
    assert result["raw_orientation"]["pitch"] == pytest.approx(1.0)
