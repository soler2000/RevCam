from pathlib import Path
import math

import pytest

from rev_cam.battery import BatteryLimits
from rev_cam.config import (
    ConfigManager,
    DistanceZones,
    Orientation,
    ReversingAidPoint,
    ReversingAidsConfig,
    StreamSettings,
    generate_reversing_segments,
    DEFAULT_BATTERY_CAPACITY_MAH,
    DEFAULT_CAMERA_CHOICE,
    DEFAULT_REVERSING_AIDS,
)
from rev_cam.distance import DistanceCalibration


def test_default_orientation(tmp_path: Path):
    manager = ConfigManager(tmp_path / "config.json")
    assert manager.get_orientation() == Orientation()


def test_set_orientation_persists(tmp_path: Path):
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    updated = manager.set_orientation({"rotation": 90, "flip_horizontal": True, "flip_vertical": False})
    assert updated.rotation == 90
    # Reload to ensure persistence
    reloaded = ConfigManager(config_file)
    assert reloaded.get_orientation() == updated


def test_invalid_orientation_rejected(tmp_path: Path):
    manager = ConfigManager(tmp_path / "config.json")
    with pytest.raises(ValueError):
        manager.set_orientation({"rotation": 45})


def test_default_camera(tmp_path: Path):
    manager = ConfigManager(tmp_path / "config.json")
    assert manager.get_camera() == DEFAULT_CAMERA_CHOICE


def test_camera_persistence(tmp_path: Path):
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    manager.set_camera("synthetic")
    reloaded = ConfigManager(config_file)
    assert reloaded.get_camera() == "synthetic"


def test_camera_requires_non_empty_string(tmp_path: Path):
    manager = ConfigManager(tmp_path / "config.json")
    with pytest.raises(ValueError):
        manager.set_camera("")
    with pytest.raises(ValueError):
        manager.set_camera("unknown")


def test_default_distance_zones(tmp_path: Path):
    manager = ConfigManager(tmp_path / "config.json")
    zones = manager.get_distance_zones()
    assert isinstance(zones, DistanceZones)
    assert zones.caution >= zones.warning >= zones.danger > 0


def test_distance_zones_persistence(tmp_path: Path):
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    updated = manager.set_distance_zones({"caution": 3.0, "warning": 1.8, "danger": 0.6})
    assert isinstance(updated, DistanceZones)
    reloaded = ConfigManager(config_file)
    assert reloaded.get_distance_zones() == updated


def test_default_distance_calibration(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    calibration = manager.get_distance_calibration()
    assert isinstance(calibration, DistanceCalibration)
    assert calibration.offset_m == pytest.approx(0.0)
    assert calibration.scale == pytest.approx(1.0)


def test_distance_calibration_persistence(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    updated = manager.set_distance_calibration(DistanceCalibration(offset_m=0.3, scale=1.1))
    assert isinstance(updated, DistanceCalibration)
    reloaded = ConfigManager(config_file)
    assert reloaded.get_distance_calibration() == updated


def test_distance_calibration_validation(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    with pytest.raises(ValueError):
        manager.set_distance_calibration({"offset_m": "invalid", "scale": 1.0})


def test_default_battery_limits(tmp_path: Path):
    manager = ConfigManager(tmp_path / "config.json")
    limits = manager.get_battery_limits()
    assert isinstance(limits, BatteryLimits)
    assert limits.warning_percent >= limits.shutdown_percent


def test_battery_limits_persistence(tmp_path: Path):
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    updated = manager.set_battery_limits({"warning_percent": 32.0, "shutdown_percent": 9.0})
    assert isinstance(updated, BatteryLimits)
    reloaded = ConfigManager(config_file)
    assert reloaded.get_battery_limits() == updated


def test_default_battery_capacity(tmp_path: Path):
    manager = ConfigManager(tmp_path / "config.json")
    assert manager.get_battery_capacity() == DEFAULT_BATTERY_CAPACITY_MAH


def test_battery_capacity_persistence(tmp_path: Path):
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    updated = manager.set_battery_capacity(2400)
    assert updated == 2400
    reloaded = ConfigManager(config_file)
    assert reloaded.get_battery_capacity() == 2400


def test_default_stream_settings(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    settings = manager.get_stream_settings()
    assert isinstance(settings, StreamSettings)
    assert settings.fps == 20
    assert settings.bitrate == 1_500_000


def test_stream_settings_persistence(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    updated = manager.set_stream_settings({"fps": 18, "bitrate": 2_000_000})
    assert isinstance(updated, StreamSettings)
    assert updated.fps == 18
    assert updated.bitrate == 2_000_000
    reloaded = ConfigManager(config_file)
    assert reloaded.get_stream_settings() == updated


def test_stream_settings_validation(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    with pytest.raises(ValueError):
        manager.set_stream_settings({"fps": 0})
    with pytest.raises(ValueError):
        manager.set_stream_settings({"bitrate": 10_000})


def test_default_reversing_aids(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    aids = manager.get_reversing_aids()
    assert isinstance(aids, ReversingAidsConfig)
    assert aids == DEFAULT_REVERSING_AIDS


def test_reversing_aids_persistence(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    left_line = (
        ReversingAidPoint(0.18, 0.9),
        ReversingAidPoint(0.58, 0.22),
    )
    right_line = (
        ReversingAidPoint(0.82, 0.9),
        ReversingAidPoint(0.42, 0.22),
    )
    payload = {
        "enabled": False,
        "left": [segment.to_dict() for segment in generate_reversing_segments(*left_line)],
        "right": [segment.to_dict() for segment in generate_reversing_segments(*right_line)],
    }
    updated = manager.set_reversing_aids(payload)
    assert isinstance(updated, ReversingAidsConfig)
    reloaded = ConfigManager(config_file)
    assert reloaded.get_reversing_aids() == updated


def test_generate_reversing_segments_align() -> None:
    near = ReversingAidPoint(0.24, 0.88)
    far = ReversingAidPoint(0.6, 0.28)
    segments = generate_reversing_segments(near, far)

    assert len(segments) == 3
    assert segments[0].start == near
    assert segments[-1].end == far

    expected_dx = far.x - near.x
    expected_dy = far.y - near.y
    for segment in segments:
        for point in (segment.start, segment.end):
            dx = point.x - near.x
            dy = point.y - near.y
            assert math.isclose(dx * expected_dy, dy * expected_dx, rel_tol=1e-9, abs_tol=1e-9)


def test_reversing_aids_validation(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    with pytest.raises(ValueError):
        manager.set_reversing_aids(
            {
                "left": [
                    {"start": {"x": -0.1, "y": 0.2}, "end": {"x": 0.2, "y": 0.35}},
                    {"start": {"x": 0.4, "y": 0.5}, "end": {"x": 0.25, "y": 0.65}},
                    {"start": {"x": 0.3, "y": 0.7}, "end": {"x": 0.2, "y": 0.85}},
                ],
                "right": [
                    {"start": {"x": 0.5, "y": 0.2}, "end": {"x": 0.7, "y": 0.35}},
                    {"start": {"x": 0.6, "y": 0.5}, "end": {"x": 0.8, "y": 0.65}},
                    {"start": {"x": 0.7, "y": 0.7}, "end": {"x": 0.9, "y": 0.85}},
                ],
            }
        )
