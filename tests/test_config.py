from pathlib import Path

from pathlib import Path

import pytest

from rev_cam.battery import BatteryLimits
from rev_cam.config import (
    ConfigManager,
    DistanceZones,
    Orientation,
    StreamSettings,
    DEFAULT_BATTERY_CAPACITY_MAH,
    DEFAULT_CAMERA_CHOICE,
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
    assert settings.jpeg_quality == 85


def test_stream_settings_persistence(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    updated = manager.set_stream_settings({"fps": 18, "jpeg_quality": 70})
    assert isinstance(updated, StreamSettings)
    assert updated.fps == 18
    assert updated.jpeg_quality == 70
    reloaded = ConfigManager(config_file)
    assert reloaded.get_stream_settings() == updated


def test_stream_settings_validation(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    with pytest.raises(ValueError):
        manager.set_stream_settings({"fps": 0})
    with pytest.raises(ValueError):
        manager.set_stream_settings({"jpeg_quality": 120})
