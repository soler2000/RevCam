from pathlib import Path

import pytest

from rev_cam.camera import DEFAULT_CAMERA_CHOICE
from rev_cam.config import ConfigManager, Orientation


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
