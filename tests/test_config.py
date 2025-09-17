from pathlib import Path

import pytest

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
