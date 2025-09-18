import json
from pathlib import Path

from rev_cam.config import ConfigManager, Orientation


def test_config_manager_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    manager = ConfigManager(config_path)
    assert manager.get_orientation() == Orientation()
    assert manager.get_camera_mode() == "auto"


def test_config_manager_persists_camera_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    manager = ConfigManager(config_path)

    mode = manager.set_camera_mode("Picamera2")
    assert mode == "picamera2"
    assert manager.get_camera_mode() == "picamera2"

    payload = json.loads(config_path.read_text())
    assert payload["camera"]["mode"] == "picamera2"

    manager.set_orientation({"rotation": 90, "flip_horizontal": True})
    stored = json.loads(config_path.read_text())
    assert stored["orientation"]["rotation"] == 90
    assert stored["orientation"]["flip_horizontal"] is True


def test_config_manager_loads_legacy_format(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"rotation": 180, "flip_vertical": True}))

    manager = ConfigManager(config_path)
    orientation = manager.get_orientation()

    assert orientation.rotation == 180
    assert orientation.flip_vertical is True
    assert manager.get_camera_mode() == "auto"
