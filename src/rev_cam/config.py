"""Configuration management for RevCam."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping


@dataclass(frozen=True, slots=True)
class Orientation:
    """Represents the rotation and flips applied to a frame."""

    rotation: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def normalise(self) -> "Orientation":
        """Return a normalised copy with the rotation wrapped to 0-270."""

        rotation = (self.rotation // 90) % 4 * 90
        return Orientation(rotation, self.flip_horizontal, self.flip_vertical)


@dataclass(frozen=True, slots=True)
class Settings:
    """Complete persistent configuration for the application."""

    orientation: Orientation = Orientation()
    camera_mode: str = "auto"


def _parse_orientation(data: Mapping[str, Any]) -> Orientation:
    try:
        rotation_raw = data.get("rotation", 0)
        if not isinstance(rotation_raw, int):
            raise ValueError("Rotation must be an integer")
        if rotation_raw % 90 != 0:
            raise ValueError("Rotation must be a multiple of 90 degrees")
        rotation = (rotation_raw // 90) % 4 * 90
        flip_horizontal = bool(data.get("flip_horizontal", False))
        flip_vertical = bool(data.get("flip_vertical", False))
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    return Orientation(rotation=rotation, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)


def _normalise_camera_mode(raw: Any) -> str:
    if isinstance(raw, str):
        cleaned = raw.strip().lower()
        return cleaned or "auto"
    return "auto"


def _parse_settings(data: Mapping[str, Any]) -> Settings:
    orientation_source: Mapping[str, Any]
    maybe_orientation = data.get("orientation")
    if isinstance(maybe_orientation, Mapping):
        orientation_source = maybe_orientation
    else:
        orientation_source = data

    orientation = _parse_orientation(orientation_source)

    camera_mode = "auto"
    camera_section = data.get("camera")
    if isinstance(camera_section, Mapping):
        camera_mode = _normalise_camera_mode(camera_section.get("mode"))
    else:
        if isinstance(camera_section, str):
            camera_mode = _normalise_camera_mode(camera_section)
        else:
            camera_mode = _normalise_camera_mode(data.get("camera_mode"))

    return Settings(orientation=orientation, camera_mode=camera_mode)


class ConfigManager:
    """Stores configuration state on disk with thread-safety."""

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._lock = Lock()
        self._ensure_parent()
        self._data: Settings = self._load()

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Settings:
        if not self._path.exists():
            return Settings()
        try:
            payload = json.loads(self._path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("Configuration file must contain a JSON object")
            return _parse_settings(payload)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load configuration: {exc}") from exc

    def _save(self, settings: Settings) -> None:
        payload: Dict[str, Any] = {
            "orientation": asdict(settings.orientation.normalise()),
            "camera": {"mode": settings.camera_mode},
        }
        self._path.write_text(json.dumps(payload, indent=2))

    def get_orientation(self) -> Orientation:
        with self._lock:
            return self._data.orientation

    def set_orientation(self, data: Mapping[str, Any]) -> Orientation:
        orientation = _parse_orientation(data).normalise()
        with self._lock:
            self._data = Settings(orientation=orientation, camera_mode=self._data.camera_mode)
            self._save(self._data)
        return orientation

    def get_camera_mode(self) -> str:
        with self._lock:
            return self._data.camera_mode

    def set_camera_mode(self, mode: str) -> str:
        camera_mode = _normalise_camera_mode(mode)
        with self._lock:
            self._data = Settings(orientation=self._data.orientation, camera_mode=camera_mode)
            self._save(self._data)
        return camera_mode


__all__ = ["ConfigManager", "Orientation", "Settings"]
