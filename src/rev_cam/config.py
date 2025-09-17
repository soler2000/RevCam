"""Configuration management for RevCam."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
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


class ConfigManager:
    """Stores configuration state on disk with thread-safety."""

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._lock = Lock()
        self._ensure_parent()
        self._data: Orientation = self._load()

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Orientation:
        if not self._path.exists():
            return Orientation()
        try:
            payload = json.loads(self._path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("Configuration file must contain a JSON object")
            return _parse_orientation(payload)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load configuration: {exc}") from exc

    def _save(self, orientation: Orientation) -> None:
        payload: Dict[str, Any] = asdict(orientation)
        self._path.write_text(json.dumps(payload, indent=2))

    def get_orientation(self) -> Orientation:
        with self._lock:
            return self._data

    def set_orientation(self, data: Mapping[str, Any]) -> Orientation:
        orientation = _parse_orientation(data)
        with self._lock:
            self._data = orientation
            self._save(orientation)
        return orientation


__all__ = ["ConfigManager", "Orientation"]
