"""Configuration management for RevCam."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping

from .camera import CAMERA_SOURCES, DEFAULT_CAMERA_CHOICE


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
        self._orientation, self._camera = self._load()

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> tuple[Orientation, str]:
        if not self._path.exists():
            return Orientation(), DEFAULT_CAMERA_CHOICE
        try:
            payload = json.loads(self._path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("Configuration file must contain a JSON object")
            if "orientation" in payload and isinstance(payload["orientation"], Mapping):
                orientation_payload = payload["orientation"]
            else:
                orientation_payload = payload
            orientation = _parse_orientation(orientation_payload)
            camera_raw = payload.get("camera", DEFAULT_CAMERA_CHOICE)
            if isinstance(camera_raw, str) and camera_raw.strip():
                camera = camera_raw.strip().lower()
                if camera not in CAMERA_SOURCES:
                    camera = DEFAULT_CAMERA_CHOICE
            else:
                camera = DEFAULT_CAMERA_CHOICE
            return orientation, camera
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load configuration: {exc}") from exc

    def _save(self) -> None:
        payload: Dict[str, Any] = {
            "orientation": asdict(self._orientation),
            "camera": self._camera,
        }
        self._path.write_text(json.dumps(payload, indent=2))

    def get_orientation(self) -> Orientation:
        with self._lock:
            return self._orientation

    def set_orientation(self, data: Mapping[str, Any]) -> Orientation:
        orientation = _parse_orientation(data)
        with self._lock:
            self._orientation = orientation
            self._save()
        return orientation

    def get_camera(self) -> str:
        with self._lock:
            return self._camera

    def set_camera(self, camera: str) -> str:
        if not isinstance(camera, str) or not camera.strip():
            raise ValueError("Camera selection must be a non-empty string")
        normalised = camera.strip().lower()
        if normalised not in CAMERA_SOURCES:
            raise ValueError(f"Unknown camera selection: {camera}")
        with self._lock:
            self._camera = normalised
            self._save()
        return normalised


__all__ = ["ConfigManager", "Orientation"]
