"""Configuration management for RevCam."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Mapping, Sequence

try:
    from .camera import CAMERA_SOURCES, DEFAULT_CAMERA_CHOICE
except ImportError:  # pragma: no cover - fallback when optional dependencies missing
    CAMERA_SOURCES = {
        "auto": "Automatic (PiCamera2 with synthetic fallback)",
        "picamera": "PiCamera2",
        "opencv": "OpenCV (USB webcam)",
        "synthetic": "Synthetic test pattern",
    }
    DEFAULT_CAMERA_CHOICE = "auto"
from .battery import BatteryLimits, DEFAULT_BATTERY_LIMITS
from .distance import DEFAULT_DISTANCE_ZONES, DistanceZones


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
class Resolution:
    """Represents the desired capture resolution for the camera."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Resolution dimensions must be positive integers")

    def as_tuple(self) -> tuple[int, int]:
        return (int(self.width), int(self.height))

    def key(self) -> str:
        return f"{self.width}x{self.height}"


RESOLUTION_PRESETS: Mapping[str, Resolution] = {
    "640x480": Resolution(640, 480),
    "800x600": Resolution(800, 600),
    "960x540": Resolution(960, 540),
    "1280x720": Resolution(1280, 720),
    "1920x1080": Resolution(1920, 1080),
}

DEFAULT_RESOLUTION_KEY = "1280x720"
DEFAULT_RESOLUTION = RESOLUTION_PRESETS[DEFAULT_RESOLUTION_KEY]


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


def _parse_resolution(value: Any, *, default: Resolution) -> Resolution:
    if value is None:
        return default
    if isinstance(value, Resolution):
        return Resolution(value.width, value.height)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        normalised = text.replace("×", "x")
        preset = RESOLUTION_PRESETS.get(normalised)
        if preset:
            return preset
        if "x" in normalised:
            parts = normalised.split("x", 1)
            if len(parts) == 2:
                try:
                    width = int(parts[0].strip())
                    height = int(parts[1].strip())
                    return Resolution(width, height)
                except ValueError as exc:
                    raise ValueError("Resolution values must be integers") from exc
        raise ValueError(f"Unknown resolution preset: {value}")
    if isinstance(value, Mapping):
        if "preset" in value:
            return _parse_resolution(value.get("preset"), default=default)
        width_raw = value.get("width")
        height_raw = value.get("height")
        if width_raw is None or height_raw is None:
            raise ValueError("Resolution mapping must include 'width' and 'height'")
        try:
            width = int(width_raw)
            height = int(height_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Resolution width and height must be integers") from exc
        return Resolution(width, height)
    if isinstance(value, (Sequence, Iterable)):
        iterator = iter(value)
        try:
            width_raw = next(iterator)
            height_raw = next(iterator)
        except StopIteration as exc:
            raise ValueError("Resolution sequence must contain width and height") from exc
        try:
            width = int(width_raw)
            height = int(height_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Resolution width and height must be integers") from exc
        return Resolution(width, height)
    raise ValueError("Unsupported resolution value")


def _parse_distance_zones(value: Any, *, default: DistanceZones) -> DistanceZones:
    if value is None:
        return default
    if isinstance(value, DistanceZones):
        return DistanceZones(value.caution, value.warning, value.danger)
    if isinstance(value, Mapping):
        payload = value.get("zones") if "zones" in value else value
        if not isinstance(payload, Mapping):
            raise ValueError("Distance zones must provide 'caution', 'warning' and 'danger'")
        caution = payload.get("caution", default.caution)
        warning = payload.get("warning", default.warning)
        danger = payload.get("danger", default.danger)
    elif isinstance(value, Sequence):
        items = list(value)
        if len(items) != 3:
            raise ValueError("Distance zones sequence must contain three values")
        caution, warning, danger = items
    else:
        raise ValueError("Unsupported distance zone value")
    try:
        return DistanceZones(float(caution), float(warning), float(danger))
    except (TypeError, ValueError) as exc:
        raise ValueError("Distance zone values must be numeric") from exc


def _parse_battery_limits(value: Any, *, default: BatteryLimits) -> BatteryLimits:
    if value is None:
        return default
    if isinstance(value, BatteryLimits):
        return BatteryLimits(value.warning_percent, value.shutdown_percent)
    if isinstance(value, Mapping):
        payload = value.get("limits") if "limits" in value else value
        if not isinstance(payload, Mapping):
            raise ValueError("Battery limits must provide 'warning' and 'shutdown' values")
        warning = payload.get("warning_percent", payload.get("warning", default.warning_percent))
        shutdown = payload.get(
            "shutdown_percent", payload.get("shutdown", default.shutdown_percent)
        )
    elif isinstance(value, Sequence):
        items = list(value)
        if len(items) != 2:
            raise ValueError("Battery limits sequence must contain two values")
        warning, shutdown = items
    else:
        raise ValueError("Unsupported battery limits value")
    try:
        return BatteryLimits(float(warning), float(shutdown))
    except (TypeError, ValueError) as exc:
        raise ValueError("Battery limit values must be numeric") from exc


class ConfigManager:
    """Stores configuration state on disk with thread-safety."""

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._lock = Lock()
        self._ensure_parent()
        (
            self._orientation,
            self._camera,
            self._resolution,
            self._distance_zones,
            self._battery_limits,
        ) = self._load()

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> tuple[Orientation, str, Resolution, DistanceZones, BatteryLimits]:
        if not self._path.exists():
            return (
                Orientation(),
                DEFAULT_CAMERA_CHOICE,
                DEFAULT_RESOLUTION,
                DEFAULT_DISTANCE_ZONES,
                DEFAULT_BATTERY_LIMITS,
            )
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
            resolution_payload = payload.get("resolution")
            resolution = _parse_resolution(resolution_payload, default=DEFAULT_RESOLUTION)

            distance_payload = payload.get("distance")
            distance_zones = _parse_distance_zones(distance_payload, default=DEFAULT_DISTANCE_ZONES)

            battery_payload = payload.get("battery")
            battery_limits = _parse_battery_limits(
                battery_payload, default=DEFAULT_BATTERY_LIMITS
            )

            return orientation, camera, resolution, distance_zones, battery_limits
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load configuration: {exc}") from exc

    def _save(self) -> None:
        payload: Dict[str, Any] = {
            "orientation": asdict(self._orientation),
            "camera": self._camera,
            "resolution": {"width": self._resolution.width, "height": self._resolution.height},
            "distance": {"zones": self._distance_zones.to_dict()},
            "battery": {"limits": self._battery_limits.to_dict()},
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

    def get_resolution(self) -> Resolution:
        with self._lock:
            return self._resolution

    def parse_resolution(self, value: Any) -> Resolution:
        return _parse_resolution(value, default=self._resolution)

    def set_resolution(self, value: Any) -> Resolution:
        resolution = _parse_resolution(value, default=self._resolution)
        with self._lock:
            self._resolution = resolution
            self._save()
        return resolution

    def get_distance_zones(self) -> DistanceZones:
        with self._lock:
            return self._distance_zones

    def set_distance_zones(self, data: Mapping[str, Any] | Sequence[object]) -> DistanceZones:
        zones = _parse_distance_zones(data, default=self._distance_zones)
        with self._lock:
            self._distance_zones = zones
            self._save()
        return zones

    def get_battery_limits(self) -> BatteryLimits:
        with self._lock:
            return self._battery_limits

    def set_battery_limits(self, data: Mapping[str, Any] | Sequence[object]) -> BatteryLimits:
        limits = _parse_battery_limits(data, default=self._battery_limits)
        with self._lock:
            self._battery_limits = limits
            self._save()
        return limits


__all__ = [
    "ConfigManager",
    "Orientation",
    "Resolution",
    "RESOLUTION_PRESETS",
    "DEFAULT_RESOLUTION",
    "DEFAULT_RESOLUTION_KEY",
]
