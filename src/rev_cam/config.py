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

DEFAULT_BATTERY_CAPACITY_MAH = 1000
from .distance import (
    DEFAULT_DISTANCE_CALIBRATION,
    DEFAULT_DISTANCE_ZONES,
    DistanceCalibration,
    DistanceZones,
)


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


@dataclass(frozen=True, slots=True)
class StreamSettings:
    """Configuration values for the MJPEG stream."""

    fps: int = 20
    jpeg_quality: int = 85

    def __post_init__(self) -> None:
        if self.fps < 1 or self.fps > 60:
            raise ValueError("Stream fps must be between 1 and 60")
        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError("Stream JPEG quality must be between 1 and 100")

    def to_dict(self) -> Dict[str, int]:
        return {"fps": int(self.fps), "jpeg_quality": int(self.jpeg_quality)}


RESOLUTION_PRESETS: Mapping[str, Resolution] = {
    "640x480": Resolution(640, 480),
    "800x600": Resolution(800, 600),
    "960x540": Resolution(960, 540),
    "1280x720": Resolution(1280, 720),
    "1920x1080": Resolution(1920, 1080),
}

DEFAULT_RESOLUTION_KEY = "1280x720"
DEFAULT_RESOLUTION = RESOLUTION_PRESETS[DEFAULT_RESOLUTION_KEY]
DEFAULT_STREAM_SETTINGS = StreamSettings()


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


def _parse_stream_settings(value: Any, *, default: StreamSettings) -> StreamSettings:
    if value is None:
        return default
    if isinstance(value, StreamSettings):
        fps_raw: Any = value.fps
        quality_raw: Any = value.jpeg_quality
    elif isinstance(value, Mapping):
        fps_raw = value.get("fps", default.fps)
        quality_raw = value.get("jpeg_quality", value.get("quality", default.jpeg_quality))
    else:
        raise ValueError("Stream settings must be provided as a mapping")

    try:
        fps = int(float(fps_raw))
    except (TypeError, ValueError) as exc:
        raise ValueError("Stream fps must be an integer") from exc
    if fps < 1 or fps > 60:
        raise ValueError("Stream fps must be between 1 and 60")

    try:
        quality = int(float(quality_raw))
    except (TypeError, ValueError) as exc:
        raise ValueError("Stream JPEG quality must be an integer") from exc
    if quality < 1 or quality > 100:
        raise ValueError("Stream JPEG quality must be between 1 and 100")

    return StreamSettings(fps=fps, jpeg_quality=quality)


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


def _parse_distance_calibration(
    value: Any, *, default: DistanceCalibration
) -> DistanceCalibration:
    if value is None:
        return default
    if isinstance(value, DistanceCalibration):
        return DistanceCalibration(value.offset_m, value.scale)
    if isinstance(value, Mapping):
        payload = value.get("calibration") if "calibration" in value else value
        if not isinstance(payload, Mapping):
            raise ValueError("Distance calibration must provide 'offset_m' and 'scale'")
        offset_raw = payload.get("offset_m", payload.get("offset", default.offset_m))
        scale_raw = payload.get("scale", payload.get("scale_factor", default.scale))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) != 2:
            raise ValueError("Distance calibration sequence must contain two values")
        offset_raw, scale_raw = items
    else:
        raise ValueError("Unsupported distance calibration value")
    try:
        offset = float(offset_raw)
        scale = float(scale_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Distance calibration values must be numeric") from exc
    return DistanceCalibration(offset, scale)


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


def _parse_battery_capacity(value: Any, *, default: int) -> int:
    if value is None:
        return default
    candidate: Any
    if isinstance(value, Mapping):
        if "capacity_mah" in value:
            candidate = value.get("capacity_mah")
        elif "battery" in value and isinstance(value["battery"], Mapping):
            # Allow callers to pass the entire configuration payload.
            return _parse_battery_capacity(value["battery"], default=default)
        else:
            return default
    else:
        candidate = value
    try:
        capacity = int(float(candidate))
    except (TypeError, ValueError) as exc:
        raise ValueError("Battery capacity must be numeric") from exc
    if capacity <= 0:
        raise ValueError("Battery capacity must be positive")
    if not (50 <= capacity <= 200_000):
        raise ValueError("Battery capacity must be between 50 and 200000 mAh")
    return capacity


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
            self._distance_calibration,
            self._battery_limits,
            self._battery_capacity,
            self._stream_settings,
        ) = self._load()

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(
        self,
    ) -> tuple[
        Orientation,
        str,
        Resolution,
        DistanceZones,
        DistanceCalibration,
        BatteryLimits,
        int,
        StreamSettings,
    ]:
        if not self._path.exists():
            return (
                Orientation(),
                DEFAULT_CAMERA_CHOICE,
                DEFAULT_RESOLUTION,
                DEFAULT_DISTANCE_ZONES,
                DEFAULT_DISTANCE_CALIBRATION,
                DEFAULT_BATTERY_LIMITS,
                DEFAULT_BATTERY_CAPACITY_MAH,
                DEFAULT_STREAM_SETTINGS,
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
            distance_calibration = _parse_distance_calibration(
                distance_payload, default=DEFAULT_DISTANCE_CALIBRATION
            )

            battery_payload = payload.get("battery")
            battery_limits = _parse_battery_limits(
                battery_payload, default=DEFAULT_BATTERY_LIMITS
            )
            battery_capacity = _parse_battery_capacity(
                battery_payload, default=DEFAULT_BATTERY_CAPACITY_MAH
            )

            stream_payload = payload.get("stream")
            stream_settings = _parse_stream_settings(
                stream_payload, default=DEFAULT_STREAM_SETTINGS
            )

            return (
                orientation,
                camera,
                resolution,
                distance_zones,
                distance_calibration,
                battery_limits,
                battery_capacity,
                stream_settings,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load configuration: {exc}") from exc

    def _save(self) -> None:
        payload: Dict[str, Any] = {
            "orientation": asdict(self._orientation),
            "camera": self._camera,
            "resolution": {"width": self._resolution.width, "height": self._resolution.height},
            "distance": {
                "zones": self._distance_zones.to_dict(),
                "calibration": self._distance_calibration.to_dict(),
            },
            "battery": {
                "limits": self._battery_limits.to_dict(),
                "capacity_mah": self._battery_capacity,
            },
            "stream": self._stream_settings.to_dict(),
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

    def get_distance_calibration(self) -> DistanceCalibration:
        with self._lock:
            return self._distance_calibration

    def set_distance_calibration(
        self, data: Mapping[str, Any] | Sequence[object] | DistanceCalibration
    ) -> DistanceCalibration:
        calibration = _parse_distance_calibration(data, default=self._distance_calibration)
        with self._lock:
            self._distance_calibration = calibration
            self._save()
        return calibration

    def get_battery_limits(self) -> BatteryLimits:
        with self._lock:
            return self._battery_limits

    def set_battery_limits(self, data: Mapping[str, Any] | Sequence[object]) -> BatteryLimits:
        limits = _parse_battery_limits(data, default=self._battery_limits)
        with self._lock:
            self._battery_limits = limits
            self._save()
        return limits

    def get_battery_capacity(self) -> int:
        with self._lock:
            return self._battery_capacity

    def set_battery_capacity(self, value: Any) -> int:
        capacity = _parse_battery_capacity(value, default=self._battery_capacity)
        with self._lock:
            self._battery_capacity = capacity
            self._save()
        return capacity

    def get_stream_settings(self) -> StreamSettings:
        with self._lock:
            return self._stream_settings

    def set_stream_settings(self, data: Mapping[str, Any] | StreamSettings) -> StreamSettings:
        settings = _parse_stream_settings(data, default=self._stream_settings)
        with self._lock:
            self._stream_settings = settings
            self._save()
        return settings


__all__ = [
    "ConfigManager",
    "Orientation",
    "Resolution",
    "StreamSettings",
    "RESOLUTION_PRESETS",
    "DEFAULT_RESOLUTION",
    "DEFAULT_RESOLUTION_KEY",
]
