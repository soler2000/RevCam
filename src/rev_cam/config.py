"""Configuration management for RevCam."""
from __future__ import annotations

import json
import math
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

DEFAULT_DISTANCE_OVERLAY_ENABLED = True
DEFAULT_DISTANCE_USE_PROJECTED = False

DEFAULT_OVERLAY_MASTER_ENABLED = True
DEFAULT_BATTERY_OVERLAY_ENABLED = True
DEFAULT_REVERSING_OVERLAY_ENABLED = True


@dataclass(frozen=True, slots=True)
class DistanceMounting:
    """Represents the physical installation of the distance sensor."""

    mount_height_m: float = 1.5
    mount_angle_deg: float = 40.0

    def __post_init__(self) -> None:
        try:
            height = float(self.mount_height_m)
            angle = float(self.mount_angle_deg)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError("Distance mounting values must be numeric") from exc
        if not math.isfinite(height) or height <= 0:
            raise ValueError("Mount height must be a positive finite value")
        if not math.isfinite(angle) or angle < 0 or angle >= 90:
            raise ValueError("Mount angle must be between 0 and 90 degrees")
        object.__setattr__(self, "mount_height_m", height)
        object.__setattr__(self, "mount_angle_deg", angle)

    def to_dict(self) -> dict[str, float]:
        return {
            "mount_height_m": float(self.mount_height_m),
            "mount_angle_deg": float(self.mount_angle_deg),
        }


DEFAULT_DISTANCE_MOUNTING = DistanceMounting()


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


@dataclass(frozen=True, slots=True)
class ReversingAidPoint:
    """Represents a normalised coordinate used to draw reversing aids."""

    x: float
    y: float

    def __post_init__(self) -> None:
        try:
            x = float(self.x)
            y = float(self.y)
        except (TypeError, ValueError) as exc:
            raise ValueError("Reversing aid coordinates must be numeric") from exc
        if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
            raise ValueError("Reversing aid coordinates must be between 0 and 1")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y)}


@dataclass(frozen=True, slots=True)
class ReversingAidSegment:
    """Represents a single coloured guide segment."""

    start: ReversingAidPoint
    end: ReversingAidPoint

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}


REVERSING_SEGMENT_RATIOS: tuple[tuple[float, float], ...] = (
    (0.0, 0.25),
    (0.375, 0.625),
    (0.75, 1.0),
)


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def generate_reversing_segments(
    near: ReversingAidPoint, far: ReversingAidPoint
) -> tuple[ReversingAidSegment, ...]:
    """Return evenly spaced segments along the main reversing aid line."""

    dx = far.x - near.x
    dy = far.y - near.y
    segments: list[ReversingAidSegment] = []
    for start_ratio, end_ratio in REVERSING_SEGMENT_RATIOS:
        start = ReversingAidPoint(
            _clamp_unit(near.x + dx * start_ratio),
            _clamp_unit(near.y + dy * start_ratio),
        )
        end = ReversingAidPoint(
            _clamp_unit(near.x + dx * end_ratio),
            _clamp_unit(near.y + dy * end_ratio),
        )
        segments.append(ReversingAidSegment(start=start, end=end))
    return tuple(segments)


def _normalise_reversing_segments(
    segments: Sequence[ReversingAidSegment],
) -> tuple[ReversingAidSegment, ...]:
    if len(segments) != len(REVERSING_SEGMENT_RATIOS):
        return tuple(segments)
    near = segments[0].start
    far = segments[-1].end
    return generate_reversing_segments(
        ReversingAidPoint(near.x, near.y),
        ReversingAidPoint(far.x, far.y),
    )


@dataclass(frozen=True, slots=True)
class ReversingAidsConfig:
    """Configuration describing the reversing aid overlay."""

    enabled: bool = True
    left: tuple[ReversingAidSegment, ...] = generate_reversing_segments(
        ReversingAidPoint(0.14, 0.86),
        ReversingAidPoint(0.52, 0.18),
    )
    right: tuple[ReversingAidSegment, ...] = generate_reversing_segments(
        ReversingAidPoint(0.86, 0.86),
        ReversingAidPoint(0.48, 0.18),
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "left": [segment.to_dict() for segment in self.left],
            "right": [segment.to_dict() for segment in self.right],
        }


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
DEFAULT_REVERSING_AIDS = ReversingAidsConfig()


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


def _parse_overlay_flag(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            raise ValueError("Overlay flags must be boolean values")
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        if text in {"true", "1", "yes", "on", "enabled"}:
            return True
        if text in {"false", "0", "no", "off", "disabled"}:
            return False
    raise ValueError("Overlay flags must be boolean values")


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


def _parse_reversing_point(value: Any) -> ReversingAidPoint:
    if isinstance(value, ReversingAidPoint):
        return ReversingAidPoint(value.x, value.y)
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            raise ValueError("Reversing aid points must include 'x' and 'y'")
        candidate = {"x": value.get("x"), "y": value.get("y")}
    elif isinstance(value, Sequence):
        items = list(value)
        if len(items) != 2:
            raise ValueError("Reversing aid points must contain two values")
        candidate = {"x": items[0], "y": items[1]}
    else:
        raise ValueError("Unsupported reversing aid point value")
    return ReversingAidPoint(candidate["x"], candidate["y"])


def _parse_reversing_segment(value: Any) -> ReversingAidSegment:
    if isinstance(value, ReversingAidSegment):
        return ReversingAidSegment(
            start=ReversingAidPoint(value.start.x, value.start.y),
            end=ReversingAidPoint(value.end.x, value.end.y),
        )
    if isinstance(value, Mapping):
        if "start" in value or "end" in value:
            start_value = value.get("start")
            end_value = value.get("end")
            if start_value is None or end_value is None:
                raise ValueError("Reversing aid segments must include 'start' and 'end'")
        else:
            keys = {key.lower(): key for key in value.keys()}
            start_x = value.get(keys.get("x0")) or value.get(keys.get("start_x"))
            start_y = value.get(keys.get("y0")) or value.get(keys.get("start_y"))
            end_x = value.get(keys.get("x1")) or value.get(keys.get("end_x"))
            end_y = value.get(keys.get("y1")) or value.get(keys.get("end_y"))
            if None in (start_x, start_y, end_x, end_y):
                raise ValueError(
                    "Reversing aid segments must provide start and end coordinates"
                )
            start_value = {"x": start_x, "y": start_y}
            end_value = {"x": end_x, "y": end_y}
        start_point = _parse_reversing_point(start_value)
        end_point = _parse_reversing_point(end_value)
        return ReversingAidSegment(start=start_point, end=end_point)
    if isinstance(value, Sequence):
        items = list(value)
        if len(items) != 4:
            raise ValueError("Reversing aid segments must contain four values")
        start_point = _parse_reversing_point(items[:2])
        end_point = _parse_reversing_point(items[2:])
        return ReversingAidSegment(start=start_point, end=end_point)
    raise ValueError("Unsupported reversing aid segment value")


def _parse_reversing_segments(
    value: Any, *, default: tuple[ReversingAidSegment, ...]
) -> tuple[ReversingAidSegment, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, Mapping):
        payload = value.get("segments") if "segments" in value else value
        if isinstance(payload, Mapping):
            payload = list(payload.values())
        value = payload
    if isinstance(value, Sequence):
        segments = [_parse_reversing_segment(item) for item in value]
    else:
        raise ValueError("Reversing aid segments must be provided as a sequence")
    if len(segments) != len(default):
        raise ValueError(
            f"Reversing aids require exactly {len(default)} segments per side"
        )
    return _normalise_reversing_segments(segments)


def _parse_reversing_aids(
    value: Any, *, default: ReversingAidsConfig
) -> ReversingAidsConfig:
    if value is None:
        return default
    if isinstance(value, ReversingAidsConfig):
        return ReversingAidsConfig(
            enabled=value.enabled,
            left=tuple(_parse_reversing_segment(segment) for segment in value.left),
            right=tuple(_parse_reversing_segment(segment) for segment in value.right),
        )
    if isinstance(value, Mapping):
        payload = value.get("reversing_aids") if "reversing_aids" in value else value
        if not isinstance(payload, Mapping):
            raise ValueError("Reversing aids configuration must be a mapping")
        enabled = payload.get("enabled", default.enabled)
        left = _parse_reversing_segments(payload.get("left"), default=default.left)
        right = _parse_reversing_segments(payload.get("right"), default=default.right)
        return ReversingAidsConfig(enabled=bool(enabled), left=left, right=right)
    raise ValueError("Unsupported reversing aids configuration value")


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


def _parse_distance_mounting(
    value: Any, *, default: DistanceMounting
) -> DistanceMounting:
    if value is None:
        return default
    if isinstance(value, DistanceMounting):
        return DistanceMounting(value.mount_height_m, value.mount_angle_deg)
    if isinstance(value, Mapping):
        payload = value.get("geometry") if "geometry" in value else value
        if not isinstance(payload, Mapping):
            raise ValueError("Distance geometry must provide 'mount_height_m' and 'mount_angle_deg'")
        height_raw = payload.get("mount_height_m", payload.get("height", default.mount_height_m))
        angle_raw = payload.get("mount_angle_deg", payload.get("angle", default.mount_angle_deg))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) != 2:
            raise ValueError("Distance geometry sequence must contain two values")
        height_raw, angle_raw = items
    else:
        raise ValueError("Unsupported distance geometry value")
    try:
        height = float(height_raw)
        angle = float(angle_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Distance geometry values must be numeric") from exc
    return DistanceMounting(height, angle)


def _parse_distance_overlay_enabled(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    candidate: Any
    if isinstance(value, Mapping):
        if "overlay_enabled" in value:
            candidate = value.get("overlay_enabled", default)
        else:
            overlay_payload = value.get("overlay") if "overlay" in value else None
            if isinstance(overlay_payload, Mapping):
                candidate = overlay_payload.get("enabled", default)
            else:
                candidate = default
    else:
        candidate = value
    if isinstance(candidate, bool):
        return candidate
    if isinstance(candidate, (int, float)):
        return bool(candidate)
    if isinstance(candidate, str):
        normalised = candidate.strip().lower()
        if normalised in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if normalised in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
    raise ValueError("Distance overlay flag must be a boolean value")


def _parse_distance_use_projected(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    candidate: Any
    if isinstance(value, Mapping):
        if "use_projected_distance" in value:
            candidate = value.get("use_projected_distance", default)
        else:
            distance_payload = value.get("distance") if "distance" in value else None
            display_payload = value.get("display") if "display" in value else None
            if isinstance(display_payload, Mapping):
                candidate = display_payload.get("use_projected", default)
            elif isinstance(distance_payload, Mapping):
                candidate = distance_payload.get("use_projected_distance", default)
            else:
                candidate = default
    else:
        candidate = value
    if isinstance(candidate, bool):
        return candidate
    if isinstance(candidate, (int, float)):
        return bool(candidate)
    if isinstance(candidate, str):
        normalised = candidate.strip().lower()
        if normalised in {"1", "true", "yes", "on", "projected"}:
            return True
        if normalised in {"0", "false", "no", "off", "actual"}:
            return False
    raise ValueError("Distance display mode must be a boolean value")


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
            self._distance_overlay_enabled,
            self._distance_mounting,
            self._distance_use_projected,
            self._battery_limits,
            self._battery_capacity,
            self._stream_settings,
            self._reversing_aids,
            self._overlays_master_enabled,
            self._battery_overlay_enabled,
            self._reversing_overlay_enabled,
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
        bool,
        DistanceMounting,
        bool,
        BatteryLimits,
        int,
        StreamSettings,
        ReversingAidsConfig,
    ]:
        if not self._path.exists():
            return (
                Orientation(),
                DEFAULT_CAMERA_CHOICE,
                DEFAULT_RESOLUTION,
                DEFAULT_DISTANCE_ZONES,
                DEFAULT_DISTANCE_CALIBRATION,
                DEFAULT_DISTANCE_OVERLAY_ENABLED,
                DEFAULT_DISTANCE_MOUNTING,
                DEFAULT_DISTANCE_USE_PROJECTED,
                DEFAULT_BATTERY_LIMITS,
                DEFAULT_BATTERY_CAPACITY_MAH,
                DEFAULT_STREAM_SETTINGS,
                DEFAULT_REVERSING_AIDS,
                DEFAULT_OVERLAY_MASTER_ENABLED,
                DEFAULT_BATTERY_OVERLAY_ENABLED,
                DEFAULT_REVERSING_OVERLAY_ENABLED,
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
            distance_overlay_enabled = _parse_distance_overlay_enabled(
                distance_payload, default=DEFAULT_DISTANCE_OVERLAY_ENABLED
            )
            distance_mounting = _parse_distance_mounting(
                distance_payload, default=DEFAULT_DISTANCE_MOUNTING
            )
            distance_use_projected = _parse_distance_use_projected(
                distance_payload, default=DEFAULT_DISTANCE_USE_PROJECTED
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

            reversing_payload = payload.get("reversing_aids")
            reversing_aids = _parse_reversing_aids(
                reversing_payload, default=DEFAULT_REVERSING_AIDS
            )

            overlays_payload = payload.get("overlays")
            overlay_master_enabled = DEFAULT_OVERLAY_MASTER_ENABLED
            battery_overlay_enabled = DEFAULT_BATTERY_OVERLAY_ENABLED
            reversing_overlay_enabled = (
                reversing_aids.enabled if isinstance(reversing_aids, ReversingAidsConfig) else DEFAULT_REVERSING_OVERLAY_ENABLED
            )
            if isinstance(overlays_payload, Mapping):
                overlay_master_enabled = _parse_overlay_flag(
                    overlays_payload.get("master"), default=DEFAULT_OVERLAY_MASTER_ENABLED
                )
                battery_overlay_enabled = _parse_overlay_flag(
                    overlays_payload.get("battery"), default=DEFAULT_BATTERY_OVERLAY_ENABLED
                )
                if "distance" in overlays_payload:
                    distance_overlay_enabled = _parse_overlay_flag(
                        overlays_payload.get("distance"), default=distance_overlay_enabled
                    )
                reversing_overlay_enabled = _parse_overlay_flag(
                    overlays_payload.get("reversing_aids"), default=reversing_overlay_enabled
                )

            return (
                orientation,
                camera,
                resolution,
                distance_zones,
                distance_calibration,
                distance_overlay_enabled,
                distance_mounting,
                distance_use_projected,
                battery_limits,
                battery_capacity,
                stream_settings,
                reversing_aids,
                overlay_master_enabled,
                battery_overlay_enabled,
                reversing_overlay_enabled,
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
                "overlay_enabled": self._distance_overlay_enabled,
                "geometry": self._distance_mounting.to_dict(),
                "use_projected_distance": self._distance_use_projected,
            },
            "battery": {
                "limits": self._battery_limits.to_dict(),
                "capacity_mah": self._battery_capacity,
            },
            "stream": self._stream_settings.to_dict(),
            "reversing_aids": self._reversing_aids.to_dict(),
            "overlays": {
                "master": self._overlays_master_enabled,
                "battery": self._battery_overlay_enabled,
                "distance": self._distance_overlay_enabled,
                "reversing_aids": self._reversing_overlay_enabled,
            },
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

    def get_distance_overlay_enabled(self) -> bool:
        with self._lock:
            return self._distance_overlay_enabled

    def set_distance_overlay_enabled(self, value: Any) -> bool:
        enabled = _parse_distance_overlay_enabled(value, default=self._distance_overlay_enabled)
        with self._lock:
            self._distance_overlay_enabled = enabled
            self._save()
        return enabled

    def get_overlay_master_enabled(self) -> bool:
        with self._lock:
            return self._overlays_master_enabled

    def set_overlay_master_enabled(self, value: Any) -> bool:
        enabled = _parse_overlay_flag(value, default=self._overlays_master_enabled)
        with self._lock:
            self._overlays_master_enabled = enabled
            self._save()
        return enabled

    def get_battery_overlay_enabled(self) -> bool:
        with self._lock:
            return self._battery_overlay_enabled

    def set_battery_overlay_enabled(self, value: Any) -> bool:
        enabled = _parse_overlay_flag(value, default=self._battery_overlay_enabled)
        with self._lock:
            self._battery_overlay_enabled = enabled
            self._save()
        return enabled

    def get_reversing_overlay_enabled(self) -> bool:
        with self._lock:
            return self._reversing_overlay_enabled

    def set_reversing_overlay_enabled(self, value: Any) -> bool:
        enabled = _parse_overlay_flag(value, default=self._reversing_overlay_enabled)
        with self._lock:
            self._reversing_overlay_enabled = enabled
            if self._reversing_aids.enabled != enabled:
                self._reversing_aids = ReversingAidsConfig(
                    enabled=enabled,
                    left=self._reversing_aids.left,
                    right=self._reversing_aids.right,
                )
            self._save()
        return enabled

    def get_distance_mounting(self) -> DistanceMounting:
        with self._lock:
            return self._distance_mounting

    def set_distance_mounting(
        self, data: Mapping[str, Any] | Sequence[object] | DistanceMounting
    ) -> DistanceMounting:
        mounting = _parse_distance_mounting(data, default=self._distance_mounting)
        with self._lock:
            self._distance_mounting = mounting
            self._save()
        return mounting

    def get_distance_use_projected(self) -> bool:
        with self._lock:
            return self._distance_use_projected

    def set_distance_use_projected(self, value: Any) -> bool:
        use_projected = _parse_distance_use_projected(value, default=self._distance_use_projected)
        with self._lock:
            self._distance_use_projected = use_projected
            self._save()
        return use_projected

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

    def get_reversing_aids(self) -> ReversingAidsConfig:
        with self._lock:
            return self._reversing_aids

    def set_reversing_aids(
        self, data: Mapping[str, Any] | ReversingAidsConfig
    ) -> ReversingAidsConfig:
        aids = _parse_reversing_aids(data, default=self._reversing_aids)
        with self._lock:
            self._reversing_aids = aids
            self._reversing_overlay_enabled = aids.enabled
            self._save()
        return aids


__all__ = [
    "ConfigManager",
    "Orientation",
    "Resolution",
    "StreamSettings",
    "RESOLUTION_PRESETS",
    "DEFAULT_RESOLUTION",
    "DEFAULT_RESOLUTION_KEY",
    "ReversingAidsConfig",
    "ReversingAidSegment",
    "ReversingAidPoint",
    "generate_reversing_segments",
    "DEFAULT_REVERSING_AIDS",
    "DistanceMounting",
    "DEFAULT_DISTANCE_MOUNTING",
    "DEFAULT_DISTANCE_USE_PROJECTED",
]
