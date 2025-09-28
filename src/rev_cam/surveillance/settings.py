"""Surveillance mode configuration structures."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import json
import math


class LedBehaviour(str, Enum):
    """LED behaviour options when surveillance recording is active."""

    OFF = "off"
    STEADY = "steady"
    FLASH = "flash"


@dataclass(slots=True)
class PrivacyMask:
    """Rectangular exclusion region expressed as normalised coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "width", "height"):
            value = getattr(self, field_name)
            try:
                value_f = float(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
                raise ValueError("Privacy mask values must be numeric") from exc
            if not math.isfinite(value_f):  # pragma: no cover - defensive branch
                raise ValueError("Privacy mask values must be finite")
            if field_name in {"x", "y"} and not (0.0 <= value_f <= 1.0):
                raise ValueError("Mask coordinates must be between 0 and 1")
            if field_name in {"width", "height"} and not (0.0 < value_f <= 1.0):
                raise ValueError("Mask dimensions must be between 0 and 1")
            setattr(self, field_name, value_f)

    def to_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }


@dataclass(slots=True)
class SurveillanceSettings:
    """User configurable options for surveillance recording."""

    resolution: str = "1280x720"
    framerate: int = 15
    encoding: str = "h264"
    clip_max_length_s: int = 120
    pre_roll_s: int = 3
    post_motion_gap_s: int = 4
    audio_enabled: bool = False
    sensitivity: float = 0.5
    min_changed_area_percent: float = 1.5
    min_motion_duration_ms: int = 750
    detection_fps: int = 6
    privacy_masks: list[PrivacyMask] = field(default_factory=list)
    record_on_motion: bool = True
    led_behaviour: LedBehaviour = LedBehaviour.STEADY
    led_flash_rate_hz: float = 2.0
    webhook_url: str | None = None
    webhook_method: str = "POST"
    webhook_payload_template: str | None = None
    storage_max_days: int | None = 14
    storage_max_size_gb: float | None = 5.0
    export_path: str | None = None

    def __post_init__(self) -> None:
        if "x" in self.resolution:
            width, height = self.resolution.split("x", 1)
            try:
                width_i = int(width)
                height_i = int(height)
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise ValueError("Resolution must be formatted as <width>x<height>") from exc
            if width_i <= 0 or height_i <= 0:
                raise ValueError("Resolution values must be positive integers")
        if self.framerate < 1 or self.framerate > 60:
            raise ValueError("Framerate must be between 1 and 60 fps")
        if self.clip_max_length_s <= 0:
            raise ValueError("Clip length must be positive")
        if self.pre_roll_s < 0:
            raise ValueError("Pre-roll must not be negative")
        if self.post_motion_gap_s < 0:
            raise ValueError("Post-motion gap must not be negative")
        if self.min_changed_area_percent < 0:
            raise ValueError("Minimum changed area must not be negative")
        if self.min_motion_duration_ms < 0:
            raise ValueError("Minimum motion duration must not be negative")
        if self.detection_fps < 1 or self.detection_fps > 30:
            raise ValueError("Detection FPS must be between 1 and 30")
        if self.led_flash_rate_hz <= 0 and self.led_behaviour is LedBehaviour.FLASH:
            raise ValueError("LED flash rate must be positive when flashing")
        self.privacy_masks = [
            mask if isinstance(mask, PrivacyMask) else PrivacyMask(**dict(mask))
            for mask in self.privacy_masks
        ]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["privacy_masks"] = [mask.to_dict() for mask in self.privacy_masks]
        data["led_behaviour"] = self.led_behaviour.value
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurveillanceSettings":
        data: dict[str, Any] = dict(payload)
        masks: Iterable[Any] = data.get("privacy_masks", [])
        data["privacy_masks"] = [
            mask if isinstance(mask, PrivacyMask) else PrivacyMask(**dict(mask))
            for mask in masks
        ]
        behaviour = data.get("led_behaviour", LedBehaviour.STEADY)
        if isinstance(behaviour, str):
            data["led_behaviour"] = LedBehaviour(behaviour)
        elif isinstance(behaviour, LedBehaviour):
            data["led_behaviour"] = behaviour
        else:  # pragma: no cover - defensive branch
            raise ValueError("Invalid LED behaviour value")
        return cls(**data)


class SurveillanceSettingsStore:
    """Simple JSON backed persistence for :class:`SurveillanceSettings`."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> SurveillanceSettings:
        if not self._path.exists():
            return SurveillanceSettings()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
            raise ValueError("Invalid surveillance settings JSON") from exc
        return SurveillanceSettings.from_dict(raw)

    def save(self, settings: SurveillanceSettings) -> None:
        payload = settings.to_dict()
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "LedBehaviour",
    "PrivacyMask",
    "SurveillanceSettings",
    "SurveillanceSettingsStore",
]
