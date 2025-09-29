"""Utilities for analysing trailer levelness and ramp requirements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _ensure_positive(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return numeric


@dataclass(frozen=True, slots=True)
class TrailerGeometry:
    """Physical characteristics of the trailer relevant to levelling."""

    axle_width_m: float = 2.4
    hitch_to_axle_m: float = 4.8
    length_m: float = 7.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "axle_width_m", _ensure_positive(self.axle_width_m, "Axle width"))
        object.__setattr__(self, "hitch_to_axle_m", _ensure_positive(self.hitch_to_axle_m, "Hitch distance"))
        object.__setattr__(self, "length_m", _ensure_positive(self.length_m, "Trailer length"))

    def to_dict(self) -> dict[str, float]:
        return {
            "axle_width_m": float(self.axle_width_m),
            "hitch_to_axle_m": float(self.hitch_to_axle_m),
            "length_m": float(self.length_m),
        }


@dataclass(frozen=True, slots=True)
class RampSpecification:
    """Represents a levelling ramp's physical properties."""

    length_m: float = 0.8
    height_m: float = 0.12

    def __post_init__(self) -> None:
        object.__setattr__(self, "length_m", _ensure_positive(self.length_m, "Ramp length"))
        object.__setattr__(self, "height_m", _ensure_positive(self.height_m, "Ramp height"))

    def to_dict(self) -> dict[str, float]:
        return {"length_m": float(self.length_m), "height_m": float(self.height_m)}


@dataclass(frozen=True, slots=True)
class LevelingSettings:
    """Aggregated settings for the trailer levelling system."""

    geometry: TrailerGeometry = TrailerGeometry()
    ramp: RampSpecification = RampSpecification()

    def to_dict(self) -> dict[str, object]:
        return {"geometry": self.geometry.to_dict(), "ramp": self.ramp.to_dict()}


@dataclass(frozen=True, slots=True)
class OrientationAngles:
    """Orientation expressed using roll, pitch and yaw (in degrees)."""

    roll: float
    pitch: float
    yaw: float

    def normalised(self) -> "OrientationAngles":
        """Normalise angles to the range [-180, 180]."""

        return OrientationAngles(
            roll=_wrap_degrees(self.roll),
            pitch=_wrap_degrees(self.pitch),
            yaw=_wrap_degrees(self.yaw),
        )


def _wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return wrapped


DEFAULT_TRAILER_GEOMETRY = TrailerGeometry()
DEFAULT_RAMP_SPECIFICATION = RampSpecification()
DEFAULT_LEVELING_SETTINGS = LevelingSettings()


def _parse_geometry(payload: Mapping[str, Any] | TrailerGeometry, *, default: TrailerGeometry) -> TrailerGeometry:
    if isinstance(payload, TrailerGeometry):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("Trailer geometry must be provided as a mapping")
    axle_width = payload.get("axle_width_m", default.axle_width_m)
    hitch_distance = payload.get("hitch_to_axle_m", default.hitch_to_axle_m)
    length = payload.get("length_m", default.length_m)
    return TrailerGeometry(axle_width, hitch_distance, length)


def _parse_ramp(payload: Mapping[str, Any] | RampSpecification, *, default: RampSpecification) -> RampSpecification:
    if isinstance(payload, RampSpecification):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("Ramp specification must be provided as a mapping")
    length = payload.get("length_m", default.length_m)
    height = payload.get("height_m", default.height_m)
    return RampSpecification(length, height)


def parse_leveling_settings(
    value: Mapping[str, Any] | LevelingSettings | None, *, default: LevelingSettings
) -> LevelingSettings:
    """Parse levelling settings from JSON-compatible input."""

    if value is None:
        return default
    if isinstance(value, LevelingSettings):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Levelling settings must be provided as a mapping")
    geometry_payload = value.get("geometry", value)
    ramp_payload = value.get("ramp", value)
    geometry = _parse_geometry(geometry_payload, default=default.geometry)
    ramp = _parse_ramp(ramp_payload, default=default.ramp)
    return LevelingSettings(geometry=geometry, ramp=ramp)


def _compute_side_adjustment(roll_deg: float, width_m: float) -> tuple[str, float, float]:
    """Return side-to-raise, raise height and inter-wheel height delta."""

    roll_rad = math.radians(roll_deg)
    half_width = width_m / 2.0
    required_raise = math.sin(roll_rad) * half_width
    total_difference = math.sin(roll_rad) * width_m
    if abs(required_raise) < 1e-4:
        return ("level", 0.0, abs(total_difference))
    side = "left" if required_raise > 0 else "right"
    return (side, abs(required_raise), abs(total_difference))


def _compute_ramp_usage(required_raise: float, ramp: RampSpecification) -> dict[str, float | bool]:
    if required_raise <= 0:
        return {
            "travel_m": 0.0,
            "travel_percent": 0.0,
            "limited": False,
            "achievable_raise_m": 0.0,
        }
    slope = ramp.height_m / ramp.length_m if ramp.length_m > 0 else float("inf")
    if slope <= 0 or not math.isfinite(slope):
        return {
            "travel_m": 0.0,
            "travel_percent": 0.0,
            "limited": True,
            "achievable_raise_m": 0.0,
        }
    travel = required_raise / slope
    limited = False
    achievable = required_raise
    max_travel = ramp.length_m * 0.9
    if required_raise > ramp.height_m:
        achievable = ramp.height_m
        travel = ramp.length_m
        limited = True
    if travel > max_travel:
        travel = max_travel
        achievable = travel * slope
        limited = True
    percent = travel / ramp.length_m * 100.0 if ramp.length_m > 0 else 0.0
    return {
        "travel_m": travel,
        "travel_percent": percent,
        "limited": limited,
        "achievable_raise_m": achievable,
    }


def compute_hitched_leveling(orientation: OrientationAngles, settings: LevelingSettings) -> dict[str, object]:
    """Return levelling guidance while the trailer remains hitched."""

    side, required_raise, difference = _compute_side_adjustment(
        orientation.roll, settings.geometry.axle_width_m
    )
    ramp_usage = _compute_ramp_usage(required_raise, settings.ramp)
    within_limits = not bool(ramp_usage["limited"])
    if side == "level":
        message = "Trailer is level across the axle"
    elif within_limits:
        message = (
            f"Raise the {side} wheel by {required_raise * 100:.1f} cm "
            f"(drive {ramp_usage['travel_m'] * 100:.1f} cm up the ramp)."
        )
    else:
        message = (
            f"Raise the {side} wheel by approximately {required_raise * 100:.1f} cm. "
            "Ramp limits prevent reaching full level in one step."
        )
    return {
        "side_to_raise": side,
        "required_raise_m": required_raise,
        "height_difference_m": difference,
        "ramp": ramp_usage,
        "within_ramp_limits": within_limits,
        "message": message,
    }


def compute_unhitched_leveling(orientation: OrientationAngles, settings: LevelingSettings) -> dict[str, object]:
    """Return levelling guidance for a free-standing trailer."""

    side, required_raise, difference = _compute_side_adjustment(
        orientation.roll, settings.geometry.axle_width_m
    )
    ramp_usage = _compute_ramp_usage(required_raise, settings.ramp)
    pitch_rad = math.radians(orientation.pitch)
    hitch_adjustment = math.tan(pitch_rad) * settings.geometry.hitch_to_axle_m
    if abs(hitch_adjustment) < 1e-4:
        hitch_direction = "level"
    elif hitch_adjustment > 0:
        hitch_direction = "lower"
    else:
        hitch_direction = "raise"
    summary: list[str] = []
    if side == "level":
        summary.append("Side-to-side level achieved")
    else:
        summary.append(f"Raise the {side} side by {required_raise * 100:.1f} cm")
    if hitch_direction == "level":
        summary.append("Hitch height is level")
    else:
        summary.append(
            f"{hitch_direction.capitalize()} the hitch by {abs(hitch_adjustment) * 100:.1f} cm"
        )
    max_deviation = max(abs(orientation.roll), abs(orientation.pitch))
    score = max(0.0, 1.0 - min(max_deviation / 10.0, 1.0))
    return {
        "side_to_raise": side,
        "required_raise_m": required_raise,
        "height_difference_m": difference,
        "ramp": ramp_usage,
        "hitch_adjustment_m": abs(hitch_adjustment),
        "hitch_direction": hitch_direction,
        "message": ". ".join(summary),
        "level_score": score,
        "max_deviation_deg": max_deviation,
    }


def evaluate_leveling(orientation: OrientationAngles, settings: LevelingSettings) -> dict[str, object]:
    """Return a summary of levelling recommendations."""

    orientation = orientation.normalised()
    hitched = compute_hitched_leveling(orientation, settings)
    unhitched = compute_unhitched_leveling(orientation, settings)
    return {
        "orientation": {
            "roll": orientation.roll,
            "pitch": orientation.pitch,
            "yaw": orientation.yaw,
        },
        "hitched": hitched,
        "unhitched": unhitched,
    }

