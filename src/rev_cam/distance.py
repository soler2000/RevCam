"""Helpers for reading and overlaying VL53L1X distance measurements."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from statistics import median
from threading import Lock
from typing import Callable, Deque

try:  # pragma: no cover - optional dependency on numpy for overlays
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None

from .overlay_text import (
    apply_background as _apply_background,
    draw_text as _draw_text,
    measure_text as _measure_text,
)

SensorFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class DistanceZones:
    """Represents the configurable warning thresholds in metres."""

    caution: float
    warning: float
    danger: float

    def __post_init__(self) -> None:
        caution = float(self.caution)
        warning = float(self.warning)
        danger = float(self.danger)
        if caution <= 0 or warning <= 0 or danger <= 0:
            raise ValueError("Distance thresholds must be positive values")
        if not (caution >= warning >= danger):
            raise ValueError("Distance thresholds must decrease from caution to danger")
        object.__setattr__(self, "caution", caution)
        object.__setattr__(self, "warning", warning)
        object.__setattr__(self, "danger", danger)

    def to_dict(self) -> dict[str, float]:
        return {
            "caution": float(self.caution),
            "warning": float(self.warning),
            "danger": float(self.danger),
        }

    def classify(self, distance_m: float | None) -> str | None:
        if distance_m is None or not math.isfinite(distance_m):
            return None
        if distance_m <= self.danger:
            return "danger"
        if distance_m <= self.warning:
            return "warning"
        if distance_m <= self.caution:
            return "caution"
        return "clear"


DEFAULT_DISTANCE_ZONES = DistanceZones(caution=2.5, warning=1.5, danger=0.7)


@dataclass(slots=True)
class DistanceReading:
    """A processed distance reading expressed in metres."""

    available: bool
    distance_m: float | None
    raw_distance_m: float | None
    timestamp: float
    error: str | None = None

    def to_dict(self) -> dict[str, object | None]:
        return {
            "available": self.available,
            "distance_m": self.distance_m,
            "raw_distance_m": self.raw_distance_m,
            "timestamp": self.timestamp,
            "error": self.error,
        }


class DistanceMonitor:
    """Read and smooth measurements from a VL53L1X time-of-flight sensor."""

    DEFAULT_I2C_ADDRESS = 0x29

    def __init__(
        self,
        sensor_factory: SensorFactory | None = None,
        *,
        i2c_bus: int | None = None,
        i2c_address: int = DEFAULT_I2C_ADDRESS,
        smoothing_alpha: float = 0.6,
        history_size: int = 5,
        spike_threshold_m: float = 2.0,
        min_distance_m: float = 0.04,
        max_distance_m: float = 8.0,
        update_interval: float = 0.05,
    ) -> None:
        if not (0.0 < smoothing_alpha <= 1.0):
            raise ValueError("smoothing_alpha must be between 0 and 1")
        if history_size < 1:
            raise ValueError("history_size must be a positive integer")
        if spike_threshold_m <= 0:
            raise ValueError("spike_threshold_m must be positive")
        if min_distance_m <= 0 or max_distance_m <= 0 or min_distance_m >= max_distance_m:
            raise ValueError("Distance bounds must be positive and min < max")

        self._sensor_factory = sensor_factory
        self._sensor: object | None = None
        self._i2c_bus = i2c_bus
        self._i2c_address = i2c_address
        self._alpha = smoothing_alpha
        self._history: Deque[float] = deque(maxlen=history_size)
        self._spike_threshold = spike_threshold_m
        self._min_distance = min_distance_m
        self._max_distance = max_distance_m
        self._min_interval = max(0.0, float(update_interval))
        self._ema: float | None = None
        self._unit_scale: float | None = None
        self._spike_candidate: float | None = None
        self._spike_rejects: int = 0
        self._last_error: str | None = None
        self._last_reading: DistanceReading | None = None
        self._last_timestamp: float = 0.0
        self._lock = Lock()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _create_default_sensor(self) -> object:
        try:  # pragma: no cover - hardware dependency
            import adafruit_vl53l1x  # type: ignore
        except Exception as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError("VL53L1X driver unavailable") from exc

        if self._i2c_bus is not None:
            try:  # pragma: no cover - optional dependency
                from adafruit_extended_bus import ExtendedI2C  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Extended I2C support unavailable; install adafruit-circuitpython-extended-bus"
                ) from exc

            try:  # pragma: no cover - hardware dependency
                i2c = ExtendedI2C(self._i2c_bus)
            except Exception as exc:  # pragma: no cover - hardware dependency
                raise RuntimeError(f"Unable to access I2C bus {self._i2c_bus}: {exc}") from exc
        else:
            try:  # pragma: no cover - hardware dependency
                import board  # type: ignore
            except Exception as exc:  # pragma: no cover - hardware dependency
                raise RuntimeError("VL53L1X driver unavailable") from exc

            try:  # pragma: no cover - hardware dependency
                i2c = board.I2C()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - hardware dependency
                raise RuntimeError("Unable to access I2C bus") from exc

        try:  # pragma: no cover - hardware dependency
            sensor = adafruit_vl53l1x.VL53L1X(i2c, address=self._i2c_address)
        except Exception as exc:  # pragma: no cover - hardware dependency
            raise RuntimeError(
                (
                    "Failed to initialise VL53L1X "
                    f"(expected address 0x{self._i2c_address:02X}): {exc}"
                )
            ) from exc

        self._configure_sensor(sensor)
        return sensor

    def _configure_sensor(self, sensor: object) -> None:
        try:
            if hasattr(sensor, "distance_mode"):
                try:
                    setattr(sensor, "distance_mode", 1)
                except Exception:
                    try:
                        setattr(sensor, "distance_mode", "short")
                    except Exception:
                        pass
            if hasattr(sensor, "timing_budget"):
                try:
                    setattr(sensor, "timing_budget", 50)
                except Exception:
                    pass
            if hasattr(sensor, "start_ranging"):
                try:
                    getattr(sensor, "start_ranging")()
                except Exception:
                    pass
        except Exception:  # pragma: no cover - defensive guard
            return

    def _obtain_sensor(self) -> object | None:
        if self._sensor is not None:
            return self._sensor

        factory = self._sensor_factory
        if factory is None:
            try:
                sensor = self._create_default_sensor()
            except RuntimeError as exc:
                self._last_error = str(exc)
                self._sensor = None
                return None
        else:
            try:
                sensor = factory()
            except Exception as exc:
                self._last_error = str(exc)
                self._sensor = None
                return None
            self._configure_sensor(sensor)

        self._sensor = sensor
        self._last_error = None
        return sensor

    def _read_sensor_distance(self, sensor: object) -> float | None:
        value = getattr(sensor, "distance", None)
        if value is None:
            return None
        try:
            measurement = value() if callable(value) else value
        except Exception:
            return None
        if measurement is None:
            return None
        try:
            raw = float(measurement)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(raw):
            return None
        return self._convert_raw_distance(raw)

    def _convert_raw_distance(self, raw: float) -> float | None:
        """Normalise a raw sensor measurement into metres.

        Different VL53L1X drivers report readings in metres, centimetres,
        or millimetres.  Try each representation and keep the first value
        that falls inside the configured operating range.  Once a matching
        scale is found reuse it for subsequent samples so that we stay
        consistent while the sensor keeps reporting values in the same unit.
        """
        scale = self._unit_scale
        if scale is not None:
            distance_m = raw * scale
            if self._min_distance <= distance_m <= self._max_distance:
                return distance_m
            self._unit_scale = None

        conversions = (0.01, 0.001, 1.0)
        for scale in conversions:
            distance_m = raw * scale
            if self._min_distance <= distance_m <= self._max_distance:
                self._unit_scale = scale
                return distance_m
        return None

    def _reset_smoothing(self) -> None:
        self._history.clear()
        self._ema = None

    def _reset_spike_tracking(self) -> None:
        self._spike_candidate = None
        self._spike_rejects = 0

    def _filter_measurement(self, distance_m: float) -> float | None:
        if not math.isfinite(distance_m):
            self._reset_spike_tracking()
            return None
        if distance_m <= 0:
            self._reset_spike_tracking()
            return None
        if distance_m < self._min_distance or distance_m > self._max_distance:
            self._reset_spike_tracking()
            return None
        if self._ema is not None:
            delta = abs(distance_m - self._ema)
            if delta > self._spike_threshold:
                candidate = self._spike_candidate
                if candidate is None or abs(distance_m - candidate) > self._spike_threshold:
                    self._spike_candidate = distance_m
                    self._spike_rejects = 1
                    return None
                self._spike_rejects += 1
                if self._spike_rejects < 2:
                    return None
                self._reset_smoothing()
                self._reset_spike_tracking()
                return distance_m
        self._reset_spike_tracking()
        return distance_m

    def _apply_smoothing(self, value: float) -> float:
        history = self._history
        history.append(value)
        centre = median(history) if history else value
        if self._ema is None:
            self._ema = centre
        else:
            self._ema = self._alpha * centre + (1.0 - self._alpha) * self._ema
        return float(round(self._ema, 4))

    def _handle_invalid_sample(self, now: float, message: str) -> DistanceReading:
        last = self._last_reading
        if last and last.available and last.distance_m is not None:
            reading = DistanceReading(
                available=True,
                distance_m=last.distance_m,
                raw_distance_m=None,
                timestamp=now,
                error=message,
            )
        else:
            reading = DistanceReading(
                available=False,
                distance_m=None,
                raw_distance_m=None,
                timestamp=now,
                error=message,
            )
        self._last_error = message
        self._last_reading = reading
        return reading

    def read(self) -> DistanceReading:
        """Return a smoothed distance reading."""

        now = time.monotonic()
        with self._lock:
            if (
                self._last_reading is not None
                and now - self._last_timestamp < self._min_interval
            ):
                return self._last_reading

            sensor = self._obtain_sensor()
            if sensor is None:
                reading = DistanceReading(
                    available=False,
                    distance_m=self._last_reading.distance_m if self._last_reading else None,
                    raw_distance_m=None,
                    timestamp=now,
                    error=self._last_error,
                )
                self._last_reading = reading
                self._last_timestamp = now
                return reading

            try:
                measurement = self._read_sensor_distance(sensor)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._sensor = None
                message = f"Failed to read distance: {exc}"
                reading = self._handle_invalid_sample(now, message)
                self._last_timestamp = now
                return reading

            if measurement is None:
                reading = self._handle_invalid_sample(now, "Distance measurement unavailable")
                self._last_timestamp = now
                return reading

            filtered = self._filter_measurement(measurement)
            if filtered is None:
                reading = self._handle_invalid_sample(now, "Filtered invalid distance sample")
                self._last_timestamp = now
                return reading

            smoothed = self._apply_smoothing(filtered)
            reading = DistanceReading(
                available=True,
                distance_m=smoothed,
                raw_distance_m=filtered,
                timestamp=now,
                error=None,
            )
            self._last_error = None
            self._last_reading = reading
            self._last_timestamp = now
            return reading


_ZONE_COLOURS = {
    "danger": (255, 69, 58),
    "warning": (255, 159, 10),
    "caution": (255, 214, 10),
    "clear": (48, 209, 88),
    "unavailable": (142, 142, 147),
}

_ZONE_LABELS = {
    "danger": "DANGER",
    "warning": "WARNING",
    "caution": "CAUTION",
    "clear": "CLEAR",
    "unavailable": "N/A",
}

def create_distance_overlay(
    monitor: DistanceMonitor, zonedist_provider: Callable[[], DistanceZones]
):
    """Return an overlay function that renders the current distance reading."""

    def _overlay(frame):
        if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            monitor.read()
            return frame

        zones = zonedist_provider()
        reading = monitor.read()
        zone = zones.classify(reading.distance_m)
        return _render_distance_overlay(frame, reading, zone)

    return _overlay


def _render_distance_overlay(frame, reading: DistanceReading, zone: str | None):
    if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional guard
        return frame

    height, width = frame.shape[:2]
    if height < 24 or width < 60:
        return frame

    zone_key = zone or "unavailable"
    colour = _ZONE_COLOURS.get(zone_key, _ZONE_COLOURS["unavailable"])
    label = _ZONE_LABELS.get(zone_key, _ZONE_LABELS["unavailable"])

    if reading.distance_m is not None and math.isfinite(reading.distance_m):
        distance_text = f"{reading.distance_m:.1f}M"
    else:
        distance_text = "---"

    lines = [distance_text, label]

    scale = max(2, min(width, height) // 200)
    padding = 4 * scale
    line_spacing = 2 * scale
    glyph_width = 5 * scale
    glyph_height = 7 * scale
    char_spacing = 1 * scale

    line_widths = [_measure_text(line, glyph_width, char_spacing) for line in lines]
    box_width = max(line_widths) + padding * 2
    box_height = glyph_height * len(lines) + padding * 2 + line_spacing * (len(lines) - 1)

    x = padding * 2
    y = padding * 2
    if x + box_width > width:
        x = max(0, width - box_width - padding)
    if y + box_height > height:
        y = max(0, height - box_height - padding)

    _apply_background(frame, x, y, box_width, box_height, colour)

    text_x = x + padding
    text_y = y + padding
    for line, line_width in zip(lines, line_widths):
        offset_x = text_x
        _draw_text(frame, line, offset_x, text_y, scale, colour)
        text_y += glyph_height + line_spacing

    return frame
__all__ = [
    "DistanceMonitor",
    "DistanceReading",
    "DistanceZones",
    "DEFAULT_DISTANCE_ZONES",
    "create_distance_overlay",
]

