"""Helpers for reading and overlaying VL53L1X distance measurements."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from statistics import median
from threading import Lock
from typing import Callable, Deque, Sequence

try:  # pragma: no cover - optional dependency on numpy for overlays
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None

from .overlay_text import (
    FONT_BASE_HEIGHT as _FONT_HEIGHT,
    FONT_BASE_WIDTH as _FONT_WIDTH,
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


@dataclass(frozen=True, slots=True)
class DistanceCalibration:
    """Represents calibration adjustments applied to sensor readings."""

    offset_m: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        offset = float(self.offset_m)
        scale = float(self.scale)
        if not math.isfinite(offset):
            raise ValueError("Calibration offset must be a finite value")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Calibration scale must be a positive, finite value")
        object.__setattr__(self, "offset_m", offset)
        object.__setattr__(self, "scale", scale)

    def to_dict(self) -> dict[str, float]:
        return {"offset_m": float(self.offset_m), "scale": float(self.scale)}


DEFAULT_DISTANCE_CALIBRATION = DistanceCalibration()


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


@dataclass(frozen=True, slots=True)
class DistanceStatistics:
    """Summary statistics describing recent raw distance samples."""

    count: int
    minimum_m: float
    maximum_m: float
    mean_m: float
    median_m: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "DistanceStatistics":
        if not samples:
            raise ValueError("samples must contain at least one value")

        minimum = min(samples)
        maximum = max(samples)
        count = len(samples)
        mean = float(sum(samples) / count)
        med = float(median(samples))
        return cls(count=count, minimum_m=minimum, maximum_m=maximum, mean_m=mean, median_m=med)


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
        calibration: DistanceCalibration | None = None,
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
        self._raw_samples: Deque[float] = deque(maxlen=history_size)
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
        if calibration is None:
            self._calibration = DistanceCalibration()
        else:
            self._calibration = DistanceCalibration(calibration.offset_m, calibration.scale)

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

    def _record_raw_sample(self, measurement: float) -> None:
        if math.isfinite(measurement):
            self._raw_samples.append(float(measurement))

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

    def _apply_calibration(self, distance_m: float) -> float:
        calibration = self._calibration
        calibrated = distance_m * calibration.scale + calibration.offset_m
        return max(0.0, calibrated)

    def get_calibration(self) -> DistanceCalibration:
        with self._lock:
            calibration = self._calibration
            return DistanceCalibration(calibration.offset_m, calibration.scale)

    def set_calibration(
        self,
        calibration: DistanceCalibration | None = None,
        *,
        offset_m: float | None = None,
        scale: float | None = None,
    ) -> DistanceCalibration:
        if calibration is not None and (offset_m is not None or scale is not None):
            raise ValueError("Provide either a calibration object or offset/scale values, not both")
        with self._lock:
            current = self._calibration
            if calibration is None:
                new_calibration = DistanceCalibration(
                    offset_m=current.offset_m if offset_m is None else offset_m,
                    scale=current.scale if scale is None else scale,
                )
            else:
                new_calibration = DistanceCalibration(calibration.offset_m, calibration.scale)
            if new_calibration != current:
                self._calibration = new_calibration
                self._reset_smoothing()
                self._last_reading = None
                self._last_timestamp = 0.0
            return self._calibration

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

            self._record_raw_sample(measurement)
            filtered = self._filter_measurement(measurement)
            if filtered is None:
                reading = self._handle_invalid_sample(now, "Filtered invalid distance sample")
                self._last_timestamp = now
                return reading

            calibrated = self._apply_calibration(filtered)
            smoothed = self._apply_smoothing(calibrated)
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

    def get_raw_history(self) -> tuple[float, ...]:
        """Return a snapshot of the recent raw (pre-calibration) samples."""

        with self._lock:
            return tuple(self._raw_samples)

    def get_raw_statistics(self) -> DistanceStatistics | None:
        """Return descriptive statistics for the recent raw samples."""

        with self._lock:
            if not self._raw_samples:
                return None
            return DistanceStatistics.from_samples(tuple(self._raw_samples))


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
    monitor: DistanceMonitor,
    zonedist_provider: Callable[[], DistanceZones],
    enabled_provider: Callable[[], bool] | None = None,
):
    """Return an overlay function that renders the current distance reading."""

    def _overlay(frame):
        reading = monitor.read()
        if enabled_provider is not None and not enabled_provider():
            return frame

        if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return frame

        zones = zonedist_provider()
        zone = zones.classify(reading.distance_m)
        return _render_distance_overlay(frame, reading, zone)

    return _overlay


def _render_distance_overlay(frame, reading: DistanceReading, zone: str | None):
    if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional guard
        return frame

    height, width = frame.shape[:2]
    if height < 48 or width < 80:
        return frame

    zone_key = zone or "unavailable"
    colour = _ZONE_COLOURS.get(zone_key, _ZONE_COLOURS["unavailable"])
    label = _ZONE_LABELS.get(zone_key, _ZONE_LABELS["unavailable"])

    if reading.distance_m is not None and math.isfinite(reading.distance_m):
        distance_text = f"{reading.distance_m:.1f} m"
    else:
        distance_text = "---"

    main_scale = max(4, min(width, height) // 80)
    secondary_scale = max(2, main_scale // 2)
    line_spacing = max(4, secondary_scale)

    line_specs: list[tuple[str, int, float]] = [
        (distance_text, main_scale, 0.8),
        (label, secondary_scale, 0.55),
    ]

    measurements: list[tuple[int, int]] = []
    for text, scale, _ in line_specs:
        glyph_width = _FONT_WIDTH * scale
        char_spacing = 1 * scale
        text_width = _measure_text(text, glyph_width, char_spacing)
        text_height = _FONT_HEIGHT * scale
        measurements.append((text_width, text_height))

    block_width = max(width for width, _ in measurements)
    block_height = sum(height for _, height in measurements) + line_spacing * (len(line_specs) - 1)

    bottom_margin = max(line_spacing * 2, main_scale * 2)
    start_x = max(0, (width - block_width) // 2)
    start_y = max(0, height - bottom_margin - block_height)

    cursor_y = start_y
    for (text, scale, alpha), (line_width, line_height) in zip(line_specs, measurements):
        offset_x = start_x + max(0, (block_width - line_width) // 2)
        _blend_text(frame, text, offset_x, cursor_y, scale, colour, alpha)
        cursor_y += line_height + line_spacing

    return frame


def _blend_text(
    frame,
    text: str,
    x: int,
    y: int,
    scale: int,
    colour: tuple[int, int, int],
    alpha: float = 0.7,
) -> None:
    if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional guard
        return
    if frame.ndim < 3:
        return
    alpha = float(alpha)
    if alpha <= 0.0:
        return
    glyph_width = _FONT_WIDTH * scale
    char_spacing = 1 * scale
    text_width = _measure_text(text, glyph_width, char_spacing)
    text_height = _FONT_HEIGHT * scale
    if text_width <= 0 or text_height <= 0:
        return

    channels = frame.shape[2] if frame.ndim >= 3 else 1
    colour_array = _np.array(colour, dtype=_np.uint8)
    if channels >= 3:
        buffer = _np.zeros((text_height, text_width, channels), dtype=frame.dtype)
        target = buffer if channels == 3 else buffer[..., :3]
        draw_colour = tuple(int(c) for c in colour_array[:3])
        _draw_text(target, text, 0, 0, scale, draw_colour)
    else:
        buffer = _np.zeros((text_height, text_width), dtype=frame.dtype)
        draw_colour = int(colour_array[0]) if colour_array.size else 255
        _draw_text(buffer, text, 0, 0, scale, draw_colour)

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame.shape[1], x + text_width)
    y1 = min(frame.shape[0], y + text_height)
    if x1 <= x0 or y1 <= y0:
        return

    frame_slice = frame[y0:y1, x0:x1].astype(_np.float32)
    buffer_slice = buffer[y0 - y : y1 - y, x0 - x : x1 - x].astype(_np.float32)
    if frame_slice.ndim == 2:
        mask = buffer_slice > 0
    else:
        mask = _np.any(buffer_slice > 0, axis=2)
    if not _np.any(mask):
        return
    alpha = min(1.0, max(0.0, alpha))
    if frame_slice.ndim == 2:
        blended = frame_slice[mask] * (1.0 - alpha) + buffer_slice[mask] * alpha
    else:
        blended = frame_slice[mask] * (1.0 - alpha) + buffer_slice[mask] * alpha
    frame_slice[mask] = _np.clip(blended, 0, 255)
    frame[y0:y1, x0:x1] = frame_slice.astype(frame.dtype)
__all__ = [
    "DistanceCalibration",
    "DistanceMonitor",
    "DistanceReading",
    "DistanceZones",
    "DEFAULT_DISTANCE_CALIBRATION",
    "DEFAULT_DISTANCE_ZONES",
    "create_distance_overlay",
]

