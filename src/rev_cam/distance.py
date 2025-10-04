"""Helpers for reading and overlaying VL53L1X distance measurements."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from statistics import median
from threading import Event, Lock, Thread, current_thread
from typing import Callable, Deque

from .system_log import SystemLog

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


class _DriverUnavailableError(RuntimeError):
    """Raised when no supported VL53L1X driver can be loaded."""


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


class DistanceMonitor:
    """Read and smooth measurements from a VL53L1X time-of-flight sensor.

    A background sampler thread keeps the latest processed reading cached so
    that overlays and API handlers can fetch distance data without blocking on
    slow I²C calls.
    """

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
        auto_start: bool = True,
        system_log: SystemLog | None = None,
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
        self._last_logged_error: str | None = None
        self._last_reading: DistanceReading | None = None
        self._last_timestamp: float = 0.0
        self._lock = Lock()
        self._stop_event = Event()
        self._sampling_thread: Thread | None = None
        self._auto_start = bool(auto_start)
        self._system_log = system_log
        if calibration is None:
            self._calibration = DistanceCalibration()
        else:
            self._calibration = DistanceCalibration(calibration.offset_m, calibration.scale)
        self._owned_i2c_bus: object | None = None
        self._owns_i2c_bus = False
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        if self._auto_start:
            self.start_sampling()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _create_default_sensor(self) -> object:
        try:  # pragma: no cover - hardware dependency
            import adafruit_vl53l1x  # type: ignore
        except Exception as exc:  # pragma: no cover - hardware dependency
            raise _DriverUnavailableError("VL53L1X driver unavailable") from exc

        if self._i2c_bus is not None:
            try:  # pragma: no cover - optional dependency
                from adafruit_extended_bus import ExtendedI2C  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise _DriverUnavailableError(
                    "Extended I2C support unavailable; install adafruit-circuitpython-extended-bus"
                ) from exc

            try:  # pragma: no cover - hardware dependency
                i2c = ExtendedI2C(self._i2c_bus)
            except Exception as exc:  # pragma: no cover - hardware dependency
                raise RuntimeError(f"Unable to access I2C bus {self._i2c_bus}: {exc}") from exc
            self._owned_i2c_bus = i2c
            self._owns_i2c_bus = True
        else:
            try:  # pragma: no cover - hardware dependency
                import board  # type: ignore
            except Exception as exc:  # pragma: no cover - hardware dependency
                raise _DriverUnavailableError("VL53L1X driver unavailable") from exc

            try:  # pragma: no cover - hardware dependency
                i2c = board.I2C()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - hardware dependency
                raise RuntimeError("Unable to access I2C bus") from exc
            self._owned_i2c_bus = i2c
            self._owns_i2c_bus = True
        try:  # pragma: no cover - hardware dependency
            sensor = adafruit_vl53l1x.VL53L1X(i2c, address=self._i2c_address)
        except Exception as exc:  # pragma: no cover - hardware dependency
            self._release_owned_i2c_bus()
            raise RuntimeError(
                (
                    "Failed to initialise VL53L1X "
                    f"(expected address 0x{self._i2c_address:02X}): {exc}"
                )
            ) from exc

        self._configure_sensor(sensor)
        return sensor

    def _release_sensor(self, sensor: object | None) -> None:
        if sensor is None:
            return
        for method_name in ("deinit", "close", "shutdown"):
            method = getattr(sensor, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:  # pragma: no cover - defensive guard
                    pass
                break
        for attr in ("i2c_device", "_i2c_device", "device", "_device"):
            device = getattr(sensor, attr, None)
            if device is None:
                continue
            unlock = getattr(device, "unlock", None)
            if callable(unlock):
                try:
                    unlock()
                except Exception:  # pragma: no cover - defensive guard
                    pass

    def _release_owned_i2c_bus(self) -> None:
        bus = self._owned_i2c_bus
        self._owned_i2c_bus = None
        owns_bus = self._owns_i2c_bus
        self._owns_i2c_bus = False
        if not owns_bus or bus is None:
            return
        unlock = getattr(bus, "unlock", None)
        if callable(unlock):
            try:
                unlock()
            except Exception:  # pragma: no cover - defensive guard
                pass
        for method_name in ("deinit", "close", "shutdown"):
            method = getattr(bus, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:  # pragma: no cover - defensive guard
                    pass
                break

    def _configure_sensor(self, sensor: object) -> None:
        try:
            desired_budget = 140
            if hasattr(sensor, "distance_mode"):
                def _try_distance_modes(modes: tuple[object, ...]) -> bool:
                    for mode in modes:
                        try:
                            setattr(sensor, "distance_mode", mode)
                        except Exception:
                            continue
                        else:
                            return True
                    return False

                preferred_modes = (2, "long")
                fallback_modes = (1, "short")
                if not _try_distance_modes(preferred_modes):
                    _try_distance_modes(fallback_modes)
            if hasattr(sensor, "timing_budget"):
                try:
                    current_budget = getattr(sensor, "timing_budget")
                except Exception:
                    current_budget = None
                try:
                    if not isinstance(current_budget, (int, float)) or current_budget < desired_budget:
                        setattr(sensor, "timing_budget", desired_budget)
                except Exception:
                    pass
            if hasattr(sensor, "inter_measurement"):
                try:
                    budget = getattr(sensor, "timing_budget", None)
                except Exception:
                    budget = None
                try:
                    interval = getattr(sensor, "inter_measurement")
                except Exception:
                    interval = None
                target_interval = budget if isinstance(budget, (int, float)) and budget > 0 else desired_budget
                try:
                    if not isinstance(interval, (int, float)) or interval < target_interval:
                        setattr(sensor, "inter_measurement", target_interval)
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
            except _DriverUnavailableError as exc:
                self._record_error(str(exc), level=logging.WARNING)
                self._sensor = None
                return None
            except RuntimeError as exc:
                self._record_error(str(exc), exc_info=True)
                self._sensor = None
                return None
        else:
            try:
                sensor = factory()
            except Exception as exc:
                self._record_error(str(exc), exc_info=True)
                self._sensor = None
                return None
            self._configure_sensor(sensor)

        self._sensor = sensor
        self._clear_error_state()
        return sensor

    def close(self) -> None:
        """Release the underlying sensor and any I2C resources."""

        sensors_to_release: list[object] = []
        thread: Thread | None
        with self._lock:
            thread = self._sampling_thread
            self._sampling_thread = None
            self._stop_event.set()
            sensor = self._sensor
            self._sensor = None
            if sensor is not None:
                sensors_to_release.append(sensor)
            self._last_reading = None
            self._clear_error_state()
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            sensor = self._sensor
            self._sensor = None
            if sensor is not None and all(sensor is not existing for existing in sensors_to_release):
                sensors_to_release.append(sensor)
        for sensor in sensors_to_release:
            self._release_sensor(sensor)
        self._release_owned_i2c_bus()

    def _clear_error_state(self) -> None:
        previous_error = self._last_error
        self._last_error = None
        self._last_logged_error = None
        if previous_error and isinstance(self._system_log, SystemLog):
            self._system_log.record(
                "distance",
                "distance_recovered",
                "Distance monitor recovered.",
                metadata={"previous_error": previous_error},
            )

    def _record_error(
        self, message: str, *, level: int = logging.ERROR, exc_info: bool = False
    ) -> str:
        detail = str(message).strip()
        if not detail:
            detail = "Distance monitor error"
        self._last_error = detail
        if detail != self._last_logged_error:
            self._logger.log(level, detail, exc_info=exc_info)
            if isinstance(self._system_log, SystemLog):
                self._system_log.record(
                    "distance",
                    "distance_error",
                    detail,
                    metadata={"thread": current_thread().name},
                )
            self._last_logged_error = detail
        return detail

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

    def _handle_invalid_sample(
        self,
        now: float,
        message: str,
        *,
        level: int = logging.WARNING,
        exc_info: bool = False,
    ) -> DistanceReading:
        detail = self._record_error(message, level=level, exc_info=exc_info)
        last = self._last_reading
        if last and last.available and last.distance_m is not None:
            reading = DistanceReading(
                available=True,
                distance_m=last.distance_m,
                raw_distance_m=None,
                timestamp=now,
                error=detail,
            )
        else:
            reading = DistanceReading(
                available=False,
                distance_m=None,
                raw_distance_m=None,
                timestamp=now,
                error=detail,
            )
        self._last_reading = reading
        return reading

    def start_sampling(self) -> None:
        """Start a background thread that continually refreshes the sensor reading."""

        with self._lock:
            thread = self._sampling_thread
            if thread is not None and thread.is_alive():
                return
            self._stop_event.clear()
            thread = Thread(target=self._run_sampling_loop, name="DistanceMonitorSampler", daemon=True)
            self._sampling_thread = thread
        thread.start()

    def refresh(self) -> DistanceReading:
        """Synchronously fetch a new reading from the sensor."""

        now = time.monotonic()
        with self._lock:
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
            self._release_sensor(sensor)
            self._release_owned_i2c_bus()
            with self._lock:
                self._sensor = None
                message = f"Failed to read distance: {exc}"
                reading = self._handle_invalid_sample(
                    now, message, level=logging.ERROR, exc_info=True
                )
                self._last_timestamp = now
                return reading

        with self._lock:
            if measurement is None:
                reading = self._handle_invalid_sample(
                    now, "Distance measurement unavailable"
                )
                self._last_timestamp = now
                return reading

            filtered = self._filter_measurement(measurement)
            if filtered is None:
                reading = self._handle_invalid_sample(
                    now, "Filtered invalid distance sample"
                )
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
            self._clear_error_state()
            self._last_reading = reading
            self._last_timestamp = now
            return reading

    def read(self) -> DistanceReading:
        """Return the latest cached distance reading without touching the sensor."""

        if self._auto_start:
            self.start_sampling()
        with self._lock:
            if self._last_reading is not None:
                return self._last_reading
            now = time.monotonic()
            return DistanceReading(
                available=False,
                distance_m=None,
                raw_distance_m=None,
                timestamp=now,
                error=self._last_error,
            )

    def _run_sampling_loop(self) -> None:
        while not self._stop_event.is_set():
            start = time.monotonic()
            try:
                self.refresh()
            except Exception:  # pragma: no cover - defensive guard
                self._logger.exception("Unexpected error refreshing distance reading")
            elapsed = time.monotonic() - start
            wait_time = max(0.0, self._min_interval - elapsed)
            if wait_time == 0.0:
                continue
            if self._stop_event.wait(wait_time):
                break
        with self._lock:
            if self._sampling_thread is current_thread():
                self._sampling_thread = None


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


def _projected_distance_from_mounting(mounting, measured_distance) -> float | None:
    if mounting is None:
        return None
    try:
        height = float(getattr(mounting, "mount_height_m"))
        angle = float(getattr(mounting, "mount_angle_deg"))
    except (AttributeError, TypeError, ValueError):
        return None
    angle_rad = math.radians(angle)
    projection: float
    if measured_distance is not None and math.isfinite(measured_distance):
        try:
            projection = float(measured_distance) * math.sin(angle_rad)
        except (TypeError, ValueError):
            return None
    else:
        projection = height * math.tan(angle_rad)
    return projection if math.isfinite(projection) else None

def create_distance_overlay(
    monitor: DistanceMonitor,
    zonedist_provider: Callable[[], DistanceZones],
    enabled_provider: Callable[[], bool] | None = None,
    *,
    geometry_provider: Callable[[], object] | None = None,
    display_mode_provider: Callable[[], bool] | None = None,
):
    """Return an overlay function that renders the current distance reading."""

    monitor.start_sampling()

    def _overlay(frame):
        reading = monitor.read()
        if enabled_provider is not None and not enabled_provider():
            return frame

        if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return frame

        zones = zonedist_provider()
        mounting = geometry_provider() if geometry_provider is not None else None
        projection = _projected_distance_from_mounting(mounting, reading.distance_m)
        use_projected = False
        if display_mode_provider is not None:
            try:
                use_projected = bool(display_mode_provider())
            except Exception:  # pragma: no cover - defensive guard
                use_projected = False
        if use_projected and projection is not None and math.isfinite(projection):
            display_value = float(projection)
        elif reading.distance_m is not None and math.isfinite(reading.distance_m):
            display_value = float(reading.distance_m)
        else:
            display_value = None
        zone_distance = display_value if display_value is not None else reading.distance_m
        zone = zones.classify(zone_distance)
        return _render_distance_overlay(
            frame,
            reading,
            zone,
            display_distance=display_value,
        )

    return _overlay


def _render_distance_overlay(
    frame,
    reading: DistanceReading,
    zone: str | None,
    *,
    display_distance: float | None = None,
):
    if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional guard
        return frame

    height, width = frame.shape[:2]
    if height < 48 or width < 80:
        return frame

    zone_key = zone or "unavailable"
    colour = _ZONE_COLOURS.get(zone_key, _ZONE_COLOURS["unavailable"])
    label = _ZONE_LABELS.get(zone_key, _ZONE_LABELS["unavailable"])

    value = display_distance if display_distance is not None else reading.distance_m
    if value is not None and math.isfinite(value):
        distance_text = f"{value:.1f} m"
    else:
        distance_text = "---"

    main_scale = max(4, min(width, height) // 80)
    secondary_scale = max(2, main_scale // 2)
    line_spacing = max(4, secondary_scale)

    line_specs: list[tuple[str, int, float]] = [
        (distance_text, main_scale, 0.8),
        (label, secondary_scale, 0.55),
    ]

    line_entries: list[tuple[str, int, float, _TextBitmap]] = []
    measurements: list[tuple[int, int]] = []
    for text, scale, alpha in line_specs:
        bitmap = _get_text_bitmap(text, scale)
        measurements.append((bitmap.width, bitmap.height))
        line_entries.append((text, scale, alpha, bitmap))

    block_width = max(width for width, _ in measurements)
    block_height = sum(height for _, height in measurements) + line_spacing * (len(line_specs) - 1)

    bottom_margin = max(line_spacing * 2, main_scale * 2)
    start_x = max(0, (width - block_width) // 2)
    start_y = max(0, height - bottom_margin - block_height)

    cursor_y = start_y
    for (text, scale, alpha, bitmap), (line_width, line_height) in zip(line_entries, measurements):
        offset_x = start_x + max(0, (block_width - line_width) // 2)
        _blend_text(frame, text, offset_x, cursor_y, scale, colour, alpha, bitmap)
        cursor_y += line_height + line_spacing

    return frame


@dataclass(slots=True)
class _TextBitmap:
    """Cache entry storing the rendered mask for a text string."""

    width: int
    height: int
    mask: "_np.ndarray"


_TEXT_BITMAP_CACHE: dict[tuple[str, int], _TextBitmap] = {}


def _get_text_bitmap(text: str, scale: int) -> _TextBitmap:
    """Return a cached bitmap and metrics for *text* at *scale*."""

    key = (text.upper(), int(scale))
    cached = _TEXT_BITMAP_CACHE.get(key)
    if cached is not None:
        return cached

    glyph_width = _FONT_WIDTH * scale
    char_spacing = 1 * scale
    text_width = _measure_text(text, glyph_width, char_spacing)
    text_height = _FONT_HEIGHT * scale

    if text_width <= 0 or text_height <= 0:
        mask = _np.zeros((0, 0), dtype=_np.bool_)
    else:
        mask = _np.zeros((text_height, text_width), dtype=_np.bool_)
        _draw_text(mask, text, 0, 0, scale, True)

    bitmap = _TextBitmap(width=text_width, height=text_height, mask=mask)
    _TEXT_BITMAP_CACHE[key] = bitmap
    return bitmap


def _blend_text(
    frame,
    text: str,
    x: int,
    y: int,
    scale: int,
    colour: tuple[int, int, int],
    alpha: float = 0.7,
    bitmap: _TextBitmap | None = None,
) -> None:
    if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional guard
        return
    if frame.ndim < 3:
        return
    alpha = float(alpha)
    if alpha <= 0.0:
        return
    entry = bitmap if bitmap is not None else _get_text_bitmap(text, scale)
    text_width = entry.width
    text_height = entry.height
    if text_width <= 0 or text_height <= 0:
        return

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame.shape[1], x + text_width)
    y1 = min(frame.shape[0], y + text_height)
    if x1 <= x0 or y1 <= y0:
        return

    mask_slice = entry.mask[y0 - y : y1 - y, x0 - x : x1 - x]
    if mask_slice.size == 0:
        return

    active = mask_slice
    if not _np.any(active):
        return

    frame_slice = frame[y0:y1, x0:x1]
    alpha = min(1.0, max(0.0, float(alpha)))
    alpha_u8 = int(round(alpha * 255))
    if alpha_u8 <= 0:
        return
    inv_alpha = 255 - alpha_u8

    if frame_slice.ndim == 2:
        target_value = int(colour[0]) if colour else 255
        src_vals = frame_slice[active]
        if src_vals.size == 0:
            return
        blended = (
            src_vals.astype(_np.uint16) * inv_alpha
            + target_value * alpha_u8
        ) // 255
        frame_slice[active] = blended.astype(frame.dtype, copy=False)
        return

    channels = frame_slice.shape[2]
    if colour:
        last_value = int(colour[-1])
        colour_values = [int(colour[i]) if i < len(colour) else last_value for i in range(min(3, channels))]
    else:
        colour_values = [255] * min(3, channels)
    for channel, colour_value in enumerate(colour_values):
        channel_slice = frame_slice[..., channel]
        src_vals = channel_slice[active]
        if src_vals.size == 0:
            continue
        blended = (
            src_vals.astype(_np.uint16) * inv_alpha
            + colour_value * alpha_u8
        ) // 255
        channel_slice[active] = blended.astype(frame.dtype, copy=False)
__all__ = [
    "DistanceCalibration",
    "DistanceMonitor",
    "DistanceReading",
    "DistanceZones",
    "DEFAULT_DISTANCE_CALIBRATION",
    "DEFAULT_DISTANCE_ZONES",
    "create_distance_overlay",
]

