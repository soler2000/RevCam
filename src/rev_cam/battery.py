"""Battery monitoring utilities for INA219-backed LiPo packs."""

from __future__ import annotations

import asyncio
import logging
import math
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, Sequence

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

if TYPE_CHECKING:
    from .wifi import WiFiStatus

SensorFactory = Callable[[], object]


logger = logging.getLogger(__name__)


class _DriverUnavailableError(RuntimeError):
    """Raised when no supported INA219 driver can be loaded."""


@dataclass(frozen=True, slots=True)
class BatteryLimits:
    """Represents configurable warning and shutdown thresholds."""

    warning_percent: float = 20.0
    shutdown_percent: float = 5.0

    def __post_init__(self) -> None:
        warning = float(self.warning_percent)
        shutdown = float(self.shutdown_percent)
        if not (0.0 <= warning <= 100.0 and 0.0 <= shutdown <= 100.0):
            raise ValueError("Battery thresholds must be between 0 and 100")
        if warning < shutdown:
            raise ValueError("Warning threshold must be greater than shutdown threshold")
        object.__setattr__(self, "warning_percent", warning)
        object.__setattr__(self, "shutdown_percent", shutdown)

    def classify(self, percentage: float | None) -> str | None:
        """Classify *percentage* relative to the configured thresholds."""

        if percentage is None or not math.isfinite(percentage):
            return None
        if percentage <= self.shutdown_percent:
            return "shutdown"
        if percentage <= self.warning_percent:
            return "warning"
        return "normal"

    def to_dict(self) -> dict[str, float]:
        return {
            "warning_percent": float(self.warning_percent),
            "shutdown_percent": float(self.shutdown_percent),
        }


DEFAULT_BATTERY_LIMITS = BatteryLimits()


@dataclass(slots=True)
class BatteryReading:
    """Structured battery information exposed through the API."""

    available: bool
    percentage: float | None
    voltage: float | None
    current_ma: float | None
    charging: bool | None
    capacity_mah: int
    error: str | None = None

    def to_dict(self) -> dict[str, object | None]:
        """Serialise the reading into a JSON-friendly dictionary."""
        return {
            "available": self.available,
            "percentage": self.percentage,
            "voltage": self.voltage,
            "current_ma": self.current_ma,
            "charging": self.charging,
            "capacity_mah": self.capacity_mah,
            "error": self.error,
        }


@dataclass(slots=True)
class _SmoothingState:
    level: float | None = None
    trend: float = 0.0
    ema: float | None = None


class _PiIna219Adapter:
    """Bridge the :mod:`pi-ina219` driver to the attribute-based API we expect."""

    __slots__ = ("_driver", "_device_range_error")

    def __init__(self, driver: object, device_range_error: type[Exception] | None) -> None:
        self._driver = driver
        self._device_range_error = device_range_error

    def __getattr__(self, name: str) -> object:
        return getattr(self._driver, name)

    @property
    def bus_voltage(self) -> float:
        voltage = getattr(self._driver, "voltage")
        if callable(voltage):
            return float(voltage())
        return float(voltage)

    @property
    def shunt_voltage(self) -> float:
        shunt = getattr(self._driver, "shunt_voltage", None)
        if shunt is None:
            return 0.0
        if callable(shunt):
            try:
                return float(shunt())
            except Exception:
                return 0.0
        try:
            return float(shunt)
        except Exception:
            return 0.0

    @property
    def current(self) -> float | None:
        current = getattr(self._driver, "current", None)
        if current is None:
            return None
        try:
            value = current() if callable(current) else current
        except Exception as exc:  # pragma: no cover - defensive guard
            device_range_error = self._device_range_error
            if device_range_error is not None and isinstance(exc, device_range_error):
                return None
            raise
        try:
            return float(value)
        except Exception:
            return None

    def deinit(self) -> None:
        for method_name in ("close", "deinit", "shutdown", "sleep"):
            method = getattr(self._driver, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:  # pragma: no cover - defensive guard
                    pass
                break


class BatteryMonitor:
    """Provide high-level battery readings using an INA219 sensor.

    The monitor lazily instantiates the INA219 driver to keep unit tests free
    from hardware dependencies.  When the driver or hardware is unavailable the
    :meth:`read` method returns an unavailable reading that includes the last
    error message so the caller can surface it to the UI.
    """

    DEFAULT_I2C_ADDRESS = 0x43

    _LIPO_VOLTAGE_CURVE: Sequence[tuple[float, float]] = (
        (3.3, 0.0),
        (3.35, 5.0),
        (3.4, 12.0),
        (3.5, 25.0),
        (3.6, 38.0),
        (3.7, 55.0),
        (3.8, 70.0),
        (3.9, 85.0),
        (4.0, 92.0),
        (4.1, 97.0),
        (4.2, 100.0),
    )

    def __init__(
        self,
        capacity_mah: int = 1000,
        sensor_factory: SensorFactory | None = None,
        *,
        smoothing_alpha: float | None = 0.35,
        i2c_bus: int | None = None,
        i2c_address: int = DEFAULT_I2C_ADDRESS,
    ) -> None:
        self.capacity_mah = capacity_mah
        self._sensor_factory = sensor_factory
        self._sensor: object | None = None
        self._last_error: str | None = None
        self._last_logged_error: str | None = None
        self._i2c_bus = i2c_bus
        self._i2c_address = i2c_address
        self._owned_i2c_bus: object | None = None
        self._owns_i2c_bus = False
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        if smoothing_alpha is None:
            self._smoothing_alpha: float | None = None
            self._smoothing_beta: float | None = None
            self._smoothing_gamma: float | None = None
        else:
            alpha = float(smoothing_alpha)
            if not math.isfinite(alpha) or not (0.0 < alpha <= 1.0):
                raise ValueError("smoothing_alpha must be between 0 and 1")
            self._smoothing_alpha = alpha
            self._smoothing_beta = min(0.5, alpha * 0.5)
            self._smoothing_gamma = min(0.5, self._smoothing_beta * 0.5)
        self._smoothing_states: dict[str, _SmoothingState] = {
            "percentage": _SmoothingState(),
            "voltage": _SmoothingState(),
            "current": _SmoothingState(),
        }

    @property
    def last_error(self) -> str | None:
        """Return the most recent error generated by the monitor."""

        return self._last_error

    def _create_default_sensor(self) -> object:
        """Attempt to instantiate the INA219 driver."""

        attempts: list[str] = []

        sensor = self._try_create_adafruit_sensor(attempts)
        if sensor is not None:
            return sensor

        sensor = self._try_create_pi_ina219_sensor(attempts)
        if sensor is not None:
            return sensor

        detail = "; ".join(attempts) if attempts else "required drivers are missing"
        raise _DriverUnavailableError(f"INA219 driver unavailable ({detail})")

    def _try_create_adafruit_sensor(self, attempts: list[str]) -> object | None:
        try:  # Import lazily so unit tests do not require the dependencies.
            from adafruit_ina219 import INA219  # type: ignore
        except ModuleNotFoundError:
            attempts.append("install adafruit-circuitpython-ina219")
            return None
        except Exception as exc:  # pragma: no cover - import varies by environment
            attempts.append(f"adafruit INA219 import failed: {exc}")
            return None

        bus_number = self._i2c_bus
        if bus_number is not None:
            try:
                from .i2c_bus import I2CBusDependencyError, I2CBusRuntimeError, create_i2c_bus

                i2c = create_i2c_bus(bus_number)
            except I2CBusDependencyError as exc:
                attempts.append(exc.message)
                return None
            except I2CBusRuntimeError as exc:
                attempts.append(exc.message)
                return None
            self._owned_i2c_bus = i2c
            self._owns_i2c_bus = True
        else:
            try:
                import board  # type: ignore
            except ModuleNotFoundError:
                attempts.append("install adafruit-blinka to access board.I2C")
                return None
            except Exception as exc:  # pragma: no cover - import varies by environment
                attempts.append(f"board.I2C unavailable: {exc}")
                return None

            try:
                i2c = board.I2C()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - hardware specific
                attempts.append(f"Unable to access I2C bus: {exc}")
                return None
            self._owned_i2c_bus = i2c
            self._owns_i2c_bus = True

        try:
            return INA219(i2c, addr=self._i2c_address)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - hardware specific
            self._release_owned_i2c_bus()
            attempts.append(
                (
                    "Failed to initialise INA219 "
                    f"(expected address 0x{self._i2c_address:02X}): {exc}"
                )
            )
            return None

    def _try_create_pi_ina219_sensor(self, attempts: list[str]) -> object | None:
        try:
            from ina219 import INA219 as PiINA219  # type: ignore
        except ModuleNotFoundError:
            attempts.append("install pi-ina219")
            return None
        except Exception as exc:  # pragma: no cover - import varies by environment
            attempts.append(f"pi-ina219 import failed: {exc}")
            return None

        try:
            from ina219 import DeviceRangeError as PiDeviceRangeError  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            PiDeviceRangeError = None

        kwargs: dict[str, object] = {"address": self._i2c_address}
        if self._i2c_bus is not None:
            kwargs["busnum"] = self._i2c_bus

        try:
            driver = PiINA219(0.1, **kwargs)  # type: ignore[call-arg]
        except TypeError:
            driver = PiINA219(0.1, self._i2c_bus, self._i2c_address)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - hardware specific
            attempts.append(f"Failed to initialise pi-ina219: {exc}")
            return None

        configure = getattr(driver, "configure", None)
        if callable(configure):
            try:
                configure()
            except Exception as exc:  # pragma: no cover - hardware specific
                attempts.append(f"Failed to configure pi-ina219: {exc}")
                return None

        return _PiIna219Adapter(driver, PiDeviceRangeError)

    def _obtain_sensor(self) -> object | None:
        if self._sensor is not None:
            return self._sensor

        factory = self._sensor_factory
        if factory is None:
            try:
                sensor = self._create_default_sensor()
            except _DriverUnavailableError as exc:
                self._record_error(str(exc), level=logging.WARNING)
                return None
            except RuntimeError as exc:
                self._record_error(str(exc), exc_info=True)
                return None
        else:
            try:
                sensor = factory()
            except Exception as exc:
                self._record_error(str(exc), exc_info=True)
                return None

        self._sensor = sensor
        self._clear_error_state()
        return sensor

    def close(self) -> None:
        """Release the underlying sensor and any associated I2C resources."""

        sensor = self._sensor
        self._sensor = None
        self._clear_error_state()
        self._release_sensor(sensor)
        self._release_owned_i2c_bus()

    def _clear_error_state(self) -> None:
        self._last_error = None
        self._last_logged_error = None

    def _record_error(
        self, message: str, *, level: int = logging.ERROR, exc_info: bool = False
    ) -> None:
        detail = str(message).strip()
        if not detail:
            detail = "Battery monitor error"
        self._last_error = detail
        if detail != self._last_logged_error:
            self._logger.log(level, detail, exc_info=exc_info)
            self._last_logged_error = detail

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

    def _estimate_percentage(self, voltage: float) -> float:
        curve: Sequence[tuple[float, float]] = self._LIPO_VOLTAGE_CURVE
        if not curve:
            return 0.0

        if voltage <= curve[0][0]:
            return max(0.0, curve[0][1])
        if voltage >= curve[-1][0]:
            return min(100.0, curve[-1][1])

        for lower, upper in _pairwise(curve):
            lower_v, lower_p = lower
            upper_v, upper_p = upper
            if voltage <= upper_v:
                span = upper_v - lower_v
                if span <= 0:
                    return max(0.0, min(100.0, upper_p))
                ratio = (voltage - lower_v) / span
                percentage = lower_p + ratio * (upper_p - lower_p)
                return max(0.0, min(100.0, percentage))

        return max(0.0, min(100.0, curve[-1][1]))

    def _reset_smoothing(self) -> None:
        for state in self._smoothing_states.values():
            state.level = None
            state.trend = 0.0
            state.ema = None

    def _smooth_numeric(
        self,
        key: str,
        value: float | None,
        *,
        allow_cross_zero: bool = True,
    ) -> float | None:
        if value is None or not math.isfinite(value):
            state = self._smoothing_states[key]
            state.level = None
            state.trend = 0.0
            state.ema = None
            return None

        alpha = self._smoothing_alpha
        if alpha is None:
            state = self._smoothing_states[key]
            state.level = value
            state.trend = 0.0
            state.ema = value
            return value

        state = self._smoothing_states[key]
        beta = self._smoothing_beta
        assert beta is not None

        previous_level = state.level
        if previous_level is None or not math.isfinite(previous_level):
            state.level = value
            state.trend = 0.0
            state.ema = value
            return value

        if not allow_cross_zero and (
            previous_level == 0.0 or value == 0.0 or (previous_level > 0.0) != (value > 0.0)
        ):
            state.level = value
            state.trend = 0.0
            state.ema = value
            return value

        previous_trend = state.trend
        level = alpha * value + (1.0 - alpha) * (previous_level + previous_trend)
        trend = beta * (level - previous_level) + (1.0 - beta) * previous_trend

        state.level = level
        state.trend = trend
        gamma = self._smoothing_gamma
        if gamma is None:
            state.ema = level
            return level

        previous_ema = state.ema
        if previous_ema is None or not math.isfinite(previous_ema):
            ema = level
        else:
            ema = gamma * level + (1.0 - gamma) * previous_ema
        state.ema = ema
        return ema

    def read(self) -> BatteryReading:
        """Return the latest battery reading, handling hardware failures."""

        sensor = self._obtain_sensor()
        if sensor is None:
            self._reset_smoothing()
            return BatteryReading(
                available=False,
                percentage=None,
                voltage=None,
                current_ma=None,
                charging=None,
                capacity_mah=self.capacity_mah,
                error=self._last_error,
            )

        try:
            bus_voltage = float(getattr(sensor, "bus_voltage"))
        except Exception as exc:
            self._release_sensor(sensor)
            self._release_owned_i2c_bus()
            self._sensor = None
            self._record_error(f"Failed to read battery voltage: {exc}", exc_info=True)
            self._reset_smoothing()
            return BatteryReading(
                available=False,
                percentage=None,
                voltage=None,
                current_ma=None,
                charging=None,
                capacity_mah=self.capacity_mah,
                error=self._last_error,
            )

        try:
            shunt_voltage = float(getattr(sensor, "shunt_voltage", 0.0))
        except Exception:
            shunt_voltage = 0.0

        voltage = bus_voltage + shunt_voltage
        raw_percentage = self._estimate_percentage(voltage)

        current_ma: float | None
        charging: bool | None
        try:
            current_value = getattr(sensor, "current", None)
            if current_value is None:
                current_ma = None
                charging = None
            else:
                current_ma = float(current_value)
                charging = current_ma > 0
        except Exception:
            current_ma = None
            charging = None

        smoothed_voltage = self._smooth_numeric("voltage", float(voltage))
        smoothed_percentage = self._smooth_numeric("percentage", float(raw_percentage))
        if smoothed_percentage is not None:
            smoothed_percentage = max(0.0, min(100.0, smoothed_percentage))
        smoothed_current = self._smooth_numeric(
            "current",
            None if current_ma is None else float(current_ma),
            allow_cross_zero=False,
        )

        reading = BatteryReading(
            available=True,
            percentage=
            round(smoothed_percentage, 1)
            if smoothed_percentage is not None
            else None,
            voltage=round(smoothed_voltage, 3) if smoothed_voltage is not None else None,
            current_ma=round(smoothed_current, 2) if smoothed_current is not None else None,
            charging=charging,
            capacity_mah=self.capacity_mah,
            error=None,
        )
        self._clear_error_state()
        return reading


class BatterySupervisor:
    """Background task that triggers a shutdown when the pack is depleted."""

    def __init__(
        self,
        monitor: BatteryMonitor,
        limits_provider: Callable[[], BatteryLimits],
        *,
        check_interval: float = 30.0,
        shutdown_handler: Callable[[BatteryReading], Awaitable[None] | None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if check_interval <= 0:
            raise ValueError("check_interval must be positive")
        self._monitor = monitor
        self._limits_provider = limits_provider
        self._check_interval = float(check_interval)
        self._shutdown_handler = shutdown_handler or self._default_shutdown
        self._logger = logger or logging.getLogger(__name__)
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._shutdown_requested = False

    def start(self) -> None:
        """Start the background polling task."""

        if self._task is not None:
            return
        loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._task = loop.create_task(self._run())

    async def aclose(self) -> None:
        """Stop the supervisor and wait for the worker to exit."""

        task = self._task
        if task is None:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        try:
            await task
        finally:
            self._task = None
            self._stop_event = None
            self._shutdown_requested = False

    async def _run(self) -> None:
        assert self._stop_event is not None
        try:
            while not self._stop_event.is_set():
                await self._poll_once()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._check_interval
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:  # pragma: no cover - defensive guard
            raise

    async def _poll_once(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            reading = await loop.run_in_executor(None, self._monitor.read)
        except Exception:  # pragma: no cover - defensive guard
            self._logger.exception("Battery monitor raised an unexpected exception")
            return

        if not reading.available or reading.charging:
            return

        try:
            limits = self._limits_provider()
        except Exception:  # pragma: no cover - defensive guard
            self._logger.exception("Battery limits provider raised an exception")
            return

        state = limits.classify(reading.percentage)
        if state != "shutdown" or self._shutdown_requested:
            return

        await self._trigger_shutdown(reading)

    async def _trigger_shutdown(self, reading: BatteryReading) -> None:
        self._shutdown_requested = True
        handler = self._shutdown_handler
        try:
            result = handler(reading)
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # pragma: no cover - defensive guard
            self._logger.exception("Battery shutdown handler failed")
        finally:
            if self._stop_event is not None:
                self._stop_event.set()

    @staticmethod
    def _default_shutdown(reading: BatteryReading) -> None:
        del reading  # Unused but kept for signature consistency.
        message = "Low battery detected - shutting down"
        logger = logging.getLogger(__name__)
        commands = [
            ["sudo", "shutdown", "-h", "now", message],
            ["shutdown", "-h", "now", message],
        ]
        for command in commands:
            try:
                subprocess.run(command, check=False)
                return
            except FileNotFoundError:
                continue
            except Exception:  # pragma: no cover - depends on platform
                logger.exception("Failed to invoke shutdown command: %s", command[0])
                return
        logger.error("Shutdown command not found; unable to power off automatically")


_BATTERY_COLOURS = {
    "charging": (48, 209, 88),
    "normal": (10, 132, 255),
    "warning": (255, 159, 10),
    "critical": (255, 69, 58),
    "unknown": (255, 214, 10),
    "unavailable": (142, 142, 147),
}

def create_battery_overlay(
    monitor: BatteryMonitor,
    limits_provider: Callable[[], BatteryLimits],
    *,
    enabled_provider: Callable[[], bool] | None = None,
):
    """Return an overlay function that renders the current battery status."""

    def _overlay(frame):
        if enabled_provider is not None:
            try:
                if not enabled_provider():
                    if _np is None or not isinstance(frame, _np.ndarray):
                        monitor.read()
                    return frame
            except Exception:  # pragma: no cover - best effort guard
                logger.debug("Battery overlay enabled provider failed", exc_info=True)
        if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            monitor.read()
            return frame

        reading = monitor.read()
        try:
            limits = limits_provider()
        except Exception:  # pragma: no cover - defensive guard
            limits = DEFAULT_BATTERY_LIMITS
        return _render_battery_overlay(frame, reading, limits)

    return _overlay


def _render_battery_overlay(
    frame,
    reading: BatteryReading,
    limits: BatteryLimits,
):
    if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional guard
        return frame

    height, width = frame.shape[:2]
    if height < 24 or width < 60:
        return frame

    status = _classify_reading(reading, limits)
    colour = _BATTERY_COLOURS.get(status, _BATTERY_COLOURS["unavailable"])
    percentage = reading.percentage if reading.available else None
    if percentage is not None and not math.isfinite(percentage):
        percentage = None

    voltage = reading.voltage if reading.available else None
    if voltage is not None and not math.isfinite(voltage):
        voltage = None

    if percentage is not None:
        percentage_text = f"{percentage:.0f}%"
    else:
        percentage_text = None

    if voltage is not None:
        voltage_text = f"{voltage:.2f}V"
    else:
        voltage_text = "---"

    if percentage_text:
        battery_text = f"{percentage_text} {voltage_text}"
    else:
        battery_text = voltage_text

    scale = max(2, min(width, height) // 200)
    glyph_width = _FONT_WIDTH * scale
    glyph_height = _FONT_HEIGHT * scale
    char_spacing = 1 * scale
    padding = 4 * scale
    text_gap = 2 * scale

    battery_icon_width, battery_icon_height = _battery_icon_dimensions(scale)

    bar_height = max(
        glyph_height + padding * 2,
        battery_icon_height + padding * 2,
    )

    text_y = max(0, (bar_height - glyph_height) // 2)

    battery_text_width = _measure_text(battery_text, glyph_width, char_spacing)
    if battery_text_width:
        battery_total_width = battery_icon_width + text_gap + battery_text_width
    else:
        battery_total_width = battery_icon_width

    battery_icon_x = max(padding, width - padding - battery_total_width)
    battery_icon_y = max(0, (bar_height - battery_icon_height) // 2)
    _draw_battery_icon(frame, battery_icon_x, battery_icon_y, scale, percentage, colour)

    battery_text_x = battery_icon_x + battery_icon_width + (text_gap if battery_text_width else 0)

    text_colour = (255, 255, 255)

    if battery_text_width:
        _draw_text(frame, battery_text, battery_text_x, text_y, scale, text_colour)

    return frame


def create_wifi_overlay(
    wifi_status_provider: Callable[[], "WiFiStatus | None"],
    *,
    enabled_provider: Callable[[], bool] | None = None,
):
    """Return an overlay function that renders the current Wi-Fi status."""

    def _overlay(frame):
        if enabled_provider is not None:
            try:
                if not enabled_provider():
                    return frame
            except Exception:  # pragma: no cover - best effort guard
                logger.debug("Wi-Fi overlay enabled provider failed", exc_info=True)
        if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return frame

        try:
            status = wifi_status_provider()
        except Exception:  # pragma: no cover - best effort logging
            logger.debug("Wi-Fi status provider failed", exc_info=True)
            status = None
        return _render_wifi_overlay(frame, status)

    return _overlay


def _render_wifi_overlay(frame, status: "WiFiStatus | None"):
    if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional guard
        return frame

    height, width = frame.shape[:2]
    if height < 24 or width < 60:
        return frame

    wifi_level, wifi_text = _prepare_wifi_display(status)

    scale = max(2, min(width, height) // 200)
    glyph_width = _FONT_WIDTH * scale
    glyph_height = _FONT_HEIGHT * scale
    char_spacing = 1 * scale
    padding = 4 * scale
    text_gap = 2 * scale

    wifi_icon_width, wifi_icon_height = _wifi_icon_dimensions(scale)

    bar_height = max(
        glyph_height + padding * 2,
        wifi_icon_height + padding * 2,
    )

    text_y = max(0, (bar_height - glyph_height) // 2)

    wifi_icon_x = padding
    wifi_icon_y = max(0, (bar_height - wifi_icon_height) // 2)
    _draw_wifi_icon(frame, wifi_icon_x, wifi_icon_y, scale, wifi_level)

    wifi_text_x = wifi_icon_x + wifi_icon_width + text_gap
    wifi_text_width = _measure_text(wifi_text, glyph_width, char_spacing)
    max_wifi_width = max(0, width - padding - wifi_text_x)
    if wifi_text_width > max_wifi_width:
        if max_wifi_width >= glyph_width:
            wifi_text = "---"
            wifi_text_width = _measure_text(wifi_text, glyph_width, char_spacing)
        else:
            wifi_text = ""
            wifi_text_width = 0

    text_colour = (255, 255, 255)

    if wifi_text_width:
        _draw_text(frame, wifi_text, wifi_text_x, text_y, scale, text_colour)

    return frame


def _classify_reading(reading: BatteryReading, limits: BatteryLimits) -> str:
    if not reading.available:
        return "unavailable"
    if reading.charging:
        return "charging"
    state = limits.classify(reading.percentage)
    if state == "shutdown":
        return "critical"
    if state == "warning":
        return "warning"
    if state == "normal":
        return "normal"
    return "unknown"


def _prepare_wifi_display(status: "WiFiStatus | None") -> tuple[int, str]:
    if status is None:
        return 0, "---"

    if getattr(status, "hotspot_active", False):
        return 4, "HOTSPOT"

    if getattr(status, "connected", False):
        signal = getattr(status, "signal", None)
        level = _signal_to_level(signal)
        if level == 0:
            level = 1
        if isinstance(signal, (int, float)) and math.isfinite(signal):
            value = max(0.0, min(100.0, float(signal)))
            text = f"{value:.0f}%"
        else:
            text = "---"
        return level, text

    return 0, "NO WIFI"


def _signal_to_level(signal: object) -> int:
    try:
        value = float(signal)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(value):
        return 0
    value = max(0.0, min(100.0, value))
    if value >= 70.0:
        return 4
    if value >= 45.0:
        return 3
    if value >= 20.0:
        return 2
    if value > 5.0:
        return 1
    return 0


def _wifi_icon_dimensions(scale: int) -> tuple[int, int]:
    thickness = max(1, scale)
    spacing = thickness
    arcs = 3
    base_radius = thickness * 2
    max_radius = base_radius + arcs * (thickness + spacing)
    width = int(math.ceil((max_radius + thickness) * 2))
    height = int(math.ceil(max_radius + thickness))
    return width, height


def _draw_wifi_icon(frame, x: int, y: int, scale: int, level: int) -> None:
    width, height = _wifi_icon_dimensions(scale)
    thickness = max(1, scale)
    spacing = thickness
    arcs = 3
    base_radius = thickness * 2
    centre_x = x + width / 2.0
    centre_y = y + height

    palette = {
        0: (190, 190, 190),  # muted grey when nothing is available
        1: (60, 60, 220),  # red tint for extremely weak signal
        2: (0, 165, 255),  # amber for middling reception
        3: (60, 200, 60),  # green for good connection
        4: (40, 220, 40),  # brighter green for excellent connection
    }

    active_colour = palette.get(max(0, min(4, level)), (255, 255, 255))
    inactive_colour = (190, 190, 190)

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame.shape[1], x + width)
    y1 = min(frame.shape[0], y + height)
    if x0 >= x1 or y0 >= y1:
        return

    for py in range(y0, y1):
        for px in range(x0, x1):
            dx = (px + 0.5) - centre_x
            dy = centre_y - (py + 0.5)
            if dy < 0:
                continue
            distance = math.hypot(dx, dy)
            for arc_index in range(arcs):
                radius = base_radius + (arc_index + 1) * (thickness + spacing)
                if radius - thickness <= distance <= radius + thickness:
                    colour = active_colour if level >= arc_index + 2 else inactive_colour
                    frame[py, px] = colour
                    break

    dot_radius = thickness + max(0, scale // 2)
    dot_cx = int(round(centre_x))
    dot_cy = min(frame.shape[0] - 1, int(round(centre_y - thickness)))
    dot_colour = active_colour if level >= 1 else inactive_colour
    _fill_circle(frame, dot_cx, dot_cy, dot_radius, dot_colour)


def _battery_icon_dimensions(scale: int) -> tuple[int, int]:
    body_width = 12 * scale
    body_height = 6 * scale
    cap_width = max(scale, int(round(1.5 * scale)))
    icon_width = body_width + cap_width + scale
    icon_height = body_height
    return icon_width, icon_height


def _draw_battery_icon(
    frame,
    x: int,
    y: int,
    scale: int,
    percentage: float | None,
    fill_colour: tuple[int, int, int],
) -> None:
    body_width = 12 * scale
    body_height = 6 * scale
    cap_width = max(scale, int(round(1.5 * scale)))
    cap_gap = scale
    thickness = max(1, scale)

    icon_width = body_width + cap_gap + cap_width
    icon_height = body_height

    if x >= frame.shape[1] or y >= frame.shape[0]:
        return

    border_colour = (255, 255, 255)
    empty_colour = (40, 40, 40)

    _draw_rect_outline(frame, x, y, body_width, body_height, thickness, border_colour)

    inner_x = x + thickness
    inner_y = y + thickness
    inner_width = max(0, body_width - thickness * 2)
    inner_height = max(0, body_height - thickness * 2)
    if inner_width and inner_height:
        _fill_rect(frame, inner_x, inner_y, inner_width, inner_height, empty_colour)
        if percentage is not None:
            ratio = max(0.0, min(1.0, float(percentage) / 100.0))
        else:
            ratio = 0.0
        fill_width = int(round(inner_width * ratio))
        if fill_width > 0:
            _fill_rect(frame, inner_x, inner_y, fill_width, inner_height, fill_colour)

    cap_height = max(thickness * 2, body_height // 2)
    cap_x = x + body_width + cap_gap
    cap_y = y + (body_height - cap_height) // 2
    _fill_rect(frame, cap_x, cap_y, cap_width, cap_height, border_colour)


def _draw_rect_outline(
    frame, x: int, y: int, width: int, height: int, thickness: int, colour: tuple[int, int, int]
) -> None:
    if thickness <= 0:
        return
    _fill_rect(frame, x, y, width, thickness, colour)
    _fill_rect(frame, x, y + height - thickness, width, thickness, colour)
    _fill_rect(frame, x, y, thickness, height, colour)
    _fill_rect(frame, x + width - thickness, y, thickness, height, colour)


def _fill_rect(frame, x: int, y: int, width: int, height: int, colour: tuple[int, int, int]) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame.shape[1], x + width)
    y1 = min(frame.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] = colour


def _fill_circle(frame, cx: int, cy: int, radius: int, colour: tuple[int, int, int]) -> None:
    if radius <= 0:
        return
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(frame.shape[1], cx + radius + 1)
    y1 = min(frame.shape[0], cy + radius + 1)
    radius_sq = radius * radius
    for py in range(y0, y1):
        dy = py - cy
        for px in range(x0, x1):
            dx = px - cx
            if dx * dx + dy * dy <= radius_sq:
                frame[py, px] = colour


def _pairwise(values: Sequence[tuple[float, float]]) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    """Yield successive pairs from *values*."""

    for index in range(1, len(values)):
        yield values[index - 1], values[index]


__all__ = [
    "BatteryMonitor",
    "BatteryReading",
    "BatteryLimits",
    "DEFAULT_BATTERY_LIMITS",
    "BatterySupervisor",
    "create_battery_overlay",
    "create_wifi_overlay",
]
