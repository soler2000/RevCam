"""Battery monitoring helpers for RevCam."""
from __future__ import annotations

import asyncio
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class BatteryMonitorError(RuntimeError):
    """Raised when the battery monitor cannot be initialised or queried."""


@dataclass(slots=True)
class BatteryStatus:
    """Snapshot of the battery telemetry."""

    voltage: float | None
    current: float | None
    percentage: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "voltage": self.voltage,
            "current": self.current,
            "percentage": self.percentage,
        }


class BaseBatteryMonitor(ABC):
    """Abstract interface for battery monitors."""

    @abstractmethod
    async def read(self) -> BatteryStatus:  # pragma: no cover - interface only
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - optional override
        return None


class INA219BatteryMonitor(BaseBatteryMonitor):
    """Battery monitor backed by an INA219 sensor."""

    def __init__(
        self,
        *,
        address: int = 0x40,
        shunt_resistance_ohms: float = 0.1,
        max_expected_amps: float = 2.0,
        voltage_to_percentage: Callable[[float], float | None] | None = None,
    ) -> None:
        try:  # pragma: no cover - hardware dependent
            import board  # type: ignore
            import busio  # type: ignore
            from adafruit_ina219 import INA219  # type: ignore
        except ImportError as exc:  # pragma: no cover - hardware dependent
            raise BatteryMonitorError("INA219 dependencies are not installed") from exc

        try:  # pragma: no cover - hardware dependent
            i2c = busio.I2C(board.SCL, board.SDA)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise BatteryMonitorError("Unable to initialise I2C bus for INA219") from exc

        try:  # pragma: no cover - hardware dependent
            sensor = INA219(shunt_resistance_ohms, max_expected_amps, i2c, addr=address)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise BatteryMonitorError("Unable to communicate with INA219 sensor") from exc

        self._sensor = sensor
        self._lock = asyncio.Lock()
        self._voltage_to_percentage = voltage_to_percentage

    async def read(self) -> BatteryStatus:  # pragma: no cover - hardware dependent
        async with self._lock:
            return await asyncio.to_thread(self._read_sync)

    def _read_sync(self) -> BatteryStatus:  # pragma: no cover - hardware dependent
        try:
            bus_voltage = float(self._sensor.bus_voltage)
        except Exception as exc:
            raise BatteryMonitorError("Failed to read bus voltage from INA219") from exc

        current = None
        try:
            current = float(self._sensor.current) / 1000.0
        except Exception:
            # Some firmware builds may not expose current sensing – ignore gracefully.
            current = None

        percentage = None
        if self._voltage_to_percentage is not None:
            try:
                percentage = self._voltage_to_percentage(bus_voltage)
            except Exception:
                percentage = None

        return BatteryStatus(voltage=bus_voltage, current=current, percentage=percentage)


class SyntheticBatteryMonitor(BaseBatteryMonitor):
    """Synthetic battery monitor used when hardware is unavailable."""

    def __init__(self, *, min_voltage: float = 11.0, max_voltage: float = 12.6) -> None:
        self._min = float(min_voltage)
        self._max = float(max_voltage)
        if self._max <= self._min:
            raise BatteryMonitorError("max_voltage must be greater than min_voltage")
        self._start = time.monotonic()

    async def read(self) -> BatteryStatus:
        elapsed = time.monotonic() - self._start
        span = self._max - self._min
        voltage = self._min + (math.sin(elapsed / 30.0) + 1.0) / 2.0 * span
        percentage = (voltage - self._min) / span * 100.0
        return BatteryStatus(voltage=voltage, current=0.0, percentage=percentage)


def _build_voltage_percentage_mapper(min_voltage: float, max_voltage: float) -> Callable[[float], float | None]:
    span = max_voltage - min_voltage
    if span <= 0:
        raise BatteryMonitorError("Maximum voltage must be greater than minimum voltage")

    def _mapper(voltage: float) -> float | None:
        if voltage is None:
            return None
        percentage = (voltage - min_voltage) / span * 100.0
        return max(0.0, min(100.0, percentage))

    return _mapper


def _get_voltage_limits() -> tuple[float, float]:
    min_voltage_env = os.getenv("REVCAM_BATTERY_MIN_VOLTAGE")
    max_voltage_env = os.getenv("REVCAM_BATTERY_MAX_VOLTAGE")

    def _parse(value: str | None, fallback: float) -> float:
        if value is None:
            return fallback
        try:
            return float(value)
        except ValueError as exc:
            raise BatteryMonitorError(f"Invalid voltage value {value!r}") from exc

    min_voltage = _parse(min_voltage_env, 11.0)
    max_voltage = _parse(max_voltage_env, 12.6)
    if max_voltage <= min_voltage:
        raise BatteryMonitorError("Maximum voltage must exceed minimum voltage")
    return min_voltage, max_voltage


def create_battery_monitor() -> BaseBatteryMonitor | None:
    """Create the battery monitor selected via configuration."""

    choice = os.getenv("REVCAM_BATTERY", "auto").strip().lower()
    min_voltage = 11.0
    max_voltage = 12.6
    try:
        min_voltage, max_voltage = _get_voltage_limits()
    except BatteryMonitorError:
        # Keep defaults if parsing fails; validation happens later for actual monitors.
        min_voltage, max_voltage = 11.0, 12.6

    mapper: Callable[[float], float | None] | None = None
    try:
        mapper = _build_voltage_percentage_mapper(min_voltage, max_voltage)
    except BatteryMonitorError:
        mapper = None

    if choice in {"none", "off", "disable", "disabled"}:
        return None
    if choice == "synthetic":
        return SyntheticBatteryMonitor(min_voltage=min_voltage, max_voltage=max_voltage)

    if choice in {"ina219", "auto", "default"}:
        try:
            return INA219BatteryMonitor(voltage_to_percentage=mapper)
        except BatteryMonitorError:
            if choice != "auto":
                raise
            return SyntheticBatteryMonitor(min_voltage=min_voltage, max_voltage=max_voltage)

    raise BatteryMonitorError(f"Unknown battery monitor selection: {choice}")


__all__ = [
    "BaseBatteryMonitor",
    "BatteryMonitorError",
    "BatteryStatus",
    "INA219BatteryMonitor",
    "SyntheticBatteryMonitor",
    "create_battery_monitor",
]

