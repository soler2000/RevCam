"""Battery monitoring utilities using the INA219 sensor."""
from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional hardware dependency
    from ina219 import INA219, DeviceRangeError
except ImportError:  # pragma: no cover - hardware dependency not installed
    INA219 = None

    class DeviceRangeError(RuntimeError):
        """Fallback error type used when the INA219 library is unavailable."""


class BatteryMonitorError(RuntimeError):
    """Raised when the battery monitor cannot provide a reading."""


@dataclass(frozen=True, slots=True)
class BatteryStatus:
    """Represents an instantaneous reading from the battery monitor."""

    voltage: float
    percentage: float
    current: float | None = None
    estimated_capacity_mah: float | None = None

    def to_payload(self) -> dict[str, float]:
        """Serialise the measurement as a JSON-compatible payload."""

        payload: dict[str, float] = {
            "voltage": round(self.voltage, 3),
            "percentage": round(self.percentage, 2),
        }
        if self.current is not None:
            payload["current"] = round(self.current, 3)
        if self.estimated_capacity_mah is not None:
            payload["estimated_capacity_mah"] = round(self.estimated_capacity_mah, 2)
        return payload


class BaseBatteryMonitor(ABC):
    """Abstract battery monitor interface."""

    @abstractmethod
    async def get_status(self) -> BatteryStatus:  # pragma: no cover - interface only
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - optional override
        return None


class DummyBatteryMonitor(BaseBatteryMonitor):
    """Simple monitor that returns a fixed reading.

    This is primarily intended for development machines where the INA219
    hardware is not available but the API still needs to be exercised.
    """

    def __init__(self, voltage: float = 12.0, percentage: float = 100.0, *, current: float | None = None, full_capacity_mah: float | None = None) -> None:
        self._voltage = voltage
        self._percentage = percentage
        self._current = current
        self._capacity = full_capacity_mah

    async def get_status(self) -> BatteryStatus:
        estimated_capacity = (
            None
            if self._capacity is None
            else max(0.0, self._capacity * (self._percentage / 100.0))
        )
        return BatteryStatus(
            voltage=self._voltage,
            percentage=self._percentage,
            current=self._current,
            estimated_capacity_mah=estimated_capacity,
        )


class INA219BatteryMonitor(BaseBatteryMonitor):
    """Battery monitor backed by the INA219 sensor."""

    def __init__(
        self,
        shunt_ohms: float = 0.1,
        max_expected_amps: float = 2.0,
        *,
        address: Optional[int] = None,
        min_voltage: float = 10.0,
        max_voltage: float = 12.6,
        full_capacity_mah: float | None = None,
    ) -> None:
        if INA219 is None:  # pragma: no cover - hardware dependency not installed
            raise BatteryMonitorError("ina219 library is not available")
        if max_voltage <= min_voltage:
            raise BatteryMonitorError("max_voltage must be greater than min_voltage")
        self._ina = INA219(shunt_ohms, max_expected_amps, address=address)
        # Default configuration provides a good balance between range and
        # resolution for typical automotive 12 V systems.
        self._ina.configure()
        self._min_voltage = min_voltage
        self._max_voltage = max_voltage
        self._capacity = full_capacity_mah

    async def get_status(self) -> BatteryStatus:
        return await asyncio.to_thread(self._read_status)

    def _read_status(self) -> BatteryStatus:
        try:
            voltage = float(self._ina.supply_voltage())
        except DeviceRangeError as exc:  # pragma: no cover - hardware specific path
            raise BatteryMonitorError("Battery voltage out of range") from exc

        try:
            current_ma = self._ina.current()
        except DeviceRangeError:  # pragma: no cover - hardware specific path
            current_ma = None

        current = None if current_ma is None else float(current_ma) / 1000.0
        percentage = self._estimate_percentage(voltage)
        estimated_capacity = (
            None
            if self._capacity is None
            else max(0.0, self._capacity * (percentage / 100.0))
        )
        return BatteryStatus(
            voltage=voltage,
            percentage=percentage,
            current=current,
            estimated_capacity_mah=estimated_capacity,
        )

    def _estimate_percentage(self, voltage: float) -> float:
        span = self._max_voltage - self._min_voltage
        if span <= 0:
            return 0.0
        ratio = (voltage - self._min_voltage) / span
        return max(0.0, min(100.0, ratio * 100.0))


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise BatteryMonitorError(f"Environment variable {name} must be numeric") from exc


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value, 0)
    except ValueError as exc:
        raise BatteryMonitorError(f"Environment variable {name} must be an integer") from exc


def create_battery_monitor() -> BaseBatteryMonitor | None:
    """Create the battery monitor configured via environment variables."""

    mode = os.getenv("REVCAM_BATTERY_MONITOR", "auto").strip().lower()
    if mode in {"", "auto"}:
        target = "auto"
    elif mode in {"none", "off", "disabled"}:
        return None
    elif mode in {"dummy", "mock"}:
        return DummyBatteryMonitor()
    elif mode == "ina219":
        target = "ina219"
    else:
        raise ValueError(f"Unknown battery monitor mode: {mode}")

    min_voltage = _env_float("REVCAM_BATTERY_MIN_VOLTAGE", 10.0) or 0.0
    max_voltage = _env_float("REVCAM_BATTERY_MAX_VOLTAGE", 12.6) or 0.0
    capacity = _env_float("REVCAM_BATTERY_CAPACITY_MAH", None)
    shunt_ohms = _env_float("REVCAM_BATTERY_SHUNT_OHMS", 0.1)
    max_amps = _env_float("REVCAM_BATTERY_MAX_AMPS", 2.0)
    address = _env_int("REVCAM_BATTERY_I2C_ADDRESS", None)

    try:
        return INA219BatteryMonitor(
            shunt_ohms=shunt_ohms if shunt_ohms is not None else 0.1,
            max_expected_amps=max_amps if max_amps is not None else 2.0,
            address=address,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            full_capacity_mah=capacity,
        )
    except BatteryMonitorError:
        if target == "auto":
            return DummyBatteryMonitor()
        raise


__all__ = [
    "BaseBatteryMonitor",
    "BatteryMonitorError",
    "BatteryStatus",
    "DummyBatteryMonitor",
    "INA219BatteryMonitor",
    "create_battery_monitor",
]

