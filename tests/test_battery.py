"""Unit tests for the INA219 battery monitoring helpers."""

from __future__ import annotations

import pytest

from rev_cam.battery import BatteryMonitor, BatteryReading


class _StubSensor:
    def __init__(self, *, bus_voltage: float, shunt_voltage: float = 0.0, current: float | None = None) -> None:
        self.bus_voltage = bus_voltage
        self.shunt_voltage = shunt_voltage
        self.current = current


def test_battery_monitor_reports_reading() -> None:
    sensor = _StubSensor(bus_voltage=4.2, current=-150.0)
    monitor = BatteryMonitor(capacity_mah=1000, sensor_factory=lambda: sensor)

    reading = monitor.read()

    assert isinstance(reading, BatteryReading)
    assert reading.available is True
    assert reading.capacity_mah == 1000
    assert reading.voltage == pytest.approx(4.2)
    assert reading.percentage == pytest.approx(100.0)
    assert reading.charging is True
    assert reading.current_ma == pytest.approx(-150.0)
    assert reading.error is None


def test_battery_monitor_interpolates_voltage() -> None:
    sensor = _StubSensor(bus_voltage=3.85)
    monitor = BatteryMonitor(sensor_factory=lambda: sensor)

    reading = monitor.read()

    assert reading.available is True
    assert reading.voltage == pytest.approx(3.85, abs=1e-3)
    assert reading.current_ma is None
    assert reading.charging is None
    assert reading.percentage == pytest.approx(77.5)


def test_battery_monitor_surfaces_sensor_error() -> None:
    def _fail_factory() -> _StubSensor:
        raise RuntimeError("ina219 not detected")

    monitor = BatteryMonitor(capacity_mah=1200, sensor_factory=_fail_factory)

    reading = monitor.read()

    assert reading.available is False
    assert reading.capacity_mah == 1200
    assert reading.percentage is None
    assert reading.voltage is None
    assert reading.error is not None
    assert "ina219 not detected" in reading.error
    assert monitor.last_error is not None
    assert "ina219 not detected" in monitor.last_error
