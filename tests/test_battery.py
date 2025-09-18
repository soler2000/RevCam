"""Tests for the battery monitoring module."""
from __future__ import annotations

import asyncio
import math

import pytest

from rev_cam import battery


class _FakeINA219:
    def __init__(self, shunt_ohms: float, max_expected_amps: float, address: int | None = None) -> None:
        self.shunt_ohms = shunt_ohms
        self.max_expected_amps = max_expected_amps
        self.address = address
        self.configured = False

    def configure(self) -> None:
        self.configured = True

    def supply_voltage(self) -> float:
        return 11.0

    def current(self) -> float:
        return 750.0  # milliamps


def test_dummy_battery_monitor_returns_fixed_values() -> None:
    monitor = battery.DummyBatteryMonitor(voltage=12.4, percentage=55.0, current=0.6, full_capacity_mah=2000)
    status = asyncio.run(monitor.get_status())
    assert status.voltage == 12.4
    assert status.percentage == 55.0
    assert status.current == 0.6
    assert status.estimated_capacity_mah == pytest.approx(1100.0)
    payload = status.to_payload()
    assert payload["voltage"] == pytest.approx(12.4, rel=1e-3)
    assert payload["percentage"] == pytest.approx(55.0, rel=1e-3)
    assert payload["current"] == pytest.approx(0.6, rel=1e-3)
    assert payload["estimated_capacity_mah"] == pytest.approx(1100.0, rel=1e-3)


def test_ina219_monitor_reads_sensor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(battery, "INA219", _FakeINA219)
    monitor = battery.INA219BatteryMonitor(
        shunt_ohms=0.1,
        max_expected_amps=3.2,
        min_voltage=10.0,
        max_voltage=12.0,
        full_capacity_mah=1800.0,
    )
    status = asyncio.run(monitor.get_status())
    assert status.voltage == pytest.approx(11.0)
    assert status.percentage == pytest.approx(50.0)
    assert status.current == pytest.approx(0.75)
    assert status.estimated_capacity_mah == pytest.approx(900.0)
    payload = status.to_payload()
    assert math.isclose(payload["voltage"], 11.0, rel_tol=1e-3)
    assert math.isclose(payload["percentage"], 50.0, rel_tol=1e-3)


def test_create_battery_monitor_auto_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(battery, "INA219", None)
    monkeypatch.setenv("REVCAM_BATTERY_MONITOR", "auto")
    monitor = battery.create_battery_monitor()
    assert isinstance(monitor, battery.DummyBatteryMonitor)


def test_create_battery_monitor_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVCAM_BATTERY_MONITOR", "disabled")
    assert battery.create_battery_monitor() is None


def test_create_battery_monitor_invalid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVCAM_BATTERY_MONITOR", "mystery")
    with pytest.raises(ValueError):
        battery.create_battery_monitor()
