"""Tests for battery telemetry helpers."""
from __future__ import annotations
import asyncio

import pytest

from rev_cam.power import (
    BatteryMonitorError,
    BatteryStatus,
    SyntheticBatteryMonitor,
    create_battery_monitor,
)


def test_battery_status_to_dict() -> None:
    status = BatteryStatus(voltage=12.45, current=0.5, percentage=76.0)
    assert status.to_dict() == {
        "voltage": pytest.approx(12.45),
        "current": pytest.approx(0.5),
        "percentage": pytest.approx(76.0),
    }


def test_synthetic_monitor_within_bounds() -> None:
    monitor = SyntheticBatteryMonitor(min_voltage=11.5, max_voltage=12.5)
    status = asyncio.run(monitor.read())
    assert status.voltage is not None
    assert 11.5 <= status.voltage <= 12.5
    assert status.current == 0.0
    assert status.percentage is not None
    assert 0.0 <= status.percentage <= 100.0


def test_create_battery_monitor_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVCAM_BATTERY", "none")
    assert create_battery_monitor() is None


def test_create_battery_monitor_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVCAM_BATTERY", "synthetic")
    monitor = create_battery_monitor()
    assert isinstance(monitor, SyntheticBatteryMonitor)


def test_create_battery_monitor_voltage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVCAM_BATTERY", "synthetic")
    monkeypatch.setenv("REVCAM_BATTERY_MIN_VOLTAGE", "3.0")
    monkeypatch.setenv("REVCAM_BATTERY_MAX_VOLTAGE", "4.2")
    monitor = create_battery_monitor()
    assert isinstance(monitor, SyntheticBatteryMonitor)
    status = asyncio.run(monitor.read())
    assert status.voltage is not None
    assert 3.0 <= status.voltage <= 4.2
    assert status.percentage is not None
    assert 0.0 <= status.percentage <= 100.0


def test_create_battery_monitor_invalid_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVCAM_BATTERY", "mystery")
    with pytest.raises(BatteryMonitorError):
        create_battery_monitor()
