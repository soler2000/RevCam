"""Unit tests for the INA219 battery monitoring helpers."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from rev_cam.battery import (
    BatteryLimits,
    BatteryMonitor,
    BatteryReading,
    BatterySupervisor,
    create_battery_overlay,
)


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


def test_battery_monitor_supports_extended_i2c_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _StubINA219:
        def __init__(self, bus: object, *, addr: int = BatteryMonitor.DEFAULT_I2C_ADDRESS) -> None:
            calls["bus"] = bus
            calls["addr"] = addr
            self.bus_voltage = 3.95
            self.shunt_voltage = 0.0
            self.current = None

    class _StubExtendedI2C:
        def __init__(self, bus_number: int) -> None:
            calls["bus_number"] = bus_number

    board_stub = types.ModuleType("board")

    def _fail_board_i2c() -> None:
        raise AssertionError("board.I2C should not be used when a bus override is provided")

    board_stub.I2C = _fail_board_i2c  # type: ignore[assignment]

    monkeypatch.setitem(sys.modules, "adafruit_ina219", types.SimpleNamespace(INA219=_StubINA219))
    monkeypatch.setitem(
        sys.modules,
        "adafruit_extended_bus",
        types.SimpleNamespace(ExtendedI2C=_StubExtendedI2C),
    )
    monkeypatch.setitem(sys.modules, "board", board_stub)

    monitor = BatteryMonitor(i2c_bus=29)

    reading = monitor.read()

    assert reading.available is True
    assert calls["bus_number"] == 29
    assert calls["bus"] is not None
    assert calls["addr"] == BatteryMonitor.DEFAULT_I2C_ADDRESS


def test_battery_monitor_reports_address_hint_on_init_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingINA219:
        def __init__(self, bus: object, *, addr: int = BatteryMonitor.DEFAULT_I2C_ADDRESS) -> None:
            raise RuntimeError("No device at address 0x40")

    class _StubBoard:
        @staticmethod
        def I2C() -> object:
            return object()

    monkeypatch.setitem(sys.modules, "adafruit_ina219", types.SimpleNamespace(INA219=_FailingINA219))
    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(I2C=_StubBoard.I2C))

    monitor = BatteryMonitor()

    reading = monitor.read()

    assert reading.available is False
    assert reading.error is not None
    assert "0x43" in reading.error
    assert "Failed to initialise INA219" in reading.error
    assert monitor.last_error == reading.error


def test_battery_limits_validation() -> None:
    with pytest.raises(ValueError):
        BatteryLimits(warning_percent=-1.0, shutdown_percent=5.0)
    with pytest.raises(ValueError):
        BatteryLimits(warning_percent=5.0, shutdown_percent=10.0)


def test_battery_limits_classification() -> None:
    limits = BatteryLimits(warning_percent=30.0, shutdown_percent=10.0)
    assert limits.classify(50.0) == "normal"
    assert limits.classify(25.0) == "warning"
    assert limits.classify(5.0) == "shutdown"
    assert limits.classify(None) is None


def test_create_battery_overlay_renders_box() -> None:
    np = pytest.importorskip("numpy")

    class _OverlayMonitor:
        def __init__(self) -> None:
            self.reading = BatteryReading(
                available=True,
                percentage=42.0,
                voltage=3.86,
                current_ma=-150.0,
                charging=False,
                capacity_mah=900,
                error=None,
            )

        def read(self) -> BatteryReading:
            return self.reading

    monitor = _OverlayMonitor()
    overlay = create_battery_overlay(monitor, lambda: BatteryLimits(30.0, 10.0))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    result = overlay(frame)

    assert result is frame
    assert np.any(result != 0)


def test_battery_supervisor_triggers_shutdown_when_low() -> None:
    async def _exercise() -> None:
        reading = BatteryReading(
            available=True,
            percentage=4.0,
            voltage=3.35,
            current_ma=150.0,
            charging=False,
            capacity_mah=1000,
            error=None,
        )

        class _SupervisorMonitor:
            def __init__(self) -> None:
                self.calls = 0

            def read(self) -> BatteryReading:
                self.calls += 1
                return reading

        triggered: list[BatteryReading] = []

        supervisor = BatterySupervisor(
            monitor=_SupervisorMonitor(),
            limits_provider=lambda: BatteryLimits(20.0, 5.0),
            check_interval=0.01,
            shutdown_handler=lambda info: triggered.append(info),
        )

        supervisor.start()
        await asyncio.sleep(0.05)
        await supervisor.aclose()

        assert triggered == [reading]

    asyncio.run(_exercise())


def test_battery_supervisor_ignores_charging_state() -> None:
    async def _exercise() -> None:
        class _ChargingMonitor:
            def read(self) -> BatteryReading:
                return BatteryReading(
                    available=True,
                    percentage=2.0,
                    voltage=3.3,
                    current_ma=-200.0,
                    charging=True,
                    capacity_mah=800,
                    error=None,
                )

        triggered: list[BatteryReading] = []

        supervisor = BatterySupervisor(
            monitor=_ChargingMonitor(),
            limits_provider=lambda: BatteryLimits(15.0, 5.0),
            check_interval=0.01,
            shutdown_handler=lambda info: triggered.append(info),
        )

        supervisor.start()
        await asyncio.sleep(0.05)
        await supervisor.aclose()

        assert triggered == []

    asyncio.run(_exercise())
