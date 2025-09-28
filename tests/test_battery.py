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
    sensor = _StubSensor(bus_voltage=4.2, current=150.0)
    monitor = BatteryMonitor(capacity_mah=1000, sensor_factory=lambda: sensor)

    reading = monitor.read()

    assert isinstance(reading, BatteryReading)
    assert reading.available is True
    assert reading.capacity_mah == 1000
    assert reading.voltage == pytest.approx(4.2)
    assert reading.percentage == pytest.approx(100.0)
    assert reading.charging is True
    assert reading.current_ma == pytest.approx(150.0)
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


def test_battery_monitor_treats_3v3_as_empty() -> None:
    sensor = _StubSensor(bus_voltage=3.3)
    monitor = BatteryMonitor(sensor_factory=lambda: sensor, smoothing_alpha=None)

    reading = monitor.read()

    assert reading.available is True
    assert reading.percentage == pytest.approx(0.0)
    assert reading.voltage == pytest.approx(3.3, abs=1e-3)


def test_battery_monitor_smooths_successive_readings() -> None:
    sensor = _StubSensor(bus_voltage=4.2, current=250.0)
    monitor = BatteryMonitor(sensor_factory=lambda: sensor, smoothing_alpha=0.25)

    first = monitor.read()
    assert first.available is True
    assert first.percentage == pytest.approx(100.0)
    assert first.voltage == pytest.approx(4.2, abs=1e-3)
    assert first.current_ma == pytest.approx(250.0)
    assert first.charging is True

    sensor.bus_voltage = 3.6
    sensor.current = -150.0

    second = monitor.read()
    assert second.available is True
    assert second.charging is False
    assert second.percentage == pytest.approx(99.0, abs=0.2)
    assert second.percentage < first.percentage
    assert second.percentage > 60.0
    assert second.voltage == pytest.approx(4.191, abs=0.002)
    assert second.current_ma == pytest.approx(-150.0)

    third = monitor.read()
    assert third.available is True
    assert third.charging is False
    assert third.percentage < second.percentage
    assert third.percentage > 50.0
    assert third.percentage == pytest.approx(97.3, abs=0.3)
    assert third.voltage == pytest.approx(4.174, abs=0.003)
    assert third.current_ma == pytest.approx(-150.0)

    final = third
    for _ in range(120):
        final = monitor.read()

    assert final.percentage == pytest.approx(38.0, abs=0.5)
    assert final.voltage == pytest.approx(3.6, abs=0.01)


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


def test_battery_monitor_falls_back_to_pi_ina219(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubPiIna219:
        def __init__(self, shunt_ohms: float, *, busnum: int | None = None, address: int) -> None:
            self._shunt_ohms = shunt_ohms
            self._busnum = busnum
            self._address = address

        def configure(self) -> None:
            pass

        def voltage(self) -> float:
            return 3.95

        def shunt_voltage(self) -> float:
            return 0.05

        def current(self) -> float:
            return 120.0

    class _StubRangeError(Exception):
        pass

    monkeypatch.delitem(sys.modules, "adafruit_ina219", raising=False)
    monkeypatch.delitem(sys.modules, "adafruit_extended_bus", raising=False)
    monkeypatch.delitem(sys.modules, "board", raising=False)

    module = types.ModuleType("ina219")
    module.INA219 = _StubPiIna219  # type: ignore[attr-defined]
    module.DeviceRangeError = _StubRangeError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ina219", module)

    monitor = BatteryMonitor()

    reading = monitor.read()

    assert reading.available is True
    assert reading.voltage == pytest.approx(4.0, abs=1e-6)
    assert reading.percentage is not None
    assert reading.current_ma == pytest.approx(120.0)
    assert reading.error is None


def test_battery_monitor_close_only_releases_supplied_sensor() -> None:
    class _StubI2CBus:
        def __init__(self) -> None:
            self.deinit_called = False

        def deinit(self) -> None:  # pragma: no cover - should not be called
            self.deinit_called = True

    class _StubDevice:
        def __init__(self, bus: _StubI2CBus) -> None:
            self.i2c = bus
            self.deinit_called = False

        def deinit(self) -> None:  # pragma: no cover - should not be called
            self.deinit_called = True

    class _ClosableSensor(_StubSensor):
        def __init__(self, device: _StubDevice) -> None:
            super().__init__(bus_voltage=4.0, current=100.0)
            self.deinit_called = False
            self.i2c_device = device

        def deinit(self) -> None:
            self.deinit_called = True

    bus = _StubI2CBus()
    device = _StubDevice(bus)
    sensor = _ClosableSensor(device)
    monitor = BatteryMonitor(sensor_factory=lambda: sensor)
    monitor.read()

    monitor.close()

    assert sensor.deinit_called is True
    assert device.deinit_called is False
    assert bus.deinit_called is False
    assert monitor.last_error is None


def test_battery_monitor_close_releases_owned_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class _StubBus:
        def __init__(self) -> None:
            self.deinit_called = False

        def deinit(self) -> None:
            self.deinit_called = True

    class _StubINA219:
        def __init__(self, bus: object, *, addr: int = BatteryMonitor.DEFAULT_I2C_ADDRESS) -> None:
            created["sensor"] = self
            created["bus"] = bus
            created["addr"] = addr
            self.bus_voltage = 4.0
            self.shunt_voltage = 0.0
            self.current = None
            self.deinit_called = False

        def deinit(self) -> None:
            self.deinit_called = True

    bus = _StubBus()
    monkeypatch.setitem(sys.modules, "adafruit_ina219", types.SimpleNamespace(INA219=_StubINA219))
    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(I2C=lambda: bus))

    monitor = BatteryMonitor()
    reading = monitor.read()

    assert reading.available is True

    monitor.close()

    sensor = created["sensor"]
    assert isinstance(sensor, _StubINA219)
    assert sensor.deinit_called is True
    assert isinstance(created["bus"], _StubBus)
    assert created["bus"].deinit_called is True
    assert monitor.last_error is None


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


def test_battery_overlay_respects_enabled_provider() -> None:
    np = pytest.importorskip("numpy")

    class _OverlayMonitor:
        def __init__(self) -> None:
            self.calls = 0

        def read(self) -> BatteryReading:
            self.calls += 1
            return BatteryReading(
                available=True,
                percentage=80.0,
                voltage=3.9,
                current_ma=-120.0,
                charging=False,
                capacity_mah=900,
                error=None,
            )

    monitor = _OverlayMonitor()
    overlay = create_battery_overlay(
        monitor,
        lambda: BatteryLimits(30.0, 10.0),
        enabled_provider=lambda: False,
    )
    frame = np.zeros((60, 80, 3), dtype=np.uint8)

    result = overlay(frame.copy())

    assert np.array_equal(result, frame)
    assert monitor.calls == 0


def test_battery_supervisor_triggers_shutdown_when_low() -> None:
    async def _exercise() -> None:
        reading = BatteryReading(
            available=True,
            percentage=4.0,
            voltage=3.35,
            current_ma=-150.0,
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
                    current_ma=200.0,
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
