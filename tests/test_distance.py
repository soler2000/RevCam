"""Tests for the VL53L1X distance helpers."""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("numpy")

import numpy as np

from rev_cam.distance import (
    DistanceCalibration,
    DistanceMonitor,
    DistanceZones,
    create_distance_overlay,
)


class _SequenceSensor:
    def __init__(self, readings: list[float]):
        if not readings:
            raise ValueError("readings must not be empty")
        self._readings = list(readings)
        self._index = 0

    @property
    def distance(self) -> float:
        if self._index >= len(self._readings):
            return self._readings[-1]
        value = self._readings[self._index]
        self._index += 1
        return value


def test_distance_monitor_reports_reading() -> None:
    sensor = _SequenceSensor([123.0])
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )

    reading = monitor.refresh()
    cached = monitor.read()

    assert reading.available is True
    assert reading.distance_m == pytest.approx(1.23, rel=1e-6)
    assert reading.raw_distance_m == pytest.approx(1.23, rel=1e-6)
    assert reading.error is None
    assert cached == reading


def test_distance_monitor_filters_invalid_samples() -> None:
    sensor = _SequenceSensor([123.0, 9999.0, 140.0])
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )

    first = monitor.refresh()
    second = monitor.refresh()
    third = monitor.refresh()

    assert second.available is True
    assert second.distance_m == pytest.approx(first.distance_m)
    assert second.error is not None
    assert third.available is True
    assert third.distance_m > first.distance_m
    assert third.error is None


def test_distance_monitor_handles_measurements_in_metres() -> None:
    sensor = _SequenceSensor([1.8, 1.75])
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )

    first = monitor.refresh()
    second = monitor.refresh()

    assert first.distance_m == pytest.approx(1.8, rel=1e-6)
    assert first.raw_distance_m == pytest.approx(1.8, rel=1e-6)
    assert second.raw_distance_m == pytest.approx(1.75, rel=1e-6)
    assert second.distance_m == pytest.approx(1.785, abs=1e-6)


def test_distance_monitor_recovers_from_legitimate_jump() -> None:
    sensor = _SequenceSensor([20.0, 500.0, 500.0])
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )

    first = monitor.refresh()
    second = monitor.refresh()
    third = monitor.refresh()

    assert first.distance_m == pytest.approx(0.2, rel=1e-6)
    assert second.distance_m == pytest.approx(first.distance_m)
    assert second.error == "Filtered invalid distance sample"
    assert third.distance_m == pytest.approx(5.0, rel=1e-6)
    assert third.error is None


def test_distance_monitor_reports_unavailable_when_sensor_missing() -> None:
    def _failing_factory() -> object:
        raise RuntimeError("sensor offline")

    monitor = DistanceMonitor(
        sensor_factory=_failing_factory,
        update_interval=0.0,
        auto_start=False,
    )

    reading = monitor.refresh()

    assert reading.available is False
    assert reading.distance_m is None
    assert reading.error == "sensor offline"
    assert monitor.last_error == "sensor offline"


def test_distance_monitor_read_does_not_sample_sensor() -> None:
    sensor = _SequenceSensor([200.0])
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )

    placeholder = monitor.read()

    assert placeholder.available is False
    assert sensor._index == 0

    monitor.refresh()
    cached = monitor.read()

    assert cached.available is True
    assert sensor._index == 1


def test_distance_monitor_prefers_long_range_mode() -> None:
    class _ConfigurableSensor:
        def __init__(self) -> None:
            self.distance_mode = None
            self.timing_budget = None
            self.started = False

        def start_ranging(self) -> None:
            self.started = True

    sensor = _ConfigurableSensor()
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )

    monitor.refresh()

    assert sensor.distance_mode in (2, "long")
    assert sensor.started is True


def test_distance_monitor_falls_back_when_long_mode_unavailable() -> None:
    class _LimitedSensor:
        def __init__(self) -> None:
            self._distance_mode = None

        @property
        def distance_mode(self):  # type: ignore[override]
            return self._distance_mode

        @distance_mode.setter
        def distance_mode(self, value):  # type: ignore[override]
            if value in (2, "long"):
                raise ValueError("unsupported mode")
            self._distance_mode = value

        def start_ranging(self) -> None:
            pass

    sensor = _LimitedSensor()
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )

    monitor.refresh()

    assert sensor.distance_mode in (1, "short")


def test_distance_monitor_applies_calibration() -> None:
    sensor = _SequenceSensor([100.0])
    calibration = DistanceCalibration(offset_m=-0.1, scale=0.01)
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        calibration=calibration,
        update_interval=0.0,
        auto_start=False,
    )

    reading = monitor.refresh()

    assert reading.raw_distance_m == pytest.approx(1.0, rel=1e-6)
    assert reading.distance_m == pytest.approx(0.0, abs=1e-6)


def test_distance_monitor_set_calibration_updates_immediately() -> None:
    sensor = _SequenceSensor([100.0, 100.0])
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=5.0,
        auto_start=False,
    )

    first = monitor.refresh()
    assert first.distance_m == pytest.approx(1.0, rel=1e-6)

    updated = monitor.set_calibration(offset_m=-0.25)
    assert updated.offset_m == pytest.approx(-0.25, rel=1e-6)

    second = monitor.refresh()
    assert second.raw_distance_m == pytest.approx(1.0, rel=1e-6)
    assert second.distance_m == pytest.approx(0.75, rel=1e-6)


def test_distance_overlay_draws_on_frame() -> None:
    sensor = _SequenceSensor([150.0])
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )
    zones = DistanceZones(caution=2.0, warning=1.0, danger=0.5)
    overlay = create_distance_overlay(monitor, lambda: zones)

    monitor.refresh()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = overlay(frame)

    assert result is frame
    assert np.any(result != 0)


def test_distance_overlay_handles_non_numpy_frames() -> None:
    sensor = _SequenceSensor([150.0])
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )
    overlay = create_distance_overlay(monitor, lambda: DistanceZones(2.0, 1.0, 0.5))

    frame = [[0, 0, 0], [0, 0, 0]]
    result = overlay(frame)

    assert result == frame


def test_distance_monitor_close_only_releases_supplied_sensor() -> None:
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

    class _ClosableSensor:
        def __init__(self, device: _StubDevice) -> None:
            self.distance = 100.0
            self.deinit_called = False
            self.i2c_device = device

        def deinit(self) -> None:
            self.deinit_called = True

    bus = _StubI2CBus()
    device = _StubDevice(bus)
    sensor = _ClosableSensor(device)
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        update_interval=0.0,
        auto_start=False,
    )
    monitor.refresh()

    monitor.close()

    assert sensor.deinit_called is True
    assert device.deinit_called is False
    assert bus.deinit_called is False
    assert monitor.last_error is None


def test_distance_monitor_close_releases_owned_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class _StubBus:
        def __init__(self) -> None:
            self.deinit_called = False

        def deinit(self) -> None:
            self.deinit_called = True

    class _StubVL53:
        def __init__(self, i2c: object, *, address: int = DistanceMonitor.DEFAULT_I2C_ADDRESS) -> None:
            created["sensor"] = self
            created["bus"] = i2c
            created["address"] = address
            self.distance = 100.0
            self.deinit_called = False

        def deinit(self) -> None:
            self.deinit_called = True

    bus = _StubBus()
    monkeypatch.setitem(sys.modules, "adafruit_vl53l1x", types.SimpleNamespace(VL53L1X=_StubVL53))
    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(I2C=lambda: bus))

    monitor = DistanceMonitor(update_interval=0.0)
    reading = monitor.refresh()

    assert reading.available is True

    monitor.close()

    sensor = created["sensor"]
    assert isinstance(sensor, _StubVL53)
    assert sensor.deinit_called is True
    assert isinstance(created["bus"], _StubBus)
    assert created["bus"].deinit_called is True
    assert monitor.last_error is None

