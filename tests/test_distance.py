"""Tests for the VL53L1X distance helpers."""

from __future__ import annotations

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
    monitor = DistanceMonitor(sensor_factory=lambda: sensor, update_interval=0.0)

    reading = monitor.read()

    assert reading.available is True
    assert reading.distance_m == pytest.approx(1.23, rel=1e-6)
    assert reading.raw_distance_m == pytest.approx(1.23, rel=1e-6)
    assert reading.error is None


def test_distance_monitor_filters_invalid_samples() -> None:
    sensor = _SequenceSensor([123.0, 9999.0, 140.0])
    monitor = DistanceMonitor(sensor_factory=lambda: sensor, update_interval=0.0)

    first = monitor.read()
    second = monitor.read()
    third = monitor.read()

    assert second.available is True
    assert second.distance_m == pytest.approx(first.distance_m)
    assert second.error is not None
    assert third.available is True
    assert third.distance_m > first.distance_m
    assert third.error is None


def test_distance_monitor_handles_measurements_in_metres() -> None:
    sensor = _SequenceSensor([1.8, 1.75])
    monitor = DistanceMonitor(sensor_factory=lambda: sensor, update_interval=0.0)

    first = monitor.read()
    second = monitor.read()

    assert first.distance_m == pytest.approx(1.8, rel=1e-6)
    assert first.raw_distance_m == pytest.approx(1.8, rel=1e-6)
    assert second.raw_distance_m == pytest.approx(1.75, rel=1e-6)
    assert second.distance_m == pytest.approx(1.785, abs=1e-6)


def test_distance_monitor_recovers_from_legitimate_jump() -> None:
    sensor = _SequenceSensor([20.0, 500.0, 500.0])
    monitor = DistanceMonitor(sensor_factory=lambda: sensor, update_interval=0.0)

    first = monitor.read()
    second = monitor.read()
    third = monitor.read()

    assert first.distance_m == pytest.approx(0.2, rel=1e-6)
    assert second.distance_m == pytest.approx(first.distance_m)
    assert second.error == "Filtered invalid distance sample"
    assert third.distance_m == pytest.approx(5.0, rel=1e-6)
    assert third.error is None


def test_distance_monitor_reports_unavailable_when_sensor_missing() -> None:
    def _failing_factory() -> object:
        raise RuntimeError("sensor offline")

    monitor = DistanceMonitor(sensor_factory=_failing_factory, update_interval=0.0)

    reading = monitor.read()

    assert reading.available is False
    assert reading.distance_m is None
    assert reading.error == "sensor offline"
    assert monitor.last_error == "sensor offline"


def test_distance_monitor_applies_calibration() -> None:
    sensor = _SequenceSensor([100.0])
    calibration = DistanceCalibration(offset_m=-0.1, scale=0.01)
    monitor = DistanceMonitor(
        sensor_factory=lambda: sensor,
        calibration=calibration,
        update_interval=0.0,
    )

    reading = monitor.read()

    assert reading.raw_distance_m == pytest.approx(1.0, rel=1e-6)
    assert reading.distance_m == pytest.approx(0.0, abs=1e-6)


def test_distance_monitor_set_calibration_updates_immediately() -> None:
    sensor = _SequenceSensor([100.0, 100.0])
    monitor = DistanceMonitor(sensor_factory=lambda: sensor, update_interval=5.0)

    first = monitor.read()
    assert first.distance_m == pytest.approx(1.0, rel=1e-6)

    updated = monitor.set_calibration(offset_m=-0.25)
    assert updated.offset_m == pytest.approx(-0.25, rel=1e-6)

    second = monitor.read()
    assert second.raw_distance_m == pytest.approx(1.0, rel=1e-6)
    assert second.distance_m == pytest.approx(0.75, rel=1e-6)


def test_distance_overlay_draws_on_frame() -> None:
    sensor = _SequenceSensor([150.0])
    monitor = DistanceMonitor(sensor_factory=lambda: sensor, update_interval=0.0)
    zones = DistanceZones(caution=2.0, warning=1.0, danger=0.5)
    overlay = create_distance_overlay(monitor, lambda: zones)

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = overlay(frame)

    assert result is frame
    assert np.any(result != 0)


def test_distance_overlay_handles_non_numpy_frames() -> None:
    sensor = _SequenceSensor([150.0])
    monitor = DistanceMonitor(sensor_factory=lambda: sensor, update_interval=0.0)
    overlay = create_distance_overlay(monitor, lambda: DistanceZones(2.0, 1.0, 0.5))

    frame = [[0, 0, 0], [0, 0, 0]]
    result = overlay(frame)

    assert result == frame


def test_distance_monitor_close_releases_sensor_and_i2c() -> None:
    class _StubI2CBus:
        def __init__(self) -> None:
            self.deinit_called = False

        def deinit(self) -> None:
            self.deinit_called = True

    class _StubDevice:
        def __init__(self, bus: _StubI2CBus) -> None:
            self.i2c = bus
            self.deinit_called = False

        def deinit(self) -> None:
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
    monitor = DistanceMonitor(sensor_factory=lambda: sensor, update_interval=0.0)
    monitor.read()

    monitor.close()

    assert sensor.deinit_called is True
    assert device.deinit_called is True
    assert bus.deinit_called is True
    assert monitor.last_error is None

