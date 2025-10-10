"""Behavioural tests for the GY-85 sensor wrapper."""

from __future__ import annotations

import sys
import types

import pytest

from rev_cam.gy85 import Gy85Sensor, Gy85UnavailableError


class _DummyBus:
    def __init__(self) -> None:
        self.deinitialised = False

    def deinit(self) -> None:
        self.deinitialised = True


@pytest.fixture(autouse=True)
def _patch_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stub ``board`` module that exposes ``I2C``."""

    board_module = types.ModuleType("board")
    bus_holder: dict[str, _DummyBus] = {}

    def _make_bus() -> _DummyBus:
        bus = _DummyBus()
        bus_holder["bus"] = bus
        return bus

    board_module.I2C = _make_bus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "board", board_module)

    yield

    bus = bus_holder.get("bus")
    if bus is not None:
        assert bus.deinitialised, "expected the I2C bus to be released on failure"


def _install_i2c_device(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    """Helper to install an ``adafruit_bus_device.i2c_device`` stub."""

    package = types.ModuleType("adafruit_bus_device")
    submodule = types.ModuleType("adafruit_bus_device.i2c_device")

    class _FailingI2CDevice:  # pragma: no cover - trivial container
        def __init__(self, _bus, address: int) -> None:
            raise type(exc)(f"{exc} at 0x{address:02X}")

    submodule.I2CDevice = _FailingI2CDevice  # type: ignore[attr-defined]
    package.i2c_device = submodule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "adafruit_bus_device", package)
    monkeypatch.setitem(sys.modules, "adafruit_bus_device.i2c_device", submodule)


@pytest.mark.parametrize(
    "raised_exception, expected_message",
    [
        (ValueError("No I2C device"), "not found"),
        (OSError("Input/output error"), "Unable to communicate"),
    ],
)
def test_gy85_reports_missing_devices(monkeypatch: pytest.MonkeyPatch, raised_exception: Exception, expected_message: str) -> None:
    """The driver should surface clear errors when the IMU is absent."""

    _install_i2c_device(monkeypatch, raised_exception)

    with pytest.raises(Gy85UnavailableError) as excinfo:
        Gy85Sensor()

    message = str(excinfo.value)
    assert "0x53" in message
    assert expected_message in message
