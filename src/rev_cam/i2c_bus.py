"""Helpers for creating I2C bus instances without external helpers."""

from __future__ import annotations

from dataclasses import dataclass


class I2CBusError(RuntimeError):
    """Base exception raised when an I²C bus cannot be created."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class I2CBusDependencyError(I2CBusError):
    """Raised when optional dependencies for I²C access are missing."""


class I2CBusRuntimeError(I2CBusError):
    """Raised when the underlying system rejects the requested bus."""


@dataclass(slots=True)
class _FallbackDependencyError(Exception):
    message: str


@dataclass(slots=True)
class _FallbackRuntimeError(Exception):
    message: str


def create_i2c_bus(bus_number: int, /, *, frequency: int | None = None):
    """Create an I²C bus that is compatible with Adafruit drivers.

    The upstream project recommends installing ``adafruit-circuitpython-extended-bus``
    when a non-default bus number is required. Unfortunately that package is no
    longer published on PyPI which breaks automated installations.  This helper
    mirrors the behaviour that RevCam relies on, first attempting to import the
    official helper and falling back to a minimal local implementation when that
    fails.  The returned object provides the subset of the ``busio.I2C`` API that
    the Adafruit drivers expect.
    """

    try:
        from adafruit_extended_bus import ExtendedI2C as _ExtendedI2C  # type: ignore
    except ModuleNotFoundError:
        try:
            return _create_fallback_bus(bus_number, frequency=frequency)
        except _FallbackDependencyError as exc:  # pragma: no cover - import guard
            raise I2CBusDependencyError(exc.message) from None
        except _FallbackRuntimeError as exc:  # pragma: no cover - hardware guard
            raise I2CBusRuntimeError(exc.message) from None
    except Exception as exc:  # pragma: no cover - optional dependency
        raise I2CBusRuntimeError(f"ExtendedI2C unavailable: {exc}") from exc

    try:
        if frequency is None:
            return _ExtendedI2C(bus_number)  # type: ignore[call-arg]
        return _ExtendedI2C(bus_number, frequency=frequency)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - hardware specific
        raise I2CBusRuntimeError(f"Unable to access I2C bus {bus_number}: {exc}") from exc


def _create_fallback_bus(bus_number: int, /, *, frequency: int | None):
    try:
        from busio import I2C as _BusI2C  # type: ignore
        import board  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise _FallbackDependencyError(
            "install adafruit-blinka to access board.I2C"
        ) from exc

    scl = _lookup_pin(board, "SCL", bus_number)
    sda = _lookup_pin(board, "SDA", bus_number)

    try:
        if frequency is None:
            bus = _BusI2C(scl, sda)
        else:
            bus = _BusI2C(scl, sda, frequency=frequency)
    except ValueError as exc:  # pragma: no cover - hardware specific
        raise _FallbackRuntimeError(
            f"Unable to access I2C bus {bus_number}: {exc}"
        ) from exc
    except OSError as exc:  # pragma: no cover - hardware specific
        raise _FallbackRuntimeError(
            f"Unable to access I2C bus {bus_number}: {exc}"
        ) from exc

    return _FallbackExtendedI2C(bus)


def _lookup_pin(module, prefix: str, bus_number: int):
    """Best-effort lookup for the requested I²C pin."""

    candidates: tuple[str, ...]
    if bus_number == 0:
        candidates = (f"{prefix}_0", prefix)
    elif bus_number == 1:
        candidates = (f"{prefix}_1", prefix)
    else:
        candidates = (f"{prefix}_{bus_number}", f"{prefix}{bus_number}")

    for name in candidates:
        try:
            return getattr(module, name)
        except AttributeError:
            continue

    raise _FallbackRuntimeError(
        f"Unable to locate board pin mapping for I2C bus {bus_number}"
    )


class _FallbackExtendedI2C:
    """Proxy object that mimics :class:`busio.I2C` for Adafruit drivers."""

    __slots__ = ("_bus",)

    def __init__(self, bus) -> None:
        self._bus = bus

    def __getattr__(self, name: str):  # pragma: no cover - thin wrapper
        return getattr(self._bus, name)

    def deinit(self) -> None:  # pragma: no cover - thin wrapper
        deinit = getattr(self._bus, "deinit", None)
        if callable(deinit):
            deinit()
