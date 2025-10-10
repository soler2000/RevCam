"""Driver for interfacing with the GY-85 IMU."""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from threading import Lock

from .sensor_fusion import SensorSample, Vector3


class Gy85Error(RuntimeError):
    """Base exception raised for GY-85 failures."""


class Gy85UnavailableError(Gy85Error):
    """Raised when the sensor or required drivers are unavailable."""


@dataclass(frozen=True, slots=True)
class Gy85Reading:
    """Container for one sensor sample and associated timing."""

    sample: SensorSample
    timestamp: float
    interval_s: float | None


class Gy85Sensor:
    """Minimal driver for the GY-85 IMU module.

    The board combines an ADXL345 accelerometer, ITG3200 gyroscope and
    HMC5883L magnetometer. This class configures each sensor using I²C and
    exposes a convenience method that returns the readings as :class:`SensorSample`.
    """

    _ADXL345_ADDRESS = 0x53
    _ITG3200_ADDRESS = 0x68
    _HMC5883L_ADDRESS = 0x2C

    _ADXL345_POWER_CTL = 0x2D
    _ADXL345_DATA_FORMAT = 0x31
    _ADXL345_BW_RATE = 0x2C
    _ADXL345_DATAX0 = 0x32

    _ITG3200_DLPF_FS = 0x16
    _ITG3200_POWER_MGMT = 0x3E
    _ITG3200_SMPLRT_DIV = 0x15
    _ITG3200_DATA_START = 0x1D

    _HMC5883L_CONFIG_A = 0x00
    _HMC5883L_CONFIG_B = 0x01
    _HMC5883L_MODE = 0x02
    _HMC5883L_DATA_START = 0x03

    _ADXL345_SCALE_G = 0.0039  # g per LSB in full resolution mode
    _ITG3200_SCALE_DEG_PER_S = 1.0 / 14.375
    _HMC5883L_SCALE_UT = 0.092  # microtesla per LSB (0.92 mG)

    def __init__(self, *, i2c_bus: int | None = None) -> None:
        self._lock = Lock()
        self._bus = self._create_bus(i2c_bus)
        self._owns_bus = True
        self._i2c_devices = self._create_devices()
        self._last_timestamp: float = 0.0
        self._initialise_sensors()

    def close(self) -> None:
        """Release any owned I²C resources."""

        bus = getattr(self, "_bus", None)
        if bus is None:
            return
        if getattr(self, "_owns_bus", False):
            try:
                deinit = getattr(bus, "deinit", None)
                if callable(deinit):
                    deinit()
            finally:
                self._owns_bus = False
        self._bus = None
        self._i2c_devices = None

    def _create_bus(self, bus_number: int | None):
        if bus_number is not None:
            try:
                from adafruit_extended_bus import ExtendedI2C  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise Gy85UnavailableError(
                    "install adafruit-circuitpython-extended-bus for custom I2C buses"
                ) from exc
            except Exception as exc:  # pragma: no cover - optional dependency
                raise Gy85UnavailableError(f"ExtendedI2C unavailable: {exc}") from exc

            try:
                return ExtendedI2C(bus_number)  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - hardware specific
                raise Gy85UnavailableError(
                    f"Unable to access I2C bus {bus_number}: {exc}"
                ) from exc

        try:
            import board  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise Gy85UnavailableError("install adafruit-blinka to access board.I2C") from exc
        except Exception as exc:  # pragma: no cover - environment specific
            raise Gy85UnavailableError(f"board.I2C unavailable: {exc}") from exc

        try:
            return board.I2C()  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - hardware specific
            raise Gy85UnavailableError(f"Unable to access default I2C bus: {exc}") from exc

    def _create_devices(self):
        try:
            from adafruit_bus_device.i2c_device import I2CDevice  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise Gy85UnavailableError(
                "install adafruit-circuitpython-busdevice to access the GY-85"
            ) from exc
        except Exception as exc:  # pragma: no cover - environment specific
            raise Gy85UnavailableError(f"Unable to initialise I2C devices: {exc}") from exc

        try:
            accel = I2CDevice(self._bus, self._ADXL345_ADDRESS)
            gyro = I2CDevice(self._bus, self._ITG3200_ADDRESS)
            mag = I2CDevice(self._bus, self._HMC5883L_ADDRESS)
        except ValueError as exc:
            raise Gy85UnavailableError(str(exc)) from exc
        except OSError as exc:  # pragma: no cover - hardware specific
            raise Gy85UnavailableError(
                f"Unable to communicate with GY-85: {exc}"
            ) from exc
        return {
            "accel": accel,
            "gyro": gyro,
            "mag": mag,
        }

    def _initialise_sensors(self) -> None:
        accel = self._i2c_devices["accel"]
        gyro = self._i2c_devices["gyro"]
        mag = self._i2c_devices["mag"]

        self._write_register(accel, self._ADXL345_POWER_CTL, 0x08)
        self._write_register(accel, self._ADXL345_DATA_FORMAT, 0x08)
        self._write_register(accel, self._ADXL345_BW_RATE, 0x0A)

        time.sleep(0.01)

        self._write_register(gyro, self._ITG3200_POWER_MGMT, 0x00)
        time.sleep(0.01)
        self._write_register(gyro, self._ITG3200_SMPLRT_DIV, 0x07)
        self._write_register(gyro, self._ITG3200_DLPF_FS, 0x1A)

        time.sleep(0.01)

        self._write_register(mag, self._HMC5883L_CONFIG_A, 0x70)
        self._write_register(mag, self._HMC5883L_CONFIG_B, 0x20)
        self._write_register(mag, self._HMC5883L_MODE, 0x00)
        time.sleep(0.01)

    def _write_register(self, device, register: int, value: int) -> None:
        try:
            with device as dev:
                dev.write(bytes((register & 0xFF, value & 0xFF)))
        except Exception as exc:  # pragma: no cover - hardware specific
            raise Gy85Error(f"Failed to write register 0x{register:02X}: {exc}") from exc

    def _read_registers(self, device, register: int, length: int) -> bytes:
        buffer = bytearray(length)
        try:
            with device as dev:
                dev.write(bytes((register & 0xFF,)), stop=False)
                dev.readinto(buffer)
        except Exception as exc:  # pragma: no cover - hardware specific
            raise Gy85Error(f"Failed to read register 0x{register:02X}: {exc}") from exc
        return bytes(buffer)

    def _read_accelerometer(self) -> Vector3:
        raw = self._read_registers(self._i2c_devices["accel"], self._ADXL345_DATAX0, 6)
        x, y, z = struct.unpack_from("<hhh", raw)
        return Vector3(
            x=self._ADXL345_SCALE_G * x,
            y=self._ADXL345_SCALE_G * y,
            z=self._ADXL345_SCALE_G * z,
        )

    def _read_gyroscope(self) -> Vector3:
        raw = self._read_registers(self._i2c_devices["gyro"], self._ITG3200_DATA_START, 8)
        # First two bytes are temperature; skip them
        x, y, z = struct.unpack_from(">hhh", raw, offset=2)
        return Vector3(
            x=self._ITG3200_SCALE_DEG_PER_S * x,
            y=self._ITG3200_SCALE_DEG_PER_S * y,
            z=self._ITG3200_SCALE_DEG_PER_S * z,
        )

    def _read_magnetometer(self) -> Vector3:
        raw = self._read_registers(self._i2c_devices["mag"], self._HMC5883L_DATA_START, 6)
        x, z, y = struct.unpack_from(">hhh", raw)
        return Vector3(
            x=self._HMC5883L_SCALE_UT * x,
            y=self._HMC5883L_SCALE_UT * y,
            z=self._HMC5883L_SCALE_UT * z,
        )

    def read(self) -> Gy85Reading:
        """Return the latest sensor measurements."""

        with self._lock:
            accel = self._read_accelerometer()
            gyro = self._read_gyroscope()
            mag = self._read_magnetometer()
            timestamp = time.monotonic()
            interval: float | None
            if self._last_timestamp:
                interval = timestamp - self._last_timestamp
                if interval <= 0 or not math.isfinite(interval):
                    interval = None
            else:
                interval = None
            self._last_timestamp = timestamp
            sample = SensorSample(accelerometer=accel, gyroscope=gyro, magnetometer=mag)
            return Gy85Reading(sample=sample, timestamp=timestamp, interval_s=interval)

    def __enter__(self) -> "Gy85Sensor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
