"""Sensor fusion helpers for the GY-85 IMU."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .trailer_leveling import OrientationAngles


@dataclass(frozen=True, slots=True)
class Vector3:
    """Simple 3D vector container for sensor measurements."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class SensorSample:
    """Represents one reading from the GY-85 sensors."""

    accelerometer: Vector3
    gyroscope: Vector3
    magnetometer: Vector3


class _AxisKalmanFilter:
    """One-dimensional Kalman filter used for angle estimation."""

    def __init__(self, q_angle: float = 0.001, q_bias: float = 0.003, r_measure: float = 0.03) -> None:
        self.q_angle = float(q_angle)
        self.q_bias = float(q_bias)
        self.r_measure = float(r_measure)
        self._angle = 0.0
        self._bias = 0.0
        self._rate = 0.0
        self._p00 = 0.0
        self._p01 = 0.0
        self._p10 = 0.0
        self._p11 = 0.0

    def update(self, rate: float, measurement: float, dt: float) -> float:
        dt = float(dt)
        rate = float(rate)
        measurement = float(measurement)
        self._rate = rate - self._bias
        self._angle += dt * self._rate

        # Update covariance matrix
        self._p00 += dt * (dt * self._p11 - self._p01 - self._p10 + self.q_angle)
        self._p01 -= dt * self._p11
        self._p10 -= dt * self._p11
        self._p11 += self.q_bias * dt

        # Kalman gain
        s = self._p00 + self.r_measure
        if s == 0:
            return self._angle
        k0 = self._p00 / s
        k1 = self._p10 / s

        # Update estimate with measurement
        y = measurement - self._angle
        self._angle += k0 * y
        self._bias += k1 * y

        # Update error covariance
        p00_temp = self._p00
        p01_temp = self._p01
        self._p00 -= k0 * p00_temp
        self._p01 -= k0 * p01_temp
        self._p10 -= k1 * p00_temp
        self._p11 -= k1 * p01_temp
        return self._angle

    def reset(self, angle: float = 0.0) -> None:
        self._angle = float(angle)
        self._bias = 0.0
        self._rate = 0.0
        self._p00 = self._p01 = self._p10 = self._p11 = 0.0


class Gy85KalmanFilter:
    """Fusion filter combining gyro, accelerometer and magnetometer data."""

    def __init__(self) -> None:
        self._roll = _AxisKalmanFilter()
        self._pitch = _AxisKalmanFilter()
        self._yaw = _AxisKalmanFilter(r_measure=0.5)
        self._last_orientation = OrientationAngles(0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._roll.reset()
        self._pitch.reset()
        self._yaw.reset()
        self._last_orientation = OrientationAngles(0.0, 0.0, 0.0)

    def update(self, sample: SensorSample, dt: Optional[float] = None) -> OrientationAngles:
        """Update the filter with a new sensor sample."""

        if dt is None or not math.isfinite(dt) or dt <= 0:
            dt = 0.02
        ax, ay, az = sample.accelerometer.x, sample.accelerometer.y, sample.accelerometer.z
        gx, gy, gz = sample.gyroscope.x, sample.gyroscope.y, sample.gyroscope.z
        mx, my, mz = sample.magnetometer.x, sample.magnetometer.y, sample.magnetometer.z

        # Accelerometer-based roll and pitch (degrees)
        roll_measure = math.degrees(math.atan2(az, ay)) if ay or az else 0.0
        denominator = math.sqrt(ay * ay + az * az)
        pitch_measure = math.degrees(math.atan2(-ax, denominator)) if denominator else 0.0

        roll = self._roll.update(gx, roll_measure, dt)
        pitch = self._pitch.update(gy, pitch_measure, dt)

        # Tilt compensation for magnetometer to derive yaw
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        sin_roll = math.sin(roll_rad)
        cos_roll = math.cos(roll_rad)
        sin_pitch = math.sin(pitch_rad)
        cos_pitch = math.cos(pitch_rad)
        compensated_x = mx * cos_pitch + mz * sin_pitch
        compensated_y = mx * sin_roll * sin_pitch + my * cos_roll - mz * sin_roll * cos_pitch
        yaw_measure = math.degrees(math.atan2(-compensated_y, compensated_x)) if compensated_x or compensated_y else self._last_orientation.yaw

        yaw = self._yaw.update(gz, yaw_measure, dt)
        orientation = OrientationAngles(roll=roll, pitch=pitch, yaw=yaw).normalised()
        self._last_orientation = orientation
        return orientation

