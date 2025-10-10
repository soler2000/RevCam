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
    """Fusion filter combining gyro and accelerometer data for orientation."""

    def __init__(self) -> None:
        self._roll = _AxisKalmanFilter()
        self._pitch = _AxisKalmanFilter()
        self._smoothed_orientation = OrientationAngles(0.0, 0.0)
        self._smoothing_alpha = 0.2

    def reset(self) -> None:
        self._roll.reset()
        self._pitch.reset()
        self._smoothed_orientation = OrientationAngles(0.0, 0.0)

    def update(self, sample: SensorSample, dt: Optional[float] = None) -> OrientationAngles:
        """Update the filter with a new sensor sample."""

        if dt is None or not math.isfinite(dt) or dt <= 0:
            dt = 0.02
        ax_raw, ay_raw, az = sample.accelerometer.x, sample.accelerometer.y, sample.accelerometer.z
        gx_raw, gy_raw = sample.gyroscope.x, sample.gyroscope.y

        # The IMU is mounted rotated 90° around the Z axis relative to the trailer,
        # so the board's X axis aligns with the trailer's lateral axis (roll) and
        # the board's Y axis aligns with the trailer's longitudinal axis (pitch).
        # Remap the axes so downstream calculations operate in trailer space.
        ax = ay_raw
        ay = ax_raw
        gx = gy_raw
        gy = gx_raw
        # Accelerometer-based roll and pitch (degrees)
        roll_measure = math.degrees(math.atan2(ay, az)) if ay or az else 0.0
        denominator = math.sqrt(ay * ay + az * az)
        pitch_measure = math.degrees(math.atan2(-ax, denominator)) if denominator else 0.0

        roll = self._roll.update(gx, roll_measure, dt)
        pitch = self._pitch.update(gy, pitch_measure, dt)

        raw_orientation = OrientationAngles(roll=roll, pitch=pitch).normalised()
        smoothed = OrientationAngles(
            roll=_smooth_angle(self._smoothed_orientation.roll, raw_orientation.roll, self._smoothing_alpha),
            pitch=_smooth_angle(
                self._smoothed_orientation.pitch, raw_orientation.pitch, self._smoothing_alpha
            ),
        ).normalised()
        self._smoothed_orientation = smoothed
        return smoothed


def _smooth_angle(previous: float, current: float, alpha: float) -> float:
    """Return an exponentially smoothed angle while respecting wrap-around."""

    previous = float(previous)
    current = float(current)
    alpha = float(alpha)
    if alpha <= 0:
        return previous
    if alpha >= 1:
        return current
    delta = ((current - previous + 180.0) % 360.0) - 180.0
    return previous + alpha * delta

