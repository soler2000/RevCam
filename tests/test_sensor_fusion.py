import math

import pytest

from rev_cam.sensor_fusion import Gy85KalmanFilter, SensorSample, Vector3, _smooth_angle


@pytest.mark.parametrize(
    "previous,current,alpha,expected",
    [
        (0.0, 10.0, 0.5, 5.0),
        (45.0, 45.0, 0.5, 45.0),
        (30.0, 60.0, 1.0, 60.0),
        (15.0, -15.0, 0.25, 7.5),
        (170.0, -170.0, 0.5, 180.0),
    ],
)
def test_smooth_angle(previous, current, alpha, expected):
    result = _smooth_angle(previous, current, alpha)
    assert math.isfinite(result)
    assert result == pytest.approx(expected, abs=1e-6)


def test_smooth_angle_alpha_bounds():
    assert _smooth_angle(10.0, 20.0, 0.0) == 10.0
    assert _smooth_angle(10.0, 20.0, 1.5) == 20.0


def _make_static_sample(roll_deg: float, pitch_deg: float) -> SensorSample:
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    trailer_ax = -math.sin(pitch_rad)
    trailer_ay = math.sin(roll_rad) * math.cos(pitch_rad)
    trailer_az = math.cos(roll_rad) * math.cos(pitch_rad)
    return SensorSample(
        accelerometer=Vector3(
            x=trailer_ay,
            y=trailer_az,
            z=trailer_ax,
        ),
        gyroscope=Vector3(x=0.0, y=0.0, z=0.0),
        magnetometer=Vector3(x=0.0, y=0.0, z=0.0),
    )


def _make_dynamic_sample(
    roll_deg: float,
    pitch_deg: float,
    roll_rate: float,
    pitch_rate: float,
    yaw_rate: float = 0.0,
) -> SensorSample:
    base = _make_static_sample(roll_deg, pitch_deg)
    board_gx = pitch_rate  # IMU X axis tracks trailer pitch rate.
    board_gy = yaw_rate
    board_gz = roll_rate   # IMU Z axis tracks trailer roll rate.
    return SensorSample(
        accelerometer=base.accelerometer,
        gyroscope=Vector3(x=board_gx, y=board_gy, z=board_gz),
        magnetometer=base.magnetometer,
    )


@pytest.mark.parametrize(
    "roll_deg,pitch_deg",
    [
        (8.0, 0.0),
        (0.0, -5.0),
        (6.0, -4.0),
    ],
)
def test_kalman_filter_matches_trailer_axes(roll_deg, pitch_deg):
    filter = Gy85KalmanFilter()
    sample = _make_static_sample(roll_deg, pitch_deg)
    orientation = None
    for _ in range(400):
        orientation = filter.update(sample, dt=0.02)
    assert orientation is not None
    assert orientation.roll == pytest.approx(roll_deg, abs=0.6)
    assert orientation.pitch == pytest.approx(pitch_deg, abs=0.6)


def test_pitch_rate_changes_do_not_bias_roll():
    filter = Gy85KalmanFilter()
    level_sample = _make_dynamic_sample(0.0, 0.0, roll_rate=0.0, pitch_rate=0.0)
    for _ in range(10):
        filter.update(level_sample, dt=0.02)

    pitch_rate = 15.0
    orientation = None
    for _ in range(200):
        sample = _make_dynamic_sample(0.0, 0.0, roll_rate=0.0, pitch_rate=pitch_rate)
        orientation = filter.update(sample, dt=0.02)

    assert orientation is not None
    assert orientation.roll == pytest.approx(0.0, abs=0.2)
