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
    board_ax = trailer_ay
    board_ay = trailer_ax
    board_az = trailer_az
    return SensorSample(
        accelerometer=Vector3(x=board_ax, y=board_ay, z=board_az),
        gyroscope=Vector3(x=0.0, y=0.0, z=0.0),
        magnetometer=Vector3(x=0.0, y=0.0, z=0.0),
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
