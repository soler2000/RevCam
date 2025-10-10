import math

import pytest

from rev_cam.sensor_fusion import _smooth_angle


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
