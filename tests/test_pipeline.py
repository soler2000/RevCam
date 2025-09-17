from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline


def make_frame():
    return [
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
    ]


def orientation_provider(rotation=0, flip_h=False, flip_v=False):
    return Orientation(rotation=rotation, flip_horizontal=flip_h, flip_vertical=flip_v)


def test_rotation_90_degrees():
    frame = make_frame()
    pipeline = FramePipeline(lambda: orientation_provider(rotation=90))
    processed = pipeline.process(frame)
    expected = [
        [[2, 2, 2], [5, 5, 5]],
        [[1, 1, 1], [4, 4, 4]],
        [[0, 0, 0], [3, 3, 3]],
    ]
    assert processed == expected


def test_horizontal_flip():
    frame = make_frame()
    pipeline = FramePipeline(lambda: orientation_provider(flip_h=True))
    processed = pipeline.process(frame)
    expected = [
        [[2, 2, 2], [1, 1, 1], [0, 0, 0]],
        [[5, 5, 5], [4, 4, 4], [3, 3, 3]],
    ]
    assert processed == expected


def test_overlay_composition():
    frame = make_frame()
    pipeline = FramePipeline(lambda: orientation_provider())

    def increment(data):
        return [
            [[min(channel + 1, 255) for channel in pixel] for pixel in row]
            for row in data
        ]

    pipeline.add_overlay(increment)
    processed = pipeline.process(frame)
    assert processed == increment(frame)
