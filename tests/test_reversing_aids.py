import pytest

pytest.importorskip("numpy")

import numpy as np

from rev_cam.config import (
    DEFAULT_REVERSING_AIDS,
    ReversingAidPoint,
    ReversingAidSegment,
    ReversingAidsConfig,
)
from rev_cam import reversing_aids as reversing_aids_mod
from rev_cam.reversing_aids import create_reversing_aids_overlay


def test_reversing_aids_overlay_draws_segments() -> None:
    overlay = create_reversing_aids_overlay(lambda: DEFAULT_REVERSING_AIDS)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    result = overlay(frame)

    assert result is frame
    assert np.any(result != 0)


def test_reversing_aids_overlay_disabled() -> None:
    config = ReversingAidsConfig(enabled=False)
    overlay = create_reversing_aids_overlay(lambda: config)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = overlay(frame)

    assert not np.any(result)


def test_reversing_aids_overlay_handles_non_numpy() -> None:
    overlay = create_reversing_aids_overlay(lambda: DEFAULT_REVERSING_AIDS)
    frame = [[0, 0, 0], [0, 0, 0]]
    assert overlay(frame) == frame


def test_render_reversing_aids_alpha_blending(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ReversingAidsConfig(
        left=(
            ReversingAidSegment(
                start=ReversingAidPoint(0.0, 0.0),
                end=ReversingAidPoint(1.0, 0.0),
            ),
        ),
        right=tuple(),
    )

    def _fake_draw_segment(
        overlay: np.ndarray,
        mask: np.ndarray,
        segment: ReversingAidSegment,
        width: int,
        height: int,
        thickness: int,
        colour: tuple[int, int, int],
    ) -> None:
        overlay[0, 0] = colour
        mask[0, 0] = 255
        overlay[0, 1] = colour
        mask[0, 1] = 128

    monkeypatch.setattr(reversing_aids_mod, "_draw_segment", _fake_draw_segment)

    frame = np.full((12, 12, 3), 100, dtype=np.uint8)
    result = reversing_aids_mod._render_reversing_aids(frame, config)

    assert result is frame
    expected_colour = np.array(reversing_aids_mod._SEGMENT_COLOURS[0], dtype=np.uint8)
    assert np.array_equal(result[0, 0], expected_colour)
    blended = result[0, 1]
    assert np.all(blended > 100)
    assert np.all(blended < expected_colour)

