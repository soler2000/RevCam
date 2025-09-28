import pytest

pytest.importorskip("numpy")

import numpy as np

from rev_cam.config import DEFAULT_REVERSING_AIDS, ReversingAidsConfig
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


def test_reversing_aids_overlay_refresh_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    import rev_cam.reversing_aids as reversing_aids

    refreshes = 0

    def _fake_render(frame: np.ndarray, config: ReversingAidsConfig) -> np.ndarray:
        nonlocal refreshes
        refreshes += 1
        frame[...] = 255
        return frame

    monkeypatch.setattr(reversing_aids, "_render_reversing_aids", _fake_render)

    config = ReversingAidsConfig()
    overlay = create_reversing_aids_overlay(lambda: config)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    for _ in range(25):
        overlay(frame.copy())

    assert refreshes == 3

