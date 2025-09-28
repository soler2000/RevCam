"""Overlay helpers for rendering configurable reversing aids."""

from __future__ import annotations

from typing import Callable

try:  # pragma: no cover - optional dependency on numpy for overlays
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None

from .config import ReversingAidsConfig, ReversingAidSegment

OverlayFn = Callable[[object], object]

_SEGMENT_COLOURS: tuple[tuple[int, int, int], ...] = (
    (102, 187, 106),  # green
    (255, 193, 7),  # amber
    (239, 83, 80),  # red
)


def create_reversing_aids_overlay(
    config_provider: Callable[[], ReversingAidsConfig]
) -> OverlayFn:
    """Return an overlay function that renders reversing aid guides."""

    def _overlay(frame: object) -> object:
        if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return frame

        config = config_provider()
        if not config.enabled:
            return frame
        return _render_reversing_aids(frame, config)

    return _overlay


def _render_reversing_aids(frame: _np.ndarray, config: ReversingAidsConfig) -> _np.ndarray:
    height, width = frame.shape[:2]
    if height < 10 or width < 10:
        return frame

    thickness = max(1, int(round(min(width, height) * 0.01)))
    colours = list(_SEGMENT_COLOURS)

    for index, segment in enumerate(config.left):
        colour = colours[min(index, len(colours) - 1)]
        _draw_segment(frame, segment, width, height, thickness, colour)

    for index, segment in enumerate(config.right):
        colour = colours[min(index, len(colours) - 1)]
        _draw_segment(frame, segment, width, height, thickness, colour)

    return frame


def _draw_segment(
    frame: _np.ndarray,
    segment: ReversingAidSegment,
    width: int,
    height: int,
    thickness: int,
    colour: tuple[int, int, int],
) -> None:
    start_x = int(round(segment.start.x * (width - 1)))
    start_y = int(round(segment.start.y * (height - 1)))
    end_x = int(round(segment.end.x * (width - 1)))
    end_y = int(round(segment.end.y * (height - 1)))

    _draw_line(frame, start_x, start_y, end_x, end_y, thickness, colour)


def _draw_line(
    frame: _np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    thickness: int,
    colour: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return

    base_radius = max(0, thickness // 2)
    effective_radius = max(base_radius, 0.5)
    padding = int(_np.ceil(effective_radius)) + 2

    min_x = max(0, min(x0, x1) - padding)
    max_x = min(width - 1, max(x0, x1) + padding)
    min_y = max(0, min(y0, y1) - padding)
    max_y = min(height - 1, max(y0, y1) + padding)

    if min_x > max_x or min_y > max_y:
        return

    grid_y, grid_x = _np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    dx = x1 - x0
    dy = y1 - y0
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0:
        dist_sq = (grid_x - x0) ** 2 + (grid_y - y0) ** 2
    else:
        t = ((grid_x - x0) * dx + (grid_y - y0) * dy) / seg_len_sq
        t = _np.clip(t, 0.0, 1.0)
        nearest_x = x0 + t * dx
        nearest_y = y0 + t * dy
        dist_sq = (grid_x - nearest_x) ** 2 + (grid_y - nearest_y) ** 2

    mask = dist_sq <= effective_radius**2
    if not _np.any(mask):
        return

    region = frame[min_y : max_y + 1, min_x : max_x + 1]
    region[mask] = colour


__all__ = ["create_reversing_aids_overlay"]

