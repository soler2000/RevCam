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

    cached_overlay: _np.ndarray | None = None
    cached_mask: _np.ndarray | None = None
    cached_shape: tuple[int, ...] | None = None
    cached_config: ReversingAidsConfig | None = None
    frames_since_refresh = 0

    def _overlay(frame: object) -> object:
        nonlocal cached_overlay, cached_mask, cached_shape, cached_config, frames_since_refresh

        if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return frame

        config = config_provider()
        if not config.enabled:
            cached_overlay = None
            cached_mask = None
            cached_shape = None
            cached_config = None
            frames_since_refresh = 0
            return frame

        needs_refresh = False
        if cached_overlay is None or cached_mask is None:
            needs_refresh = True
        elif frame.shape != cached_shape:
            needs_refresh = True
        elif cached_config != config:
            needs_refresh = True
        elif frames_since_refresh >= 10:
            needs_refresh = True

        if needs_refresh:
            overlay = _np.zeros_like(frame)
            overlay = _render_reversing_aids(overlay, config)
            mask = _np.any(overlay != 0, axis=2) if overlay.ndim == 3 else overlay != 0

            cached_overlay = overlay
            cached_mask = mask
            cached_shape = frame.shape
            cached_config = config
            frames_since_refresh = 0
        else:
            frames_since_refresh += 1

        if cached_overlay is None or cached_mask is None:
            return frame

        frame[cached_mask] = cached_overlay[cached_mask]
        return frame

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
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    radius = max(0, thickness // 2)

    while True:
        _stamp_disc(frame, x0, y0, radius, colour)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _stamp_disc(
    frame: _np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    colour: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    for y in range(cy - radius, cy + radius + 1):
        if y < 0 or y >= height:
            continue
        for x in range(cx - radius, cx + radius + 1):
            if x < 0 or x >= width:
                continue
            if radius == 0 or (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                frame[y, x] = colour


__all__ = ["create_reversing_aids_overlay"]

