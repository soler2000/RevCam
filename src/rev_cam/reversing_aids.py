"""Overlay helpers for rendering configurable reversing aids."""

from __future__ import annotations

import logging
from typing import Callable

try:  # pragma: no cover - optional dependency on numpy for overlays
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None

try:  # pragma: no cover - optional dependency on OpenCV for overlays
    import cv2 as _cv2
except ImportError:  # pragma: no cover - optional dependency
    _cv2 = None

from .config import ReversingAidsConfig, ReversingAidSegment

logger = logging.getLogger(__name__)

OverlayFn = Callable[[object], object]

_SEGMENT_COLOURS: tuple[tuple[int, int, int], ...] = (
    (102, 187, 106),  # green
    (255, 193, 7),  # amber
    (239, 83, 80),  # red
)


def create_reversing_aids_overlay(
    config_provider: Callable[[], ReversingAidsConfig],
    *,
    enabled_provider: Callable[[], bool] | None = None,
) -> OverlayFn:
    """Return an overlay function that renders reversing aid guides."""

    def _overlay(frame: object) -> object:
        if _np is None or not isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return frame

        if enabled_provider is not None:
            try:
                if not enabled_provider():
                    return frame
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Reversing overlay enabled provider failed", exc_info=True)

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

    overlay = _np.zeros_like(frame)
    mask = _np.zeros(frame.shape[:2], dtype=_np.uint8)

    for index, segment in enumerate(config.left):
        colour = colours[min(index, len(colours) - 1)]
        _draw_segment(overlay, mask, segment, width, height, thickness, colour)

    for index, segment in enumerate(config.right):
        colour = colours[min(index, len(colours) - 1)]
        _draw_segment(overlay, mask, segment, width, height, thickness, colour)

    filled_y, filled_x = mask.nonzero()
    if not filled_y.size:
        return frame

    min_y = int(filled_y.min())
    max_y = int(filled_y.max())
    min_x = int(filled_x.min())
    max_x = int(filled_x.max())

    frame_region = frame[min_y : max_y + 1, min_x : max_x + 1]
    overlay_region = overlay[min_y : max_y + 1, min_x : max_x + 1]
    mask_region = mask[min_y : max_y + 1, min_x : max_x + 1]

    coverage = mask_region > 0
    if not _np.any(coverage):
        return frame

    if _np.all(mask_region[coverage] >= 255):
        frame_region[coverage] = overlay_region[coverage]
        return frame

    alpha = (mask_region.astype(_np.float32) / 255.0)[..., None]
    overlay_float = overlay_region.astype(_np.float32)
    frame_float = frame_region.astype(_np.float32)
    blended = overlay_float * alpha + frame_float * (1.0 - alpha)

    if _np.issubdtype(frame.dtype, _np.integer):
        info = _np.iinfo(frame.dtype)
        blended = _np.clip(_np.rint(blended), info.min, info.max)
    else:
        info = _np.finfo(frame.dtype)
        blended = _np.clip(blended, info.min, info.max)

    _np.copyto(frame_region, blended.astype(frame.dtype, copy=False), where=coverage[..., None])

    return frame


def _draw_segment(
    overlay: _np.ndarray,
    mask: _np.ndarray,
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

    _draw_line(overlay, mask, start_x, start_y, end_x, end_y, thickness, colour)


def _draw_line(
    overlay: _np.ndarray,
    mask: _np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    thickness: int,
    colour: tuple[int, int, int],
) -> None:
    height, width = overlay.shape[:2]
    if height == 0 or width == 0:
        return

    if _cv2 is not None:  # pragma: no cover - exercised only when OpenCV is installed
        _cv2.line(overlay, (x0, y0), (x1, y1), colour, thickness=thickness, lineType=_cv2.LINE_AA)
        _cv2.line(mask, (x0, y0), (x1, y1), 255, thickness=thickness, lineType=_cv2.LINE_AA)
        return

    half_thickness = max(thickness / 2.0, 0.5)
    padding = int(_np.ceil(half_thickness)) + 1

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

    region_mask = dist_sq <= half_thickness**2
    if not _np.any(region_mask):
        return

    overlay_region = overlay[min_y : max_y + 1, min_x : max_x + 1]
    overlay_region[region_mask] = colour
    mask_region = mask[min_y : max_y + 1, min_x : max_x + 1]
    mask_region[region_mask] = 255


__all__ = ["create_reversing_aids_overlay"]

