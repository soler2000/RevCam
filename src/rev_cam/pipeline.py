"""Video frame processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List

try:  # pragma: no cover - optional dependency
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None

from .config import Orientation

FrameLike = Any
FrameFn = Callable[[FrameLike], FrameLike]
OverlayFn = FrameFn


def _overlays_enabled_default() -> bool:
    return True


@dataclass
class FramePipeline:
    """Applies orientation and overlay transforms to frames."""

    orientation_provider: Callable[[], Orientation]
    overlays: List[OverlayFn] = field(default_factory=list)
    preprocessors: List[FrameFn] = field(default_factory=list)
    overlay_enabled_provider: Callable[[], bool] = _overlays_enabled_default

    def add_overlay(self, overlay: OverlayFn) -> None:
        self.overlays.append(overlay)

    def add_preprocessor(self, transform: FrameFn) -> None:
        self.preprocessors.append(transform)

    def _apply_orientation(self, frame: FrameLike, orientation: Orientation) -> FrameLike:
        output = frame
        k = (orientation.rotation // 90) % 4
        if k:
            output = self._rotate(output, k)
        if orientation.flip_horizontal:
            output = self._flip_horizontal(output)
        if orientation.flip_vertical:
            output = self._flip_vertical(output)
        return output

    def _apply_overlays(self, frame: FrameLike) -> FrameLike:
        if not self.overlays:
            return frame
        overlays_enabled = True
        try:
            overlays_enabled = bool(self.overlay_enabled_provider())
        except Exception:  # pragma: no cover - defensive guard
            overlays_enabled = True
        if not overlays_enabled:
            return frame
        result = frame
        for overlay in self.overlays:
            result = overlay(result)
        return result

    def process(self, frame: FrameLike) -> FrameLike:
        orientation = self.orientation_provider().normalise()
        transformed = self._apply_orientation(frame, orientation)
        transformed = self._apply_preprocessors(transformed)
        return self._apply_overlays(transformed)

    def _rotate(self, frame: FrameLike, k: int) -> FrameLike:
        if _np is not None and isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return _np.rot90(frame, k)
        data = frame
        for _ in range(k % 4):
            data = [list(row) for row in zip(*data)][::-1]
        return data

    def _flip_horizontal(self, frame: FrameLike) -> FrameLike:
        if _np is not None and isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return _np.flip(frame, axis=1)
        return [list(reversed(row)) for row in frame]

    def _flip_vertical(self, frame: FrameLike) -> FrameLike:
        if _np is not None and isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            return _np.flip(frame, axis=0)
        return list(reversed(frame))

    def _apply_preprocessors(self, frame: FrameLike) -> FrameLike:
        if not self.preprocessors:
            return frame
        result = frame
        for transform in self.preprocessors:
            result = transform(result)
        return result


def compose_overlays(overlays: Iterable[OverlayFn]) -> OverlayFn:
    """Return a composite overlay function."""

    def _composed(frame: FrameLike) -> FrameLike:
        output = frame
        for overlay in overlays:
            output = overlay(output)
        return output

    return _composed


__all__ = ["FramePipeline", "compose_overlays", "OverlayFn"]
