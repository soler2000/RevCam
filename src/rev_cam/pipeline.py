"""Video frame processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None

from .config import Orientation

FrameLike = Any
OverlayFn = Callable[[FrameLike], FrameLike]


@dataclass
class FramePipeline:
    """Applies orientation and overlay transforms to frames."""

    orientation_provider: Callable[[], Orientation]
    overlays: List[OverlayFn] = field(default_factory=list)
    overlays_enabled_provider: Optional[Callable[[], bool]] = None

    def add_overlay(self, overlay: OverlayFn) -> None:
        self.overlays.append(overlay)

    def set_overlays_enabled_provider(self, provider: Callable[[], bool] | None) -> None:
        self.overlays_enabled_provider = provider

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
        if self.overlays_enabled_provider is not None:
            try:
                overlays_enabled = self.overlays_enabled_provider()
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


def compose_overlays(overlays: Iterable[OverlayFn]) -> OverlayFn:
    """Return a composite overlay function."""

    def _composed(frame: FrameLike) -> FrameLike:
        output = frame
        for overlay in overlays:
            output = overlay(output)
        return output

    return _composed


__all__ = ["FramePipeline", "compose_overlays", "OverlayFn"]
