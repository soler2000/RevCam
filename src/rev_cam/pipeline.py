"""Video frame processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Tuple

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
    _scratch_buffers: Dict[Tuple[Tuple[int, ...], str], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def add_overlay(self, overlay: OverlayFn) -> None:
        self.overlays.append(overlay)

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
            k = k % 4
            if k == 0:
                return frame
            if frame.ndim < 2:
                return frame

            if k == 2:
                view = _np.flip(_np.flip(frame, axis=0), axis=1)
            else:
                axes = list(range(frame.ndim))
                axes[0], axes[1] = axes[1], axes[0]
                view = frame.transpose(axes)
                if k == 1:
                    view = _np.flip(view, axis=1)
                else:
                    view = _np.flip(view, axis=0)

            scratch = self._get_scratch_buffer(view.shape, frame.dtype)
            _np.copyto(scratch, view, casting="unsafe")
            return scratch
        data = frame
        for _ in range(k % 4):
            data = [list(row) for row in zip(*data)][::-1]
        return data

    def _flip_horizontal(self, frame: FrameLike) -> FrameLike:
        if _np is not None and isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            frame[...] = frame[:, ::-1, ...]
            return frame
        return [list(reversed(row)) for row in frame]

    def _flip_vertical(self, frame: FrameLike) -> FrameLike:
        if _np is not None and isinstance(frame, _np.ndarray):  # pragma: no cover - optional path
            frame[...] = frame[::-1, ...]
            return frame
        return list(reversed(frame))

    def _get_scratch_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: Any,
    ) -> _np.ndarray:
        if _np is None:
            raise RuntimeError("Scratch buffers require numpy")

        normalised_shape = tuple(int(dim) for dim in shape)
        dtype_obj = _np.dtype(dtype)
        key = (normalised_shape, dtype_obj.str)
        scratch = self._scratch_buffers.get(key)
        if scratch is None or scratch.shape != normalised_shape:
            scratch = _np.empty(normalised_shape, dtype=dtype_obj)
            self._scratch_buffers[key] = scratch
        return scratch


def compose_overlays(overlays: Iterable[OverlayFn]) -> OverlayFn:
    """Return a composite overlay function."""

    def _composed(frame: FrameLike) -> FrameLike:
        output = frame
        for overlay in overlays:
            output = overlay(output)
        return output

    return _composed


__all__ = ["FramePipeline", "compose_overlays", "OverlayFn"]
