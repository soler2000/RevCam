"""Image adjustment helpers for camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:  # pragma: no cover - optional dependency in some environments
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None


def _clamp_percent(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("Image adjustments must be numeric") from exc
    if numeric < 0.0:
        return 0.0
    if numeric > 200.0:
        return 200.0
    return numeric


def _clamp_hue(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("Hue must be numeric") from exc
    if numeric < -180.0:
        return -180.0
    if numeric > 180.0:
        return 180.0
    return numeric


@dataclass(frozen=True, slots=True)
class ImageAdjustments:
    """Describes hue, saturation and brightness tweaks."""

    brightness: float = 100.0
    saturation: float = 100.0
    hue: float = 0.0

    def __post_init__(self) -> None:
        brightness = _clamp_percent(self.brightness)
        saturation = _clamp_percent(self.saturation)
        hue = _clamp_hue(self.hue)
        object.__setattr__(self, "brightness", brightness)
        object.__setattr__(self, "saturation", saturation)
        object.__setattr__(self, "hue", hue)

    def to_dict(self) -> dict[str, float]:
        return {
            "brightness": float(self.brightness),
            "saturation": float(self.saturation),
            "hue": float(self.hue),
        }

    def is_identity(self) -> bool:
        return (
            abs(self.brightness - 100.0) < 1e-6
            and abs(self.saturation - 100.0) < 1e-6
            and abs(self.hue) < 1e-6
        )


DEFAULT_IMAGE_ADJUSTMENTS = ImageAdjustments()


def adjust_frame(frame, adjustments: ImageAdjustments):
    """Return *frame* with *adjustments* applied.

    When NumPy is unavailable or *frame* is not a 3-channel array the input frame is
    returned unchanged.
    """

    if _np is None or adjustments is None or adjustments.is_identity():  # pragma: no cover - trivial path
        return frame

    array = _np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        return frame

    rgb = array[..., :3].astype(_np.float32) / 255.0
    hue_shift = float(adjustments.hue) / 360.0
    saturation_scale = float(adjustments.saturation) / 100.0
    brightness_scale = float(adjustments.brightness) / 100.0

    hsv = _rgb_to_hsv(rgb)
    h = (hsv[..., 0] + hue_shift) % 1.0
    s = _np.clip(hsv[..., 1] * saturation_scale, 0.0, 1.0)
    v = _np.clip(hsv[..., 2], 0.0, 1.0)
    hsv_adjusted = _np.stack((h, s, v), axis=-1)
    adjusted_rgb = _hsv_to_rgb(hsv_adjusted)
    adjusted_rgb *= brightness_scale
    adjusted_rgb = _np.clip(adjusted_rgb, 0.0, 1.0)

    output_rgb = (adjusted_rgb * 255.0).astype(array.dtype, copy=False)
    if array.shape[2] > 3:
        result = array.copy()
        result[..., :3] = output_rgb
        return result
    return output_rgb


def build_adjustment_filter(
    provider: Callable[[], ImageAdjustments]
) -> Callable[[object], object]:
    """Return a callable that applies adjustments from *provider* to frames."""

    def _apply(frame):
        adjustments = provider()
        if not isinstance(adjustments, ImageAdjustments):
            return frame
        return adjust_frame(frame, adjustments)

    return _apply


def _rgb_to_hsv(rgb: "_np.ndarray") -> "_np.ndarray":
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc - minc
    delta = _np.where(delta < 1e-6, 0.0, delta)

    hue = _np.zeros_like(maxc)
    mask = delta > 0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    idx = mask & (maxc == r)
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = mask & (maxc == g)
    hue[idx] = (b[idx] - r[idx]) / delta[idx] + 2.0
    idx = mask & (maxc == b)
    hue[idx] = (r[idx] - g[idx]) / delta[idx] + 4.0
    hue /= 6.0
    hue = hue % 1.0

    saturation = _np.zeros_like(maxc)
    nonzero = maxc > 0
    saturation[nonzero] = delta[nonzero] / maxc[nonzero]

    value = maxc
    return _np.stack((hue, saturation, value), axis=-1)


def _hsv_to_rgb(hsv: "_np.ndarray") -> "_np.ndarray":
    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]

    i = _np.floor(h).astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    conditions = [
        (i_mod == 0, _np.stack((v, t, p), axis=-1)),
        (i_mod == 1, _np.stack((q, v, p), axis=-1)),
        (i_mod == 2, _np.stack((p, v, t), axis=-1)),
        (i_mod == 3, _np.stack((p, q, v), axis=-1)),
        (i_mod == 4, _np.stack((t, p, v), axis=-1)),
        (i_mod == 5, _np.stack((v, p, q), axis=-1)),
    ]

    rgb = _np.zeros_like(hsv)
    for mask, value in conditions:
        rgb[mask] = value[mask]
    return rgb


__all__ = ["ImageAdjustments", "DEFAULT_IMAGE_ADJUSTMENTS", "adjust_frame", "build_adjustment_filter"]
