"""Shared video encoder discovery helpers."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterator

try:  # pragma: no cover - dependency availability varies
    import av  # type: ignore
except ImportError:  # pragma: no cover - dependency availability varies
    av = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class H264EncoderBackend:
    """Represents a concrete H.264 encoder implementation."""

    key: str
    codec: str
    label: str
    hardware: bool = False
    requires_even_dimensions: bool = True


_H264_ENCODER_BACKENDS: tuple[H264EncoderBackend, ...] = (
    H264EncoderBackend(
        key="v4l2m2m",
        codec="h264_v4l2m2m",
        label="V4L2 M2M (Raspberry Pi hardware)",
        hardware=True,
    ),
    H264EncoderBackend(
        key="libx264",
        codec="libx264",
        label="libx264 (software)",
        hardware=False,
    ),
)

_H264_ENCODER_BY_KEY = {backend.key: backend for backend in _H264_ENCODER_BACKENDS}
_H264_ENCODER_ALIASES = {
    "auto": "auto",
    "default": "auto",
    "hardware": "v4l2m2m",
    "software": "libx264",
    "libx264": "libx264",
    "x264": "libx264",
    "v4l2m2m": "v4l2m2m",
    "h264_v4l2m2m": "v4l2m2m",
    "pi": "v4l2m2m",
}


H264_EVEN_DIMENSION_CODECS: frozenset[str] = frozenset(
    backend.codec for backend in _H264_ENCODER_BACKENDS if backend.requires_even_dimensions
)
"""Codec names which require even frame dimensions."""


def list_h264_backends() -> tuple[H264EncoderBackend, ...]:
    """Return the available H.264 encoder backend definitions."""

    return _H264_ENCODER_BACKENDS


def normalise_encoder_choice(choice: str | None) -> str:
    """Normalise a user-provided encoder choice string."""

    if not choice:
        return "auto"
    key = choice.strip().lower()
    if not key:
        return "auto"
    return _H264_ENCODER_ALIASES.get(key, key)


def _iter_candidate_backends(preference: str) -> Iterator[H264EncoderBackend]:
    if preference == "auto":
        yield from _H264_ENCODER_BACKENDS
        return

    backend = _H264_ENCODER_BY_KEY.get(preference)
    if backend is not None:
        yield backend
        for candidate in _H264_ENCODER_BACKENDS:
            if candidate.key != backend.key:
                yield candidate
        return

    # Fallback to codec name matching when the key is unknown but the codec is valid.
    for candidate in _H264_ENCODER_BACKENDS:
        if candidate.codec == preference:
            yield candidate
            for extra in _H264_ENCODER_BACKENDS:
                if extra is not candidate:
                    yield extra
            return

    # Unknown preference – fall back to auto discovery.
    logger.debug("Unknown H.264 encoder preference %r; falling back to auto", preference)
    yield from _H264_ENCODER_BACKENDS


def _probe_backend(backend: H264EncoderBackend) -> bool:
    if av is None:  # pragma: no cover - dependency availability varies
        return False
    try:
        context = av.CodecContext.create(backend.codec, "w")
    except av.FFmpegError as exc:  # pragma: no cover - codec probing failure
        logger.debug("Codec %s unavailable: %s", backend.codec, exc)
        return False
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to initialise codec %s: %s", backend.codec, exc)
        return False
    if not getattr(context, "is_encoder", True):
        logger.debug("Codec %s is not an encoder", backend.codec)
        return False
    return True


def select_h264_backend(
    preference: str | None,
    *,
    allow_auto_fallback: bool = True,
) -> tuple[H264EncoderBackend | None, tuple[str, ...]]:
    """Select an H.264 encoder backend based on ``preference``.

    Returns a tuple of ``(backend, attempted_codecs)`` where ``backend`` is ``None``
    if no usable encoder was discovered. ``attempted_codecs`` lists the codec names
    which were probed in order.
    """

    normalised = normalise_encoder_choice(preference)
    attempted: list[str] = []
    for backend in _iter_candidate_backends(normalised):
        attempted.append(backend.codec)
        if _probe_backend(backend):
            return backend, tuple(attempted)
        if not allow_auto_fallback:
            break
    return None, tuple(attempted)


__all__ = [
    "H264EncoderBackend",
    "H264_EVEN_DIMENSION_CODECS",
    "list_h264_backends",
    "normalise_encoder_choice",
    "select_h264_backend",
]
