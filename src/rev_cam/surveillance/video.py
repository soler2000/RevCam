"""Utilities for encoding surveillance clips to video files."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import av
import numpy as np

try:  # pragma: no cover - dependency availability varies on CI
    import simplejpeg
except ImportError as exc:  # pragma: no cover - dependency availability varies on CI
    simplejpeg = None  # type: ignore[assignment]
    _SIMPLEJPEG_ERROR = exc
else:  # pragma: no cover - dependency availability varies on CI
    _SIMPLEJPEG_ERROR = None


def ensure_rgb_frame(frame: np.ndarray | Sequence, *, even: bool = True) -> np.ndarray:
    """Return a contiguous RGB frame suitable for encoding."""

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
    elif array.ndim == 3:
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.shape[2] > 3:
            array = array[:, :, :3]
    else:
        raise ValueError("Expected a 2D or 3D frame for encoding")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if even:
        height, width = array.shape[:2]
        if width % 2:
            array = array[:, : width - 1, :]
        if height % 2:
            array = array[: height - 1, :, :]

    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    return array


def _codec_candidates(encoding: str) -> list[str]:
    codec = encoding.lower()
    if codec in {"h264", "libx264"}:
        return ["libx264", "h264", "mpeg4"]
    if codec in {"hevc", "h265", "libx265"}:
        return ["libx265", "hevc", "libx264", "h264"]
    return [codec, "libx264", "h264", "mpeg4"]


@dataclass(slots=True)
class VideoEncoder:
    """Incrementally encode frames into an MP4 container."""

    path: Path
    fps: int
    encoding: str
    width: int
    height: int
    _container: av.container.OutputContainer | None = field(init=False, default=None)
    _stream: av.video.stream.VideoStream | None = field(init=False, default=None)
    _frame_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self._open()

    # ------------------------------------------------------------------
    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(self.path.as_posix(), mode="w")
        stream = None
        for codec in _codec_candidates(self.encoding):
            try:
                stream = container.add_stream(codec, rate=self.fps)
            except av.AVError:
                continue
            else:
                break
        if stream is None:
            container.close()
            raise RuntimeError(f"No compatible encoder available for {self.encoding!r}")
        stream.width = int(self.width)
        stream.height = int(self.height)
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, int(self.fps))
        try:  # pragma: no cover - codec options availability varies
            stream.codec_context.options.update({"preset": "veryfast", "crf": "23"})
        except Exception:
            pass
        self._container = container
        self._stream = stream

    # ------------------------------------------------------------------
    def encode(self, frame: np.ndarray | Sequence) -> None:
        if self._stream is None or self._container is None:
            raise RuntimeError("Video encoder has been closed")
        rgb = ensure_rgb_frame(frame, even=True)
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = self._frame_index
        self._frame_index += 1
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._stream is None or self._container is None:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._stream = None
        self._container = None

    # ------------------------------------------------------------------
    def __enter__(self) -> "VideoEncoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def encode_frames_to_mp4(
    path: Path,
    frames: Iterable[np.ndarray | Sequence],
    *,
    fps: int,
    encoding: str,
) -> None:
    """Encode *frames* into an MP4 file at *path*."""

    frames_iter = list(frames)
    if not frames_iter:
        raise ValueError("At least one frame is required for encoding")
    first = ensure_rgb_frame(frames_iter[0], even=True)
    height, width = first.shape[:2]
    with VideoEncoder(path=path, fps=int(fps), encoding=encoding, width=width, height=height) as encoder:
        encoder.encode(first)
        for frame in frames_iter[1:]:
            encoder.encode(frame)


def write_thumbnail(path: Path, frame: np.ndarray | Sequence) -> None:
    """Persist *frame* as a JPEG thumbnail."""

    if simplejpeg is None:  # pragma: no cover - dependency availability varies
        raise RuntimeError("simplejpeg is required for thumbnail generation") from _SIMPLEJPEG_ERROR
    rgb = ensure_rgb_frame(frame, even=False)
    payload = simplejpeg.encode_jpeg(rgb, quality=85, colorspace="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


__all__ = ["VideoEncoder", "encode_frames_to_mp4", "ensure_rgb_frame", "write_thumbnail"]
