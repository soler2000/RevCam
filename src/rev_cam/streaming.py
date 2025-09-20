"""MJPEG streaming helpers for RevCam."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import AsyncGenerator

import av
import numpy as np

from .camera import BaseCamera
from .pipeline import FramePipeline

logger = logging.getLogger(__name__)


@dataclass
class MJPEGStreamer:
    """Encode frames from a camera into an MJPEG stream."""

    camera: BaseCamera
    pipeline: FramePipeline
    fps: int = 15
    jpeg_quality: int = 80
    boundary: str = "frame"
    _encoder: av.codec.context.CodecContext | None = field(init=False, default=None)
    _encoder_size: tuple[int, int] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be between 1 and 100")
        self._frame_interval = 1.0 / float(self.fps)

    @property
    def media_type(self) -> str:
        """Return the MIME type advertised for MJPEG responses."""

        return f"multipart/x-mixed-replace; boundary={self.boundary}"

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """Yield MJPEG chunks indefinitely."""

        while True:
            iteration_start = time.perf_counter()
            try:
                frame = await self.camera.get_frame()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to retrieve frame from camera")
                await asyncio.sleep(self._frame_interval)
                continue

            try:
                processed = self.pipeline.process(frame)
                jpeg = await asyncio.to_thread(self._encode_frame, processed)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to process frame for MJPEG stream")
                await asyncio.sleep(self._frame_interval)
                continue

            yield self._render_chunk(jpeg)

            elapsed = time.perf_counter() - iteration_start
            sleep_for = self._frame_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    def _quality_parameter(self) -> int:
        scale = max(0, min(100, int(self.jpeg_quality)))
        q_value = int(round((100 - scale) * 0.3)) + 1
        return max(1, min(31, q_value))

    def _ensure_encoder(self, width: int, height: int) -> av.codec.context.CodecContext:
        size = (width, height)
        if self._encoder is not None and self._encoder_size == size:
            return self._encoder

        options = {"q": str(self._quality_parameter())}
        encoder = av.codec.CodecContext.create("mjpeg", "w", options)
        encoder.width = width
        encoder.height = height
        encoder.pix_fmt = "yuvj420p"
        encoder.time_base = Fraction(1, max(1, self.fps))
        self._encoder = encoder
        self._encoder_size = size
        return encoder

    def _encode_frame(self, frame: np.ndarray | list) -> bytes:
        """Encode an RGB frame into JPEG bytes."""

        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("Expected an RGB frame for MJPEG encoding")
        if array.shape[2] > 3:
            array = array[..., :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        height, width, _ = array.shape
        encoder = self._ensure_encoder(width, height)
        video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        payload = b""
        for packet in encoder.encode(video_frame):
            payload += packet.to_bytes()
        if not payload:
            for packet in encoder.encode(None):
                payload += packet.to_bytes()
        if not payload:
            raise RuntimeError("MJPEG encoder produced no data")
        return payload

    def _render_chunk(self, payload: bytes) -> bytes:
        header = (
            f"--{self.boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "\r\n"
        ).encode("ascii")
        return header + payload + b"\r\n"


__all__ = ["MJPEGStreamer"]
