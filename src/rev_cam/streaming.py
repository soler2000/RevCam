"""MJPEG streaming helpers for RevCam."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator

import numpy as np

from .camera import BaseCamera
from .pipeline import FramePipeline

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependency availability varies by platform
    import simplejpeg
except ImportError as exc:  # pragma: no cover - dependency availability varies
    simplejpeg = None
    _SIMPLEJPEG_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency availability varies
    _SIMPLEJPEG_IMPORT_ERROR = None


@dataclass
class MJPEGStreamer:
    """Encode frames from a camera into an MJPEG stream."""

    camera: BaseCamera
    pipeline: FramePipeline
    fps: int = 20
    jpeg_quality: int = 85
    boundary: str = "frame"
    _frame_interval: float = field(init=False)
    _subscribers: set[asyncio.Queue[bytes]] = field(init=False, default_factory=set)
    _producer_task: asyncio.Task[None] | None = field(init=False, default=None)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be between 1 and 100")
        if simplejpeg is None:  # pragma: no cover - dependency availability varies
            raise RuntimeError(
                "simplejpeg is required for MJPEG streaming"
            ) from _SIMPLEJPEG_IMPORT_ERROR
        self._frame_interval = 1.0 / float(self.fps)

    @property
    def media_type(self) -> str:
        """Return the MIME type advertised for MJPEG responses."""

        return f"multipart/x-mixed-replace; boundary={self.boundary}"

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """Yield MJPEG chunks to the caller."""

        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        async with self._subscriber(queue):
            while True:
                payload = await queue.get()
                yield self._render_chunk(payload)

    async def aclose(self) -> None:
        """Stop the background encoder task and clear subscribers."""

        async with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
            task = self._producer_task
            self._producer_task = None
        for queue in subscribers:
            self._drain_queue(queue)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover - expected path
                pass

    @asynccontextmanager
    async def _subscriber(self, queue: asyncio.Queue[bytes]):
        await self._register(queue)
        try:
            yield
        finally:
            await self._unregister(queue)

    async def _register(self, queue: asyncio.Queue[bytes]) -> None:
        async with self._lock:
            self._subscribers.add(queue)
            if self._producer_task is None or self._producer_task.done():
                self._producer_task = asyncio.create_task(self._produce_frames())

    async def _unregister(self, queue: asyncio.Queue[bytes]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)
            should_stop = not self._subscribers
            task = self._producer_task if should_stop else None
            if should_stop:
                self._producer_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover - expected path
                pass

    async def _produce_frames(self) -> None:
        try:
            while True:
                iteration_start = time.perf_counter()
                try:
                    frame = await self.camera.get_frame()
                except asyncio.CancelledError:  # pragma: no cover - cooperative exit
                    raise
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to retrieve frame from camera")
                    await asyncio.sleep(self._frame_interval)
                    continue

                try:
                    processed = self.pipeline.process(frame)
                    jpeg = await asyncio.to_thread(self._encode_frame, processed)
                except asyncio.CancelledError:  # pragma: no cover - cooperative exit
                    raise
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to process frame for MJPEG stream")
                    await asyncio.sleep(self._frame_interval)
                    continue

                self._broadcast(jpeg)

                elapsed = time.perf_counter() - iteration_start
                sleep_for = self._frame_interval - elapsed
                if sleep_for > 0:
                    try:
                        await asyncio.sleep(sleep_for)
                    except asyncio.CancelledError:  # pragma: no cover - cooperative exit
                        raise
        except asyncio.CancelledError:  # pragma: no cover - cooperative exit
            pass
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("MJPEG producer crashed")

    def _broadcast(self, payload: bytes) -> None:
        for queue in list(self._subscribers):
            self._offer(queue, payload)

    def _offer(self, queue: asyncio.Queue[bytes], payload: bytes) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:  # pragma: no cover - rare path
            self._drain_queue(queue)
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover - defensive guard
                logger.debug("Dropping MJPEG frame after queue remained full")

    def _drain_queue(self, queue: asyncio.Queue[bytes]) -> None:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - expected when consumer caught up
            return

    def _encode_frame(self, frame: np.ndarray | list) -> bytes:
        """Encode an RGB frame into JPEG bytes."""

        array = np.asarray(frame)
        if array.ndim != 3:
            raise ValueError("Expected an RGB frame for MJPEG encoding")
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.shape[2] > 3:
            array = array[..., :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        return simplejpeg.encode_jpeg(
            array,
            quality=int(self.jpeg_quality),
            colorspace="RGB",
            fastdct=True,
            fastupsample=True,
        )

    def _render_chunk(self, payload: bytes) -> bytes:
        header = (
            f"--{self.boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "\r\n"
        ).encode("ascii")
        return header + payload + b"\r\n"


__all__ = ["MJPEGStreamer"]
