"""Asynchronous video source management for the camera pipeline."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from .camera import BaseCamera, CameraError
from .pipeline import FramePipeline


logger = logging.getLogger(__name__)


class VideoSource:
    """Continuously pulls frames from the active camera through the pipeline."""

    def __init__(self, pipeline: FramePipeline, fps: int = 30) -> None:
        self._pipeline = pipeline
        self._frame_interval = 1.0 / fps
        self._camera_lock = asyncio.Lock()
        self._camera_ready = asyncio.Event()
        self._frame_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._frame: Optional[np.ndarray] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._camera: Optional[BaseCamera] = None

    async def start(self) -> None:
        """Begin processing frames from the active camera."""

        if self._task is not None:
            return
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._run(), name="rev-cam-video-source")

    async def stop(self) -> None:
        """Stop processing and release any active camera."""

        self._shutdown_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.set_camera(None)
        self._frame_event.set()

    async def set_camera(self, camera: BaseCamera | None) -> None:
        """Switch to a new camera, closing any previous instance."""

        async with self._camera_lock:
            previous = self._camera
            self._camera = camera
            if camera is None:
                self._camera_ready.clear()
            else:
                self._camera_ready.set()

        if previous is not None and previous is not camera:
            try:
                await previous.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning("Failed to close previous camera: %s", exc)

    async def get_frame(self) -> np.ndarray:
        """Return the most recent processed frame, waiting if necessary."""

        while True:
            await self._frame_event.wait()
            self._frame_event.clear()
            if self._frame is None:
                if self._shutdown_event.is_set():
                    raise CameraError("Video source has been stopped")
                continue
            return self._frame

    async def _run(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                await self._camera_ready.wait()
                async with self._camera_lock:
                    camera = self._camera
                if camera is None:
                    await asyncio.sleep(self._frame_interval)
                    continue
                try:
                    frame = await camera.get_frame()
                except Exception as exc:  # pragma: no cover - hardware dependent
                    logger.error("Failed to read frame from camera: %s", exc)
                    await asyncio.sleep(self._frame_interval)
                    continue
                processed = self._pipeline.process(frame)
                self._frame = processed
                self._frame_event.set()
                await asyncio.sleep(self._frame_interval)
        except asyncio.CancelledError:  # pragma: no cover - task shutdown
            pass
        finally:
            self._frame_event.set()


__all__ = ["VideoSource"]
