"""WebRTC helpers for RevCam."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Set

from av import VideoFrame

from .camera import BaseCamera
from .pipeline import FramePipeline

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import VideoStreamTrack

try:  # pragma: no cover - optional dependency
    from aiortc import RTCPeerConnection as _RTCPeerConnection
    from aiortc import RTCSessionDescription as _RTCSessionDescription
    from aiortc.mediastreams import VideoStreamTrack as _VideoStreamTrack
except ImportError as exc:  # pragma: no cover - handled at runtime
    _RTCPeerConnection = None  # type: ignore[assignment]
    _RTCSessionDescription = None  # type: ignore[assignment]
    _AIORTC_IMPORT_ERROR = exc

    class _VideoStreamTrack:  # type: ignore[no-redef]
        """Stub base class used when aiortc is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

else:  # pragma: no cover - executed when dependency is installed
    _AIORTC_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


def _ensure_aiortc_available() -> None:
    """Raise a helpful error when the optional aiortc dependency is missing."""

    if _AIORTC_IMPORT_ERROR is not None:
        raise RuntimeError(
            "aiortc is required for WebRTC functionality. Install the 'aiortc' package to enable streaming."
        ) from _AIORTC_IMPORT_ERROR


class PipelineVideoTrack(_VideoStreamTrack):
    """Video track that pulls frames from a camera via the processing pipeline."""

    def __init__(self, camera: BaseCamera, pipeline: FramePipeline, fps: int = 30) -> None:
        _ensure_aiortc_available()
        super().__init__()
        self._camera = camera
        self._pipeline = pipeline
        self._frame_time = Fraction(1, fps)
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._frame_task: asyncio.Task[None] | None = None
        self._frame_error: BaseException | None = None
        self._stopped = False

    async def recv(self) -> VideoFrame:
        _ensure_aiortc_available()
        await self._ensure_producer()
        pts, time_base = await self.next_timestamp()
        producer = self._frame_task
        if producer is None:
            raise RuntimeError("Frame producer is not running")
        if producer.done():
            producer.result()
        frame = await self._frame_queue.get()
        while True:
            try:
                frame = self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        processed = self._pipeline.process(frame)
        video_frame = VideoFrame.from_ndarray(processed, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    async def _ensure_producer(self) -> None:
        if self._stopped:
            raise RuntimeError("PipelineVideoTrack has been stopped")
        if self._frame_error is not None:
            exc = self._frame_error
            self._frame_error = None
            raise exc
        if self._frame_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._frame_task = loop.create_task(self._frame_producer())
        self._frame_task.add_done_callback(self._on_producer_done)

    async def _frame_producer(self) -> None:
        try:
            while not self._stopped:
                frame = await self._camera.get_frame()
                await self._push_latest_frame(frame)
        except asyncio.CancelledError:
            raise

    async def _push_latest_frame(self, frame) -> None:
        while True:
            try:
                self._frame_queue.put_nowait(frame)
                return
            except asyncio.QueueFull:
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)

    def _on_producer_done(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - logging only
            self._frame_error = exc
            logger.exception("Frame producer terminated unexpectedly", exc_info=True)
        finally:
            self._frame_task = None

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        task = self._frame_task
        if task is not None:
            def _silence(done_task: asyncio.Task[None]) -> None:
                try:
                    done_task.result()
                except asyncio.CancelledError:
                    pass
                except Exception:  # pragma: no cover - logging only
                    logger.exception(
                        "Frame producer raised during shutdown", exc_info=True
                    )

            task.cancel()
            task.add_done_callback(_silence)
            self._frame_task = None
        super().stop()


@dataclass
class WebRTCManager:
    """Manage WebRTC peer connections."""

    camera: BaseCamera
    pipeline: FramePipeline
    connections: Set[RTCPeerConnection] | None = None

    def __post_init__(self) -> None:
        _ensure_aiortc_available()
        if self.connections is None:
            self.connections = set()

    async def handle_offer(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        _ensure_aiortc_available()
        pc = _RTCPeerConnection()
        track = PipelineVideoTrack(self.camera, self.pipeline)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:  # pragma: no cover - logging only
            if pc.connectionState in {"failed", "closed"}:
                await self._close_connection(pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        self.connections.add(pc)
        return pc.localDescription

    async def _close_connection(self, pc: RTCPeerConnection) -> None:
        await pc.close()
        self.connections.discard(pc)

    async def shutdown(self) -> None:
        await asyncio.gather(*(self._close_connection(pc) for pc in list(self.connections or [])), return_exceptions=True)
        if self.connections is not None:
            self.connections.clear()


__all__ = ["PipelineVideoTrack", "WebRTCManager"]
