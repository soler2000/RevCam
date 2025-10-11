"""Streaming helpers for RevCam."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from typing import AsyncGenerator, Callable

import numpy as np

from .camera import BaseCamera
from .pipeline import FramePipeline

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependency availability varies by platform
    import simplejpeg
except ImportError as exc:  # pragma: no cover - dependency availability varies
    simplejpeg = None
    _SIMPLEJPEG_IMPORT_ERROR = exc
    _SIMPLEJPEG_ENCODE_KWARGS: set[str] = set()
else:  # pragma: no cover - dependency availability varies
    _SIMPLEJPEG_IMPORT_ERROR = None
    try:  # pragma: no cover - dependency availability varies
        _SIMPLEJPEG_ENCODE_KWARGS = set(
            inspect.signature(simplejpeg.encode_jpeg).parameters
        )
    except (TypeError, ValueError):  # pragma: no cover - C-extension signature unsupported
        _SIMPLEJPEG_ENCODE_KWARGS = set()

try:  # pragma: no cover - dependency availability varies by platform
    from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError
except ImportError as exc:  # pragma: no cover - dependency availability varies
    MediaStreamTrack = None  # type: ignore[assignment]
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    MediaStreamError = None  # type: ignore[assignment]
    _AIORTC_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency availability varies
    _AIORTC_IMPORT_ERROR = None

try:  # pragma: no cover - dependency availability varies by platform
    from av import VideoFrame
except ImportError as exc:  # pragma: no cover - dependency availability varies
    VideoFrame = None  # type: ignore[assignment]
    _AV_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency availability varies
    _AV_IMPORT_ERROR = None


def encode_frame_to_jpeg(
    frame: np.ndarray | list,
    *,
    quality: int,
) -> bytes:
    """Encode an RGB frame into JPEG bytes using the configured quality."""

    if simplejpeg is None:  # pragma: no cover - dependency availability varies
        raise RuntimeError(
            "simplejpeg is required for JPEG encoding"
        ) from _SIMPLEJPEG_IMPORT_ERROR

    array = _prepare_rgb_frame(frame)
    encode_kwargs: dict[str, object] = {
        "quality": int(quality),
        "colorspace": "RGB",
    }
    if "fastdct" in _SIMPLEJPEG_ENCODE_KWARGS:
        encode_kwargs["fastdct"] = True
    if "fastupsample" in _SIMPLEJPEG_ENCODE_KWARGS:
        encode_kwargs["fastupsample"] = True

    return simplejpeg.encode_jpeg(array, **encode_kwargs)


def _prepare_rgb_frame(frame: np.ndarray | list) -> np.ndarray:
    """Return a contiguous uint8 RGB frame for video transmission."""

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
    elif array.ndim == 3:
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.shape[2] > 3:
            array = array[:, :, :3]
    else:
        raise ValueError("Expected a 2D or 3D frame for video streaming")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    return array


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

    def apply_settings(
        self,
        *,
        fps: int | None = None,
        jpeg_quality: int | None = None,
    ) -> None:
        """Update streaming parameters at runtime."""

        update_interval = False
        if fps is not None:
            new_fps = int(fps)
            if new_fps <= 0:
                raise ValueError("fps must be positive")
            if new_fps != self.fps:
                self.fps = new_fps
                update_interval = True

        if jpeg_quality is not None:
            new_quality = int(jpeg_quality)
            if new_quality < 1:
                new_quality = 1
            elif new_quality > 100:
                new_quality = 100
            self.jpeg_quality = new_quality

        if update_interval:
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

        return encode_frame_to_jpeg(frame, quality=self.jpeg_quality)

    def _render_chunk(self, payload: bytes) -> bytes:
        header = (
            f"--{self.boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "\r\n"
        ).encode("ascii")
        return header + payload + b"\r\n"


if MediaStreamTrack is not None and VideoFrame is not None and MediaStreamError is not None:

    class PipelineVideoTrack(MediaStreamTrack):
        """Expose camera frames via WebRTC."""

        kind = "video"

        def __init__(self, manager: "WebRTCManager") -> None:
            super().__init__()
            self._manager = manager
            self._time_base = Fraction(1, 90_000)
            self._timestamp = 0
            self._stopped = False
            self._frame_interval = 1.0 / float(manager.fps)
            self._frame_increment = max(1, int(self._frame_interval * self._time_base.denominator))
            self._manager._tracks.add(self)

        def update_fps(self, fps: int) -> None:
            if fps <= 0:
                raise ValueError("fps must be positive")
            self._frame_interval = 1.0 / float(fps)
            self._frame_increment = max(1, int(self._frame_interval * self._time_base.denominator))

        def stop(self) -> None:  # pragma: no cover - exercised indirectly
            if self._stopped:
                return
            self._stopped = True
            self._manager._tracks.discard(self)
            super().stop()

        async def recv(self):  # type: ignore[override]
            if self._stopped:
                raise MediaStreamError("Track has been stopped")

            iteration_start = time.perf_counter()

            try:
                frame = await self._manager.camera.get_frame()
            except asyncio.CancelledError:  # pragma: no cover - cooperative exit
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to retrieve frame for WebRTC stream")
                raise MediaStreamError("Camera failure") from exc

            try:
                processed = await asyncio.to_thread(self._manager.pipeline.process, frame)
                rgb = _prepare_rgb_frame(processed)
            except asyncio.CancelledError:  # pragma: no cover - cooperative exit
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to process frame for WebRTC stream")
                raise MediaStreamError("Pipeline failure") from exc

            video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
            self._timestamp += self._frame_increment
            video_frame.pts = self._timestamp
            video_frame.time_base = self._time_base

            elapsed = time.perf_counter() - iteration_start
            remaining = self._frame_interval - elapsed
            if remaining > 0:
                try:
                    await asyncio.sleep(remaining)
                except asyncio.CancelledError:  # pragma: no cover - cooperative exit
                    raise
            return video_frame


else:  # pragma: no cover - dependency availability varies

    class PipelineVideoTrack:  # type: ignore[no-redef]
        """Placeholder track used when aiortc is unavailable."""

        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("aiortc and PyAV are required for WebRTC streaming")


@dataclass(eq=False)
class _WebRTCSession:
    manager: "WebRTCManager"
    pc: RTCPeerConnection
    track: PipelineVideoTrack
    _closed: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True

        try:
            self.track.stop()
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Ignoring error while stopping WebRTC track", exc_info=True)

        try:
            await self.pc.close()
        finally:
            await self.manager._discard_session(self)


@dataclass
class WebRTCManager:
    """Manage WebRTC peer connections for streaming."""

    camera: BaseCamera
    pipeline: FramePipeline
    fps: int = 20
    peer_connection_factory: Callable[[], object] | None = None
    _sessions: set[_WebRTCSession] = field(init=False, default_factory=set)
    _tracks: set[PipelineVideoTrack] = field(init=False, default_factory=set)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if (
            MediaStreamTrack is None
            or RTCPeerConnection is None
            or RTCSessionDescription is None
            or MediaStreamError is None
            or VideoFrame is None
        ):  # pragma: no cover - dependency availability varies
            if _AIORTC_IMPORT_ERROR is not None:
                raise RuntimeError("aiortc is required for WebRTC streaming") from _AIORTC_IMPORT_ERROR
            if _AV_IMPORT_ERROR is not None:
                raise RuntimeError("PyAV is required for WebRTC streaming") from _AV_IMPORT_ERROR
            raise RuntimeError("WebRTC dependencies are unavailable")

        if self.peer_connection_factory is None:
            self.peer_connection_factory = RTCPeerConnection  # type: ignore[assignment]

    async def create_session(self, sdp: str, offer_type: str) -> RTCSessionDescription:
        """Create a WebRTC session from a remote offer and return the answer."""

        if RTCSessionDescription is None or RTCPeerConnection is None:  # pragma: no cover
            raise RuntimeError("aiortc is required for WebRTC streaming")

        offer = RTCSessionDescription(sdp=sdp, type=offer_type)
        pc = self.peer_connection_factory()  # type: ignore[operator]

        if not isinstance(pc, RTCPeerConnection):
            raise TypeError("peer_connection_factory must return an RTCPeerConnection instance")

        track = PipelineVideoTrack(self)
        session = _WebRTCSession(self, pc, track)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def _on_state_change() -> None:  # pragma: no cover - event driven
            state = pc.connectionState
            if state in {"closed", "failed", "disconnected"}:
                await session.close()

        try:
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception:
            await session.close()
            raise

        async with self._lock:
            self._sessions.add(session)

        if pc.localDescription is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Peer connection did not provide a local description")

        return pc.localDescription

    async def _discard_session(self, session: _WebRTCSession) -> None:
        async with self._lock:
            self._sessions.discard(session)

    def apply_settings(
        self,
        *,
        fps: int | None = None,
        jpeg_quality: int | None = None,  # noqa: ARG002 - kept for API parity
    ) -> None:
        """Update WebRTC streaming parameters."""

        update_tracks = False
        if fps is not None:
            new_fps = int(fps)
            if new_fps <= 0:
                raise ValueError("fps must be positive")
            if new_fps != self.fps:
                self.fps = new_fps
                update_tracks = True

        if update_tracks:
            for track in list(self._tracks):
                try:
                    track.update_fps(self.fps)
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug("Failed to update WebRTC track FPS", exc_info=True)

    async def aclose(self) -> None:
        async with self._lock:
            sessions = list(self._sessions)
            self._sessions.clear()

        for session in sessions:
            await session.close()


__all__ = ["MJPEGStreamer", "WebRTCManager", "PipelineVideoTrack", "encode_frame_to_jpeg"]
