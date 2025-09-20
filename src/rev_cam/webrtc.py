"""WebRTC helpers for RevCam."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from av import VideoFrame

from .camera import BaseCamera
from .pipeline import FramePipeline

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import VideoStreamTrack
    from httpx import AsyncClient as _HTTPXAsyncClient
else:  # pragma: no cover - runtime fallback
    _HTTPXAsyncClient = object

try:  # pragma: no cover - optional dependency
    import httpx as _httpx
except ImportError as exc:  # pragma: no cover - handled at runtime
    _httpx = None  # type: ignore[assignment]
    _HTTPX_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when dependency is installed
    _HTTPX_IMPORT_ERROR = None

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


class MediaMTXError(RuntimeError):
    """Base error raised for MediaMTX signalling failures."""


class MediaMTXConnectionError(MediaMTXError):
    """Raised when MediaMTX cannot be reached over the network."""


class WebRTCError(RuntimeError):
    """Raised when RevCam cannot complete a WebRTC exchange."""


def _ensure_aiortc_available() -> None:
    """Raise a helpful error when the optional aiortc dependency is missing."""

    if _AIORTC_IMPORT_ERROR is not None:
        raise RuntimeError(
            "aiortc is required for WebRTC functionality. Install the 'aiortc' package to enable streaming."
        ) from _AIORTC_IMPORT_ERROR


def _ensure_httpx_available() -> None:
    """Raise a consistent error when httpx is missing."""

    if _HTTPX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "httpx is required for MediaMTX WebRTC streaming. Install the 'httpx' package to enable streaming."
        ) from _HTTPX_IMPORT_ERROR


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


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


@dataclass(slots=True)
class MediaMTXConfig:
    """Connection information for a MediaMTX deployment."""

    base_url: str = "http://127.0.0.1:8889"
    stream_name: str = "revcam"
    publish_path: str | None = None
    play_path: str | None = None
    timeout: float = 10.0

    def publish_url(self) -> str:
        path = self.publish_path or f"whip/{self.stream_name}"
        return _join_url(self.base_url, path)

    def play_url(self) -> str:
        path = self.play_path or f"whep/{self.stream_name}"
        return _join_url(self.base_url, path)


class MediaMTXClient:
    """Thin async HTTP client for the MediaMTX WHIP/WHEP endpoints."""

    def __init__(self, config: MediaMTXConfig) -> None:
        self._config = config
        self._client: _HTTPXAsyncClient | None = None

    async def _exchange(
        self, url: str, offer: RTCSessionDescription
    ) -> RTCSessionDescription:
        _ensure_httpx_available()
        client = await self._get_client()
        try:
            response = await client.post(
                url,
                content=offer.sdp,
                headers={"Content-Type": "application/sdp"},
            )
            response.raise_for_status()
        except getattr(_httpx, "ConnectError") as exc:  # type: ignore[arg-type]
            raise MediaMTXConnectionError(
                f"Unable to connect to MediaMTX at {url}: {exc}"
            ) from exc
        except getattr(_httpx, "HTTPStatusError") as exc:  # type: ignore[arg-type]
            status = exc.response.status_code
            raise MediaMTXError(
                f"MediaMTX returned HTTP {status} for {url}: {exc.response.text.strip()}"
            ) from exc
        except getattr(_httpx, "HTTPError") as exc:  # type: ignore[arg-type]
            raise MediaMTXError(
                f"MediaMTX request to {url} failed: {exc}"
            ) from exc
        return _RTCSessionDescription(sdp=response.text, type="answer")

    async def _get_client(self) -> _HTTPXAsyncClient:
        _ensure_httpx_available()
        if self._client is None:
            self._client = _httpx.AsyncClient(timeout=self._config.timeout)
        return self._client

    async def publish(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        _ensure_aiortc_available()
        return await self._exchange(self._config.publish_url(), offer)

    async def play(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        _ensure_aiortc_available()
        return await self._exchange(self._config.play_url(), offer)

    async def close(self) -> None:
        if self._client is not None:
            _ensure_httpx_available()
            await self._client.aclose()
            self._client = None


async def _wait_for_ice_completion(pc: RTCPeerConnection, timeout: float) -> None:
    if pc.iceGatheringState == "complete":
        return

    event = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_state_change() -> None:  # pragma: no cover - small helper
        if pc.iceGatheringState == "complete":
            event.set()

    if pc.iceGatheringState == "complete":
        event.set()

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except asyncio.TimeoutError as exc:  # pragma: no cover - exceptional path
        raise RuntimeError("Timed out while gathering ICE candidates for MediaMTX") from exc


class MediaMTXPublisher:
    """Publish camera frames to MediaMTX over WebRTC using WHIP."""

    def __init__(
        self,
        camera: BaseCamera,
        pipeline: FramePipeline,
        client: MediaMTXClient,
        *,
        config: MediaMTXConfig,
        fps: int = 30,
    ) -> None:
        _ensure_aiortc_available()
        self._camera = camera
        self._pipeline = pipeline
        self._client = client
        self._config = config
        self._fps = fps
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._start_error: BaseException | None = None

    async def start(self) -> None:
        if self._task is None:
            self._stop_event = asyncio.Event()
            self._ready_event = asyncio.Event()
            self._start_error = None
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run())
        task = self._task
        await self._ready_event.wait()
        error = self._start_error
        if error is not None:
            self._start_error = None
            raise error
        if task is not None and task.done():
            task.result()

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._stop_event.set()
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._stop_event = asyncio.Event()
            self._ready_event = asyncio.Event()
            self._start_error = None

    async def _run(self) -> None:
        track = PipelineVideoTrack(self._camera, self._pipeline, fps=self._fps)
        pc = _RTCPeerConnection()
        pc.addTrack(track)

        try:
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await _wait_for_ice_completion(pc, timeout=self._config.timeout)
            if pc.localDescription is None:  # pragma: no cover - defensive guard
                raise RuntimeError("WebRTC offer missing after ICE gathering")
            answer = await self._client.publish(pc.localDescription)
            await pc.setRemoteDescription(answer)
            self._ready_event.set()
            await self._stop_event.wait()
        except asyncio.CancelledError:
            self._ready_event.set()
            raise
        except Exception as exc:
            self._start_error = exc
            self._ready_event.set()
            logger.exception("MediaMTX publisher failed", exc_info=True)
            raise
        finally:
            track.stop()
            with contextlib.suppress(Exception):
                await pc.close()
            self._task = None


class WebRTCManager:
    """Proxy WebRTC signalling through MediaMTX and manage the publisher lifecycle."""

    def __init__(
        self,
        *,
        camera: BaseCamera,
        pipeline: FramePipeline,
        mediamtx_url: str = "http://127.0.0.1:8889",
        stream_name: str = "revcam",
        fps: int = 30,
        client: MediaMTXClient | None = None,
    ) -> None:
        _ensure_aiortc_available()
        self.camera = camera
        self.pipeline = pipeline
        self._config = MediaMTXConfig(base_url=mediamtx_url, stream_name=stream_name)
        if client is None:
            _ensure_httpx_available()
            self._client = MediaMTXClient(self._config)
        else:
            self._client = client
        self._fps = fps
        self._publisher = MediaMTXPublisher(
            camera,
            pipeline,
            self._client,
            config=self._config,
            fps=fps,
        )
        self._publisher_lock = asyncio.Lock()
        self._publisher_started = False

    async def _ensure_publisher(self) -> None:
        async with self._publisher_lock:
            if self._publisher_started:
                return
            await self._publisher.start()
            self._publisher_started = True

    async def set_camera(self, camera: BaseCamera) -> None:
        if camera is self.camera:
            return
        async with self._publisher_lock:
            await self._publisher.stop()
            self.camera = camera
            self._publisher = MediaMTXPublisher(
                camera,
                self.pipeline,
                self._client,
                config=self._config,
                fps=self._fps,
            )
            self._publisher_started = False

    async def handle_offer(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        try:
            await self._ensure_publisher()
        except MediaMTXError as exc:
            raise WebRTCError(f"Unable to start MediaMTX publisher: {exc}") from exc
        try:
            return await self._client.play(offer)
        except MediaMTXError as exc:
            raise WebRTCError(f"Unable to connect viewer via MediaMTX: {exc}") from exc

    async def shutdown(self) -> None:
        async with self._publisher_lock:
            await self._publisher.stop()
            self._publisher_started = False
        await self._client.close()


__all__ = ["MediaMTXConfig", "PipelineVideoTrack", "WebRTCManager"]
