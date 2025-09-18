"""WebRTC helpers for RevCam."""
from __future__ import annotations

import asyncio
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

    async def recv(self) -> VideoFrame:
        _ensure_aiortc_available()
        pts, time_base = await self.next_timestamp()
        frame = await self._camera.get_frame()
        processed = self._pipeline.process(frame)
        video_frame = VideoFrame.from_ndarray(processed, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


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
