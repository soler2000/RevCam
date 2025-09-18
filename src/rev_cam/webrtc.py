"""WebRTC helpers for RevCam."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from fractions import Fraction
from typing import Set

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame

from .video import VideoSource


class PipelineVideoTrack(VideoStreamTrack):
    """Video track that pulls frames from a camera via the processing pipeline."""

    def __init__(self, video_source: VideoSource, fps: int = 30) -> None:
        super().__init__()
        self._video_source = video_source
        self._frame_time = Fraction(1, fps)

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = await self._video_source.get_frame()
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


@dataclass
class WebRTCManager:
    """Manage WebRTC peer connections."""

    video_source: VideoSource
    connections: Set[RTCPeerConnection] = None

    def __post_init__(self) -> None:
        if self.connections is None:
            self.connections = set()

    async def handle_offer(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        pc = RTCPeerConnection()
        track = PipelineVideoTrack(self.video_source)
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
        await asyncio.gather(*(self._close_connection(pc) for pc in list(self.connections)), return_exceptions=True)
        self.connections.clear()


__all__ = ["PipelineVideoTrack", "WebRTCManager"]
