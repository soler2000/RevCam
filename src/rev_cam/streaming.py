"""WebRTC streaming helpers for RevCam."""

from __future__ import annotations

import asyncio
import io
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .camera import BaseCamera
from .pipeline import FramePipeline

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependency availability varies by platform
    import simplejpeg
except Exception as exc:  # pragma: no cover - dependency availability varies
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
    from PIL import Image
except Exception as exc:  # pragma: no cover - dependency availability varies
    Image = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency availability varies
    _PIL_IMPORT_ERROR = None


try:  # pragma: no cover - dependency availability varies by platform
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamTrack as _MediaStreamTrackBase
    from aiortc.rtp import RTCRtpEncodingParameters
except ImportError as exc:  # pragma: no cover - dependency availability varies
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    RTCRtpEncodingParameters = None  # type: ignore[assignment]

    class _MediaStreamTrackBase:  # type: ignore[override]
        """Fallback base class when aiortc is unavailable."""

        kind: str = "video"

        def __init__(self) -> None:
            self.readyState = "ended"

        async def recv(self):  # pragma: no cover - defensive guard
            raise RuntimeError(
                "aiortc is required for WebRTC streaming"
            ) from exc

        def stop(self) -> None:  # pragma: no cover - defensive guard
            self.readyState = "ended"

    _AIORTC_AVAILABLE = False
    _AIORTC_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency availability varies
    _AIORTC_AVAILABLE = True
    _AIORTC_IMPORT_ERROR = None


try:  # pragma: no cover - dependency availability varies by platform
    import av
except ImportError as exc:  # pragma: no cover - dependency availability varies

    class _AvModuleStub:  # pragma: no cover - defensive guard
        class VideoFrame:  # pragma: no cover - defensive guard
            @staticmethod
            def from_ndarray(*args, **kwargs):
                raise RuntimeError("PyAV is required for WebRTC streaming") from exc

    av = _AvModuleStub()
    _AV_AVAILABLE = False
    _AV_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency availability varies
    _AV_AVAILABLE = True
    _AV_IMPORT_ERROR = None


@dataclass(slots=True)
class SessionDescription:
    """Simple representation of an SDP session description."""

    sdp: str
    type: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SessionDescription":
        try:
            sdp_raw = data["sdp"]
            type_raw = data["type"]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Session description must include 'sdp' and 'type'") from exc
        if not isinstance(sdp_raw, str) or not sdp_raw.strip():
            raise ValueError("Session description SDP must be a non-empty string")
        if not isinstance(type_raw, str) or not type_raw.strip():
            raise ValueError("Session description type must be a non-empty string")
        return cls(sdp=sdp_raw, type=type_raw)

    @classmethod
    def from_rtc(
        cls, description: RTCSessionDescription | None
    ) -> "SessionDescription":
        if description is None:
            raise ValueError("RTC session description missing")
        return cls(sdp=description.sdp, type=description.type)

    def to_rtc(self) -> RTCSessionDescription:
        if not _AIORTC_AVAILABLE or RTCSessionDescription is None:
            raise RuntimeError(
                "aiortc is required for WebRTC streaming"
            ) from _AIORTC_IMPORT_ERROR
        return RTCSessionDescription(sdp=self.sdp, type=self.type)

    def to_dict(self) -> dict[str, str]:
        return {"sdp": self.sdp, "type": self.type}


def _normalise_frame(frame: np.ndarray | list) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError("Expected an RGB frame for encoding")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array


def _encode_with_simplejpeg(array: np.ndarray, quality: int) -> bytes:
    if simplejpeg is None:  # pragma: no cover - dependency availability varies
        raise RuntimeError(
            "simplejpeg is required for JPEG encoding"
        ) from _SIMPLEJPEG_IMPORT_ERROR

    encode_kwargs: dict[str, object] = {
        "quality": quality,
        "colorspace": "RGB",
    }
    if "fastdct" in _SIMPLEJPEG_ENCODE_KWARGS:
        encode_kwargs["fastdct"] = True
    if "fastupsample" in _SIMPLEJPEG_ENCODE_KWARGS:
        encode_kwargs["fastupsample"] = True

    return simplejpeg.encode_jpeg(array, **encode_kwargs)


def _encode_with_pillow(array: np.ndarray, quality: int) -> bytes:
    if Image is None:  # pragma: no cover - dependency availability varies
        raise RuntimeError("Pillow is required for JPEG encoding") from _PIL_IMPORT_ERROR

    buffer = io.BytesIO()
    Image.fromarray(array).save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
    )
    return buffer.getvalue()


def encode_frame_to_jpeg(
    frame: np.ndarray | list,
    *,
    quality: int,
) -> bytes:
    """Encode an RGB frame into JPEG bytes using the configured quality."""

    array = _normalise_frame(frame)
    quality_int = int(quality)

    if simplejpeg is not None:
        return _encode_with_simplejpeg(array, quality_int)

    if Image is not None:
        return _encode_with_pillow(array, quality_int)

    error = _SIMPLEJPEG_IMPORT_ERROR or _PIL_IMPORT_ERROR
    raise RuntimeError("JPEG encoding requires simplejpeg or Pillow") from error


@dataclass
class _CameraStreamTrack(_MediaStreamTrackBase):
    """Media stream track that pulls frames from the camera pipeline."""

    streamer: "WebRTCStreamer"

    def __post_init__(self) -> None:
        self.kind = "video"
        self._last_pts = 0

    async def recv(self):
        if not _AV_AVAILABLE:
            raise RuntimeError("PyAV is required for WebRTC streaming") from _AV_IMPORT_ERROR

        start = time.perf_counter()
        frame = await self.streamer.camera.get_frame()
        try:
            processed = self.streamer.pipeline.process(frame)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to process frame for WebRTC stream")
            processed = frame

        array = _normalise_frame(processed)
        video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")

        self._last_pts += 1
        fps = max(1, int(self.streamer.fps))
        video_frame.pts = self._last_pts
        video_frame.time_base = (1, fps)

        elapsed = time.perf_counter() - start
        delay = self.streamer._frame_interval - elapsed
        if delay > 0:
            await asyncio.sleep(delay)

        return video_frame


@dataclass
class WebRTCStreamer:
    """Encode frames from a camera into a WebRTC stream."""

    camera: BaseCamera
    pipeline: FramePipeline
    fps: int = 20
    bitrate: int = 1_500_000
    _frame_interval: float = field(init=False)
    _peers: dict[object, _CameraStreamTrack] = field(init=False, default_factory=dict)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.bitrate <= 0:
            raise ValueError("bitrate must be positive")
        if not _AIORTC_AVAILABLE:
            raise RuntimeError(
                "aiortc is required for WebRTC streaming"
            ) from _AIORTC_IMPORT_ERROR
        if not _AV_AVAILABLE:
            raise RuntimeError(
                "PyAV is required for WebRTC streaming"
            ) from _AV_IMPORT_ERROR
        self._frame_interval = 1.0 / float(self.fps)

    def apply_settings(
        self,
        *,
        fps: int | None = None,
        bitrate: int | None = None,
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

        if bitrate is not None:
            new_bitrate = int(bitrate)
            if new_bitrate <= 0:
                raise ValueError("bitrate must be positive")
            self.bitrate = new_bitrate

        if update_interval:
            self._frame_interval = 1.0 / float(self.fps)

    async def create_session(
        self, offer: Mapping[str, object] | SessionDescription
    ) -> SessionDescription:
        """Accept a remote SDP offer and return the local answer."""

        if isinstance(offer, Mapping):
            session = SessionDescription.from_mapping(offer)
        elif isinstance(offer, SessionDescription):
            session = offer
        else:
            raise TypeError("offer must be a mapping or SessionDescription")

        peer = RTCPeerConnection()
        track = _CameraStreamTrack(self)
        sender = peer.addTrack(track)

        await self._configure_sender(sender)

        await peer.setRemoteDescription(session.to_rtc())
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)

        await self._register_peer(peer, track)
        await self._wait_for_ice(peer)

        return SessionDescription.from_rtc(peer.localDescription)

    async def aclose(self) -> None:
        """Close all active peer connections."""

        async with self._lock:
            peers = list(self._peers.items())
            self._peers.clear()

        for connection, track in peers:
            try:
                track.stop()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to stop WebRTC track cleanly", exc_info=True)
            try:
                result = connection.close()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to close WebRTC peer cleanly", exc_info=True)

    async def _configure_sender(self, sender) -> None:
        if RTCRtpEncodingParameters is None:
            return

        try:
            params = sender.getParameters()
        except AttributeError:  # pragma: no cover - defensive guard
            return

        encodings = list(getattr(params, "encodings", []))
        if not encodings:
            encodings = [RTCRtpEncodingParameters()]

        for encoding in encodings:
            try:
                encoding.maxBitrate = int(self.bitrate)
            except AttributeError:  # pragma: no cover - defensive guard
                pass
            try:
                encoding.maxFramerate = float(self.fps)
            except AttributeError:  # pragma: no cover - defensive guard
                pass

        try:
            params.encodings = encodings
        except AttributeError:  # pragma: no cover - defensive guard
            return

        result = sender.setParameters(params)
        if inspect.isawaitable(result):
            await result

    async def _register_peer(self, peer, track: _CameraStreamTrack) -> None:
        async with self._lock:
            self._peers[peer] = track

        @peer.on("connectionstatechange")
        async def _on_connection_state_change():
            state = getattr(peer, "connectionState", None)
            if state in {"failed", "closed"}:
                await self._discard_peer(peer)

        @peer.on("iceconnectionstatechange")
        async def _on_ice_connection_state_change():
            state = getattr(peer, "iceConnectionState", None)
            if state in {"failed", "closed", "disconnected"}:
                await self._discard_peer(peer)

    async def _wait_for_ice(self, peer) -> None:
        if getattr(peer, "iceGatheringState", "complete") == "complete":
            return

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        @peer.on("icegatheringstatechange")
        def _on_ice_gathering_state_change():
            state = getattr(peer, "iceGatheringState", None)
            if state == "complete" and not future.done():
                future.set_result(None)

        try:
            await asyncio.wait_for(future, timeout=5.0)
        except asyncio.TimeoutError:  # pragma: no cover - network conditions vary
            logger.debug("Timed out waiting for ICE gathering to complete")

    async def _discard_peer(self, peer) -> None:
        async with self._lock:
            track = self._peers.pop(peer, None)

        if track is not None:
            try:
                track.stop()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to stop WebRTC track cleanly", exc_info=True)

        try:
            result = peer.close()
            if inspect.isawaitable(result):
                await result
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to close WebRTC peer cleanly", exc_info=True)


__all__ = ["WebRTCStreamer", "SessionDescription", "encode_frame_to_jpeg"]

