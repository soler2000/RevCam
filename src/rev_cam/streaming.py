"""Streaming helpers for RevCam."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from typing import AsyncGenerator, Callable, Deque

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
    from aiortc import (
        MediaStreamTrack,
        RTCPeerConnection,
        RTCSessionDescription,
        sdp as aiortc_sdp,
    )
    from aiortc.mediastreams import MediaStreamError
    from aiortc.rtcrtpsender import RTCRtpSender
except ImportError as exc:  # pragma: no cover - dependency availability varies
    MediaStreamTrack = None  # type: ignore[assignment]
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    MediaStreamError = None  # type: ignore[assignment]
    aiortc_sdp = None  # type: ignore[assignment]
    RTCRtpSender = None  # type: ignore[assignment]
    _AIORTC_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency availability varies
    _AIORTC_IMPORT_ERROR = None

try:  # pragma: no cover - dependency availability varies by platform
    import av  # type: ignore
    from av import VideoFrame
    from av.video.frame import PictureType
    from av.packet import Packet as AvPacket
except ImportError as exc:  # pragma: no cover - dependency availability varies
    av = None  # type: ignore[assignment]
    VideoFrame = None  # type: ignore[assignment]
    PictureType = None  # type: ignore[assignment]
    _AV_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency availability varies
    _AV_IMPORT_ERROR = None

from .video_encoding import H264EncoderBackend, select_h264_backend


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


def _video_codecs_from_sdp(sdp_text: str) -> tuple[str, ...]:
    """Extract the declared video codec names from an SDP offer."""

    if aiortc_sdp is None:  # pragma: no cover - dependency guard
        raise RuntimeError("WebRTC SDP parsing is unavailable")

    try:
        session_description = aiortc_sdp.SessionDescription.parse(sdp_text)
    except Exception as exc:  # pragma: no cover - malformed SDP
        raise RuntimeError("Invalid WebRTC SDP offer") from exc

    codecs: list[str] = []
    video_sections = [media for media in session_description.media if media.kind == "video"]
    if not video_sections:
        raise RuntimeError("WebRTC offer is missing a video section")

    for media in video_sections:
        for codec in media.rtp.codecs:
            codecs.append(codec.name.upper())

    return tuple(codecs)


def _h264_codec_preferences() -> tuple[object, ...]:
    """Return the RTP codec capabilities corresponding to H.264."""

    if RTCRtpSender is None:  # pragma: no cover - dependency guard
        raise RuntimeError("aiortc RTP sender is unavailable")

    capabilities = RTCRtpSender.getCapabilities("video")
    codecs = tuple(
        codec for codec in capabilities.codecs if codec.mimeType.lower() == "video/h264"
    )
    if not codecs:
        raise RuntimeError("No H.264 codec support available")
    return codecs


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


class WebRTCEncoder:
    """Encode RGB frames into Annex B H.264 packets for WebRTC."""

    def __init__(
        self,
        backend: H264EncoderBackend,
        *,
        fps: int,
        disable_b_frames: bool = True,
    ) -> None:
        if av is None or VideoFrame is None or PictureType is None:  # pragma: no cover - dependency guard
            raise RuntimeError("PyAV is required for WebRTC encoding")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._backend = backend
        self._fps = int(fps)
        self._disable_b_frames = bool(disable_b_frames)
        self._time_base = Fraction(1, self._fps)
        self._frame_rate = Fraction(self._fps, 1)
        self._codec: "av.CodecContext" | None = None
        self._pending: Deque["av.Packet"] = deque()
        self._pts: int = 0
        self._force_keyframe: bool = True

    @staticmethod
    def _is_annex_b(data: bytes) -> bool:
        return data.startswith(b"\x00\x00\x00\x01") or data.startswith(b"\x00\x00\x01")

    def _emit_extradata(self, codec: "av.CodecContext") -> None:
        extradata = getattr(codec, "extradata", None)
        if not extradata:
            return
        if isinstance(extradata, memoryview):  # pragma: no cover - defensive
            extradata = extradata.tobytes()
        if not isinstance(extradata, (bytes, bytearray)):
            return

        annexb = self._extradata_to_annexb(bytes(extradata))
        if not annexb:
            return

        packet = AvPacket(annexb)
        packet.time_base = self._time_base
        packet.pts = self._pts
        packet.dts = self._pts
        self._pending.append(packet)

    def _extradata_to_annexb(self, data: bytes) -> bytes:
        if not data:
            return b""
        if self._is_annex_b(data):
            return data
        if len(data) < 7 or data[0] != 1:
            logger.debug("Unexpected H.264 extradata format for %s", self._backend.codec)
            return b""

        pos = 5
        sps_count = data[pos] & 0x1F
        pos += 1
        output = bytearray()
        for _ in range(sps_count):
            if pos + 2 > len(data):
                return b""
            length = int.from_bytes(data[pos : pos + 2], "big")
            pos += 2
            if pos + length > len(data):
                return b""
            output += b"\x00\x00\x00\x01" + data[pos : pos + length]
            pos += length

        if pos >= len(data):
            return bytes(output)

        pps_count = data[pos]
        pos += 1
        for _ in range(pps_count):
            if pos + 2 > len(data):
                return b""
            length = int.from_bytes(data[pos : pos + 2], "big")
            pos += 2
            if pos + length > len(data):
                return b""
            output += b"\x00\x00\x00\x01" + data[pos : pos + length]
            pos += length

        return bytes(output)

    def _codec_options(self) -> dict[str, str]:
        if self._backend.key == "libx264":
            return {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "profile": "baseline",
                "bf": "0",
            }
        return {}

    def _create_codec(self, width: int, height: int) -> "av.CodecContext":
        assert av is not None  # for type checkers
        codec = av.CodecContext.create(self._backend.codec, "w")
        codec.width = width
        codec.height = height
        codec.time_base = self._time_base
        codec.framerate = self._frame_rate
        codec.pix_fmt = "yuv420p"
        if self._disable_b_frames:
            try:
                codec.max_b_frames = 0
            except Exception:  # pragma: no cover - property may be read-only
                pass
        options = self._codec_options()
        if options:
            try:
                codec.options = options
            except Exception:  # pragma: no cover - some codecs do not expose options
                logger.debug(
                    "WebRTC encoder %s did not accept configuration options", self._backend.codec
                )
        try:
            codec.open()
        except Exception as exc:  # pragma: no cover - codec initialisation failure
            raise RuntimeError(f"Failed to open {self._backend.codec} encoder") from exc
        self._pending.clear()
        self._pts = 0
        self._force_keyframe = True
        self._emit_extradata(codec)
        return codec

    def _ensure_codec(self, width: int, height: int) -> "av.CodecContext":
        codec = self._codec
        if codec is None or codec.width != width or codec.height != height:
            codec = self._create_codec(width, height)
            self._codec = codec
        return codec

    def encode(self, rgb_frame: np.ndarray) -> None:
        codec = self._ensure_codec(int(rgb_frame.shape[1]), int(rgb_frame.shape[0]))
        if VideoFrame is None or PictureType is None:  # pragma: no cover - dependency guard
            raise RuntimeError("PyAV is required for WebRTC encoding")
        video_frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        yuv = video_frame.reformat(format="yuv420p")
        yuv.pts = self._pts
        yuv.time_base = self._time_base
        try:
            yuv.pict_type = PictureType.I if self._force_keyframe else PictureType.NONE
        except Exception:  # pragma: no cover - pict_type may be immutable
            pass
        self._force_keyframe = False
        self._pts += 1
        try:
            packets = codec.encode(yuv)
        except av.FFmpegError as exc:  # pragma: no cover - codec failure
            raise RuntimeError(f"WebRTC encoder {self._backend.codec} failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("WebRTC encoder failure") from exc
        for packet in packets:
            packet.time_base = codec.time_base
            if packet.pts is None:
                packet.pts = self._pts - 1
            if packet.dts is None:
                packet.dts = packet.pts
            self._pending.append(packet)

    def has_packets(self) -> bool:
        return bool(self._pending)

    def pop_packet(self):
        if self._pending:
            return self._pending.popleft()
        return None

    def request_keyframe(self) -> None:
        self._force_keyframe = True

    def update_fps(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        new_fps = int(fps)
        if new_fps == self._fps:
            return
        self._fps = new_fps
        self._time_base = Fraction(1, self._fps)
        self._frame_rate = Fraction(self._fps, 1)
        self._codec = None
        self._pending.clear()
        self._pts = 0
        self._force_keyframe = True

    def close(self) -> None:
        self._pending.clear()
        self._codec = None


@dataclass(frozen=True)
class _EncoderPreferences:
    config: str | None
    env: str | None
    cli: str | None

    def select(
        self, session: str | None = None
    ) -> tuple[H264EncoderBackend | None, tuple[str, ...], str]:
        attempted: list[str] = []

        def _extend(names: tuple[str, ...]) -> None:
            for name in names:
                if name not in attempted:
                    attempted.append(name)

        for source, choice in (
            ("session", session),
            ("cli", self.cli),
            ("env", self.env),
            ("config", self.config),
        ):
            if choice is None:
                continue
            backend, probed = select_h264_backend(choice)
            _extend(probed)
            if backend is not None:
                return backend, tuple(attempted), source
        backend, probed = select_h264_backend("auto")
        _extend(probed)
        return backend, tuple(attempted), "auto"


if MediaStreamTrack is not None and VideoFrame is not None and MediaStreamError is not None:

    class PipelineVideoTrack(MediaStreamTrack):
        """Expose camera frames via WebRTC."""

        kind = "video"

        def __init__(self, manager: "WebRTCManager", backend: H264EncoderBackend) -> None:
            super().__init__()
            self._manager = manager
            self._stopped = False
            self._frame_interval = 1.0 / float(manager.fps)
            self._encoder = WebRTCEncoder(backend, fps=manager.fps)
            self._backend = backend
            self._manager._tracks.add(self)

        def update_fps(self, fps: int) -> None:
            if fps <= 0:
                raise ValueError("fps must be positive")
            self._frame_interval = 1.0 / float(fps)
            try:
                self._encoder.update_fps(fps)
            except Exception:  # pragma: no cover - defensive logging
                logger.debug(
                    "Failed to update WebRTC encoder FPS for %s", self._backend.key, exc_info=True
                )

        def stop(self) -> None:  # pragma: no cover - exercised indirectly
            if self._stopped:
                return
            self._stopped = True
            self._manager._tracks.discard(self)
            try:
                self._encoder.close()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to close WebRTC encoder", exc_info=True)
            super().stop()

        async def recv(self):  # type: ignore[override]
            if self._stopped:
                raise MediaStreamError("Track has been stopped")

            if self._encoder.has_packets():
                packet = self._encoder.pop_packet()
                if packet is not None:
                    return packet

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

            try:
                self._encoder.encode(rgb)
            except RuntimeError as exc:  # pragma: no cover - codec failure
                logger.error("WebRTC encoder %s failed: %s", self._backend.codec, exc)
                raise MediaStreamError("Encoder failure") from exc
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Unexpected WebRTC encoder failure")
                raise MediaStreamError("Encoder failure") from exc

            packet = self._encoder.pop_packet()
            if packet is None:  # pragma: no cover - defensive guard
                raise MediaStreamError("Encoder produced no packets")

            elapsed = time.perf_counter() - iteration_start
            remaining = self._frame_interval - elapsed
            if remaining > 0:
                try:
                    await asyncio.sleep(remaining)
                except asyncio.CancelledError:  # pragma: no cover - cooperative exit
                    raise
            return packet


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
    encoder_config_choice: str | None = None
    encoder_env_choice: str | None = None
    encoder_cli_choice: str | None = None
    _sessions: set[_WebRTCSession] = field(init=False, default_factory=set)
    _tracks: set[PipelineVideoTrack] = field(init=False, default_factory=set)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _encoder_preferences: _EncoderPreferences = field(init=False)
    _default_backend: H264EncoderBackend | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if (
            MediaStreamTrack is None
            or RTCPeerConnection is None
            or RTCSessionDescription is None
            or MediaStreamError is None
            or VideoFrame is None
            or RTCRtpSender is None
            or aiortc_sdp is None
        ):  # pragma: no cover - dependency availability varies
            if _AIORTC_IMPORT_ERROR is not None:
                raise RuntimeError("aiortc is required for WebRTC streaming") from _AIORTC_IMPORT_ERROR
            if _AV_IMPORT_ERROR is not None:
                raise RuntimeError("PyAV is required for WebRTC streaming") from _AV_IMPORT_ERROR
            raise RuntimeError("WebRTC dependencies are unavailable")

        if self.peer_connection_factory is None:
            self.peer_connection_factory = RTCPeerConnection  # type: ignore[assignment]

        self._encoder_preferences = _EncoderPreferences(
            config=self.encoder_config_choice,
            env=self.encoder_env_choice,
            cli=self.encoder_cli_choice,
        )
        backend, attempted, source = self._encoder_preferences.select()
        if backend is None:
            raise RuntimeError("No usable WebRTC H.264 encoder available")
        self._default_backend = backend
        logger.info(
            "Selected WebRTC encoder backend %s via %s preference", backend.label, source
        )
        if attempted:
            logger.debug("WebRTC encoder probe order: %s", ", ".join(attempted))

    async def create_session(
        self, sdp: str, offer_type: str, *, encoder: str | None = None
    ) -> RTCSessionDescription:
        """Create a WebRTC session from a remote offer and return the answer."""

        if RTCSessionDescription is None or RTCPeerConnection is None:  # pragma: no cover
            raise RuntimeError("aiortc is required for WebRTC streaming")

        codecs_offered = _video_codecs_from_sdp(sdp)
        if "H264" not in codecs_offered:
            logger.debug("WebRTC offer advertised codecs: %s", ", ".join(codecs_offered))
            raise RuntimeError("WebRTC offer does not include an H.264 video codec")

        offer = RTCSessionDescription(sdp=sdp, type=offer_type)
        pc = self.peer_connection_factory()  # type: ignore[operator]

        if not isinstance(pc, RTCPeerConnection):
            raise TypeError("peer_connection_factory must return an RTCPeerConnection instance")
        backend, attempted, source = self._encoder_preferences.select(encoder)
        if backend is None:
            raise RuntimeError("No usable WebRTC H.264 encoder available")
        track = PipelineVideoTrack(self, backend)
        session = _WebRTCSession(self, pc, track)
        pc.addTrack(track)

        try:
            h264_preferences = _h264_codec_preferences()
        except RuntimeError:
            track.stop()
            await pc.close()
            raise

        video_transceivers = [
            transceiver for transceiver in pc.getTransceivers() if getattr(transceiver, "kind", None) == "video"
        ]
        if not video_transceivers:
            track.stop()
            await pc.close()
            raise RuntimeError("WebRTC peer connection did not create a video transceiver")

        for transceiver in video_transceivers:
            try:
                transceiver.setCodecPreferences(h264_preferences)
            except Exception as exc:
                track.stop()
                await pc.close()
                raise RuntimeError("Failed to configure H.264 codec for WebRTC session") from exc

        if encoder is not None:
            logger.info(
                "WebRTC session requested %s encoder; using %s via %s preference",
                encoder,
                backend.label,
                source,
            )
        elif backend != self._default_backend:
            logger.info(
                "WebRTC session switched to %s backend via %s preference",
                backend.label,
                source,
            )
        if attempted:
            logger.debug("WebRTC session encoder probe: %s", ", ".join(attempted))

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
        encoder: str | None = None,
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

        if encoder is not None and encoder != self.encoder_config_choice:
            self.encoder_config_choice = encoder
            self._encoder_preferences = _EncoderPreferences(
                config=self.encoder_config_choice,
                env=self.encoder_env_choice,
                cli=self.encoder_cli_choice,
            )
            backend, attempted, source = self._encoder_preferences.select()
            if backend is None:
                raise RuntimeError("No usable WebRTC H.264 encoder available")
            self._default_backend = backend
            logger.info(
                "Updated WebRTC encoder preference to %s via %s preference",
                backend.label,
                source,
            )
            if attempted:
                logger.debug(
                    "WebRTC encoder probe order after update: %s",
                    ", ".join(attempted),
                )

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
