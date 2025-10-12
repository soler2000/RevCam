"""Tests for the MJPEG streaming helpers."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

pytest.importorskip("numpy")
import numpy as np

from rev_cam import streaming
from rev_cam.camera import BaseCamera
from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline
import rev_cam.video_encoding as video_encoding
from rev_cam.video_encoding import select_h264_backend


class _StubCamera(BaseCamera):
    async def get_frame(self) -> np.ndarray:  # pragma: no cover - not used in tests
        raise AssertionError("get_frame should not be invoked during encoding tests")


class _StubSimplejpeg:
    def __init__(self) -> None:
        self.images: list[object] = []
        self.kwargs: list[dict[str, object]] = []

    def encode_jpeg(self, image, **kwargs):  # pragma: no cover - exercised in tests
        self.images.append(image)
        self.kwargs.append(kwargs)
        return b"jpeg"


@pytest.mark.parametrize(
    "supported_kwargs, expected_flags",
    [
        (set(), {"fastdct": False, "fastupsample": False}),
        (
            {"quality", "colorspace", "fastdct", "fastupsample"},
            {"fastdct": True, "fastupsample": True},
        ),
    ],
)
def test_encode_frame_respects_simplejpeg_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    supported_kwargs: set[str],
    expected_flags: dict[str, bool],
) -> None:
    stub = _StubSimplejpeg()
    monkeypatch.setattr(streaming, "simplejpeg", stub, raising=False)
    monkeypatch.setattr(streaming, "_SIMPLEJPEG_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(streaming, "_SIMPLEJPEG_ENCODE_KWARGS", supported_kwargs, raising=False)

    pipeline = FramePipeline(lambda: Orientation())
    streamer = streaming.MJPEGStreamer(camera=_StubCamera(), pipeline=pipeline)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = streamer._encode_frame(frame)

    assert result == b"jpeg"
    assert len(stub.kwargs) == 1
    kwargs = stub.kwargs[0]

    assert kwargs["quality"] == streamer.jpeg_quality
    assert kwargs["colorspace"] == "RGB"
    for name, expected in expected_flags.items():
        assert (name in kwargs) is expected


def test_encode_frame_copies_non_contiguous_input(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSimplejpeg()
    monkeypatch.setattr(streaming, "simplejpeg", stub, raising=False)
    monkeypatch.setattr(streaming, "_SIMPLEJPEG_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(streaming, "_SIMPLEJPEG_ENCODE_KWARGS", set(), raising=False)

    original = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))
    rotated = np.rot90(original, 2)
    assert not rotated.flags["C_CONTIGUOUS"]

    streaming.encode_frame_to_jpeg(rotated, quality=85)

    assert len(stub.images) == 1
    encoded = stub.images[0]
    assert encoded.flags["C_CONTIGUOUS"]
    assert np.array_equal(encoded, rotated)


def test_streamer_apply_settings_updates_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSimplejpeg()
    monkeypatch.setattr(streaming, "simplejpeg", stub, raising=False)
    monkeypatch.setattr(streaming, "_SIMPLEJPEG_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(streaming, "_SIMPLEJPEG_ENCODE_KWARGS", set(), raising=False)

    pipeline = FramePipeline(lambda: Orientation())
    streamer = streaming.MJPEGStreamer(camera=_StubCamera(), pipeline=pipeline)

    original_interval = streamer._frame_interval
    streamer.apply_settings(fps=10)
    assert streamer.fps == 10
    assert streamer._frame_interval == pytest.approx(0.1, rel=1e-6)
    assert streamer._frame_interval != original_interval

    streamer.apply_settings(jpeg_quality=150)
    assert streamer.jpeg_quality == 100

    streamer.apply_settings(jpeg_quality=0)
    assert streamer.jpeg_quality == 1

    with pytest.raises(ValueError):
        streamer.apply_settings(fps=0)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_pipeline_video_track_returns_video_frame(
    monkeypatch: pytest.MonkeyPatch, anyio_backend
) -> None:
    pytest.importorskip("aiortc")
    pytest.importorskip("av")

    class _Camera(BaseCamera):
        async def get_frame(self) -> np.ndarray:
            return np.full((2, 2, 3), 128, dtype=np.uint8)

    camera = _Camera()
    pipeline = FramePipeline(lambda: Orientation())
    backend, _ = select_h264_backend("libx264")
    if backend is None:
        pytest.skip("libx264 encoder unavailable")
    manager = SimpleNamespace(camera=camera, pipeline=pipeline, fps=30, _tracks=set())

    track = streaming.PipelineVideoTrack(manager, backend)
    packet = await track.recv()

    assert hasattr(packet, "time_base")
    assert packet.time_base.denominator == 30
    payload = bytes(packet)
    assert payload.startswith(b"\x00\x00\x00\x01")

    track.stop()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_webrtc_encoder_emits_extradata(anyio_backend) -> None:
    pytest.importorskip("aiortc")
    pytest.importorskip("av")

    backend, _ = select_h264_backend("libx264")
    if backend is None:
        pytest.skip("libx264 encoder unavailable")

    encoder = streaming.WebRTCEncoder(backend, fps=30)
    frame = np.full((2, 2, 3), 64, dtype=np.uint8)

    encoder.encode(frame)

    packets = []
    while encoder.has_packets():
        packets.append(encoder.pop_packet())

    assert packets, "encoder did not output packets"
    first = packets[0]
    assert first is not None
    assert first.pts is not None
    assert first.dts == first.pts
    assert first.time_base.denominator == 30
    assert bytes(first).startswith(b"\x00\x00\x00\x01")



_H264_OFFER = "\r\n".join(
    [
        "v=0",
        "o=- 0 0 IN IP4 127.0.0.1",
        "s=-",
        "t=0 0",
        "a=group:BUNDLE 0",
        "m=video 9 UDP/TLS/RTP/SAVPF 96 97",
        "c=IN IP4 0.0.0.0",
        "a=mid:0",
        "a=sendrecv",
        "a=rtpmap:96 H264/90000",
        "a=fmtp:96 packetization-mode=1;profile-level-id=42e01f;level-asymmetry-allowed=1",
        "a=rtpmap:97 VP8/90000",
    ]
)


_VP8_ONLY_OFFER = "\r\n".join(
    [
        "v=0",
        "o=- 0 0 IN IP4 127.0.0.1",
        "s=-",
        "t=0 0",
        "a=group:BUNDLE 0",
        "m=video 9 UDP/TLS/RTP/SAVPF 97",
        "c=IN IP4 0.0.0.0",
        "a=mid:0",
        "a=sendrecv",
        "a=rtpmap:97 VP8/90000",
    ]
)


class _StubPipelineTrack:
    kind = "video"

    def __init__(self, manager, backend):
        self._manager = manager
        self.backend = backend
        self.stopped = False
        manager._tracks.add(self)

    def stop(self) -> None:
        if not self.stopped:
            self.stopped = True
            self._manager._tracks.discard(self)


class _StubTransceiver:
    def __init__(self) -> None:
        self.kind = "video"
        self.preferences: tuple[object, ...] | None = None

    def setCodecPreferences(self, codecs) -> None:  # noqa: N802 - matching aiortc API
        self.preferences = tuple(codecs)


class _StubPeerConnection:
    def __init__(self) -> None:
        self.transceiver = _StubTransceiver()
        self._local_description = None
        self.connectionState = "new"
        self._closed = False

    def addTrack(self, track):  # noqa: N802 - matching aiortc API
        self.track = track
        return SimpleNamespace()

    def getTransceivers(self):
        return [self.transceiver]

    async def setRemoteDescription(self, description):  # pragma: no cover - trivial
        self.remote_description = description

    async def createAnswer(self):  # noqa: N802 - matching aiortc API
        return SimpleNamespace(sdp="answer", type="answer")

    async def setLocalDescription(self, description):  # noqa: N802 - matching aiortc API
        self._local_description = description

    @property
    def localDescription(self):  # noqa: N802 - matching aiortc API
        return self._local_description

    async def close(self) -> None:
        self._closed = True

    def on(self, _event):  # pragma: no cover - behaviour trivial
        def _wrapper(callback):
            return callback

        return _wrapper


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_webrtc_session_enforces_h264_codec(monkeypatch: pytest.MonkeyPatch, anyio_backend) -> None:
    pytest.importorskip("aiortc")
    pytest.importorskip("av")

    backend = video_encoding.H264EncoderBackend(
        key="libx264", codec="libx264", label="libx264", hardware=False
    )
    monkeypatch.setattr(video_encoding, "select_h264_backend", lambda _choice: (backend, (backend.codec,)))
    monkeypatch.setattr(streaming, "select_h264_backend", lambda _choice: (backend, (backend.codec,)))
    monkeypatch.setattr(streaming, "PipelineVideoTrack", _StubPipelineTrack)
    monkeypatch.setattr(streaming, "RTCPeerConnection", _StubPeerConnection)

    codecs = [
        SimpleNamespace(mimeType="video/VP8"),
        SimpleNamespace(mimeType="video/H264"),
        SimpleNamespace(mimeType="video/H264"),
    ]

    calls: list[str] = []

    class _StubSender:
        @staticmethod
        def getCapabilities(kind):  # noqa: N802 - matching aiortc API
            calls.append(kind)
            return SimpleNamespace(codecs=codecs)

    monkeypatch.setattr(streaming, "RTCRtpSender", _StubSender)

    manager = streaming.WebRTCManager(
        camera=_StubCamera(),
        pipeline=FramePipeline(lambda: Orientation()),
        fps=30,
        peer_connection_factory=_StubPeerConnection,
    )

    answer = await manager.create_session(sdp=_H264_OFFER, offer_type="offer")

    assert answer.sdp == "answer"
    assert calls == ["video"]
    session = next(iter(manager._sessions))
    transceiver = session.pc.transceiver  # type: ignore[attr-defined]
    assert transceiver.preferences == (codecs[1], codecs[2])


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_webrtc_session_rejects_non_h264_offer(monkeypatch: pytest.MonkeyPatch, anyio_backend) -> None:
    pytest.importorskip("aiortc")
    pytest.importorskip("av")

    backend = video_encoding.H264EncoderBackend(
        key="libx264", codec="libx264", label="libx264", hardware=False
    )
    monkeypatch.setattr(video_encoding, "select_h264_backend", lambda _choice: (backend, (backend.codec,)))
    monkeypatch.setattr(streaming, "select_h264_backend", lambda _choice: (backend, (backend.codec,)))
    monkeypatch.setattr(streaming, "PipelineVideoTrack", _StubPipelineTrack)
    monkeypatch.setattr(streaming, "RTCRtpSender", SimpleNamespace(getCapabilities=lambda *_: SimpleNamespace(codecs=[])))
    monkeypatch.setattr(streaming, "RTCPeerConnection", _StubPeerConnection)

    manager = streaming.WebRTCManager(
        camera=_StubCamera(),
        pipeline=FramePipeline(lambda: Orientation()),
        fps=30,
        peer_connection_factory=_StubPeerConnection,
    )

    with pytest.raises(RuntimeError, match="does not include an H\\.264"):
        await manager.create_session(sdp=_VP8_ONLY_OFFER, offer_type="offer")


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_webrtc_session_is_hashable(anyio_backend) -> None:
    manager = SimpleNamespace(_discard_session=lambda *_: None)
    session = streaming._WebRTCSession(manager, SimpleNamespace(), SimpleNamespace())

    sessions = {session}

    assert session in sessions


def test_select_h264_backend_prefers_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    def _probe(backend):
        order.append(backend.codec)
        return backend.codec == "libx264"

    monkeypatch.setattr(video_encoding, "_probe_backend", _probe, raising=False)

    backend, attempted = video_encoding.select_h264_backend("auto")
    assert backend is not None
    assert backend.codec == "libx264"
    assert attempted[0] == "h264_v4l2m2m"


def test_select_h264_backend_handles_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def _probe(_backend):
        return False

    monkeypatch.setattr(video_encoding, "_probe_backend", _probe, raising=False)

    backend, attempted = video_encoding.select_h264_backend("v4l2m2m")
    assert backend is None
    assert "h264_v4l2m2m" in attempted
    assert "libx264" in attempted
