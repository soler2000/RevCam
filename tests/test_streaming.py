"""Tests for the WebRTC streaming helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
import numpy as np

from rev_cam import streaming
from rev_cam.camera import BaseCamera
from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline


class _StubCamera(BaseCamera):
    async def get_frame(self) -> np.ndarray:  # pragma: no cover - not used in tests
        raise AssertionError("get_frame should not be invoked during unit tests")


class _StubSimplejpeg:
    def __init__(self) -> None:
        self.images: list[object] = []
        self.kwargs: list[dict[str, object]] = []

    def encode_jpeg(self, image, **kwargs):  # pragma: no cover - exercised in tests
        self.images.append(image)
        self.kwargs.append(kwargs)
        return b"jpeg"


class _StubPillowImage:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.saved: list[dict[str, object]] = []

    def save(self, buffer, *, format: str, quality: int, optimize: bool) -> None:  # pragma: no cover - exercised in tests
        self.saved.append(
            {
                "buffer": buffer,
                "format": format,
                "quality": quality,
                "optimize": optimize,
            }
        )
        buffer.write(b"piljpeg")


def _enable_webrtc_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EncodingParameters:
        def __init__(self) -> None:
            self.maxBitrate: int | None = None
            self.maxFramerate: float | None = None

    class _Sender:
        def __init__(self) -> None:
            self.params = SimpleNamespace(encodings=[])

        def getParameters(self):  # pragma: no cover - exercised in tests
            return self.params

        def setParameters(self, params):  # pragma: no cover - exercised in tests
            self.params = params
            return None

    class _RTCSessionDescription:
        def __init__(self, sdp: str, type: str) -> None:
            self.sdp = sdp
            self.type = type

    class _PeerConnection:
        def __init__(self) -> None:
            self.connectionState = "new"
            self.iceConnectionState = "new"
            self.iceGatheringState = "complete"
            self._callbacks: dict[str, object] = {}
            self.sender = _Sender()
            self.localDescription: _RTCSessionDescription | None = None

        def addTrack(self, track):  # pragma: no cover - exercised in tests
            self.track = track
            return self.sender

        async def setRemoteDescription(self, description):  # pragma: no cover - exercised
            self.remoteDescription = description

        async def createAnswer(self):  # pragma: no cover - exercised in tests
            return _RTCSessionDescription("v=0\n", "answer")

        async def setLocalDescription(self, description):  # pragma: no cover
            self.localDescription = description

        def getSenders(self):  # pragma: no cover - exercised in tests
            return [self.sender]

        def on(self, event: str):  # pragma: no cover - exercised in tests
            def _register(callback):
                self._callbacks[event] = callback
                return callback

            return _register

        def close(self):  # pragma: no cover - exercised in tests
            self.connectionState = "closed"
            return None

    class _VideoFrame:
        @staticmethod
        def from_ndarray(*args, **kwargs):  # pragma: no cover - not exercised
            return SimpleNamespace(args=args, kwargs=kwargs)

    monkeypatch.setattr(streaming, "_AIORTC_AVAILABLE", True, raising=False)
    monkeypatch.setattr(streaming, "_AV_AVAILABLE", True, raising=False)
    monkeypatch.setattr(streaming, "_AIORTC_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(streaming, "_AV_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(streaming, "RTCRtpEncodingParameters", _EncodingParameters, raising=False)
    monkeypatch.setattr(streaming, "RTCPeerConnection", _PeerConnection, raising=False)
    monkeypatch.setattr(streaming, "RTCSessionDescription", _RTCSessionDescription, raising=False)
    monkeypatch.setattr(streaming, "av", SimpleNamespace(VideoFrame=_VideoFrame), raising=False)


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

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = streaming.encode_frame_to_jpeg(frame, quality=85)

    assert result == b"jpeg"
    assert len(stub.kwargs) == 1
    kwargs = stub.kwargs[0]

    assert kwargs["quality"] == 85
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


def test_encode_frame_uses_pillow_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    frames: list[_StubPillowImage] = []

    def _fromarray(array: np.ndarray) -> _StubPillowImage:
        image = _StubPillowImage(array)
        frames.append(image)
        return image

    monkeypatch.setattr(streaming, "simplejpeg", None, raising=False)
    monkeypatch.setattr(streaming, "_SIMPLEJPEG_IMPORT_ERROR", RuntimeError("boom"), raising=False)
    monkeypatch.setattr(
        streaming,
        "Image",
        SimpleNamespace(fromarray=_fromarray),
        raising=False,
    )
    monkeypatch.setattr(streaming, "_PIL_IMPORT_ERROR", None, raising=False)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = streaming.encode_frame_to_jpeg(frame, quality=85)

    assert result == b"piljpeg"
    assert len(frames) == 1
    saved = frames[0].saved
    assert len(saved) == 1
    save_info = saved[0]
    assert save_info["format"] == "JPEG"
    assert save_info["quality"] == 85
    assert save_info["optimize"] is True
    assert save_info["buffer"].getvalue() == b"piljpeg"


def test_streamer_apply_settings_updates_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_webrtc_dependencies(monkeypatch)

    pipeline = FramePipeline(lambda: Orientation())
    streamer = streaming.WebRTCStreamer(camera=_StubCamera(), pipeline=pipeline)

    original_interval = streamer._frame_interval
    streamer.apply_settings(fps=10)
    assert streamer.fps == 10
    assert streamer._frame_interval == pytest.approx(0.1, rel=1e-6)
    assert streamer._frame_interval != original_interval

    streamer.apply_settings(bitrate=2_000_000)
    assert streamer.bitrate == 2_000_000

    with pytest.raises(ValueError):
        streamer.apply_settings(fps=0)


@pytest.mark.asyncio
async def test_streamer_create_session_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_webrtc_dependencies(monkeypatch)

    pipeline = FramePipeline(lambda: Orientation())
    streamer = streaming.WebRTCStreamer(camera=_StubCamera(), pipeline=pipeline)

    answer = await streamer.create_session({"sdp": "v=0\n", "type": "offer"})

    assert isinstance(answer, streaming.SessionDescription)
    assert answer.type == "answer"
    assert "v=0" in answer.sdp
