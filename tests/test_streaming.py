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
    manager = SimpleNamespace(camera=camera, pipeline=pipeline, fps=30, _tracks=set())

    track = streaming.PipelineVideoTrack(manager)
    frame = await track.recv()

    assert frame.pts is not None
    assert frame.time_base.denominator == 90_000
    array = frame.to_ndarray(format="rgb24")
    assert array.shape == (2, 2, 3)
    assert np.all(array == 128)

    track.stop()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_webrtc_session_is_hashable(anyio_backend) -> None:
    manager = SimpleNamespace(_discard_session=lambda *_: None)
    session = streaming._WebRTCSession(manager, SimpleNamespace(), SimpleNamespace())

    sessions = {session}

    assert session in sessions
