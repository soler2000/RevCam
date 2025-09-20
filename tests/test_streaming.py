"""Tests for the MJPEG streaming helpers."""

from __future__ import annotations

import pytest

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
        self.calls: list[dict[str, object]] = []

    def encode_jpeg(self, image, **kwargs):  # pragma: no cover - exercised in tests
        self.calls.append(kwargs)
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
    assert len(stub.calls) == 1
    kwargs = stub.calls[0]

    assert kwargs["quality"] == streamer.jpeg_quality
    assert kwargs["colorspace"] == "RGB"
    for name, expected in expected_flags.items():
        assert (name in kwargs) is expected


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
