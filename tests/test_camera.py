"""Tests for camera selection and error handling."""

from __future__ import annotations

import sys
import types

import pytest

from rev_cam.camera import (
    BaseCamera,
    CameraError,
    Picamera2Camera,
    SyntheticCamera,
    create_camera,
)


def test_picamera_initialisation_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Picamera setup errors should raise CameraError and close the device."""

    cleanup: dict[str, object] = {}

    class DummyCamera:
        def __init__(self) -> None:
            cleanup["instance"] = self
            self.stop_called = False
            self.close_called = False

        def create_video_configuration(self, **_: object) -> object:
            return object()

        def configure(self, _: object) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("camera start failed")

        def stop(self) -> None:
            self.stop_called = True

        def close(self) -> None:
            self.close_called = True

    module = types.SimpleNamespace(Picamera2=DummyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    with pytest.raises(CameraError):
        Picamera2Camera()

    dummy = cleanup["instance"]
    assert getattr(dummy, "close_called") is True
    assert getattr(dummy, "stop_called") is False


def test_create_camera_falls_back_when_picamera_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_camera should fall back to the synthetic camera when Picamera2 fails."""

    class BrokenCamera:
        def __init__(self) -> None:
            raise RuntimeError("picamera unavailable")

    module = types.SimpleNamespace(Picamera2=BrokenCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)
    monkeypatch.delenv("REVCAM_CAMERA", raising=False)

    camera = create_camera()

    assert isinstance(camera, SyntheticCamera)


def test_create_camera_prefers_opencv_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Picamera2 fails but OpenCV works we should use the OpenCV camera."""

    def broken_picamera() -> BaseCamera:
        raise CameraError("picamera unavailable")

    class DummyOpenCVCamera(BaseCamera):
        def __init__(self, index: int = 0) -> None:
            self.index = index

        async def get_frame(self):  # pragma: no cover - not used in test
            return [[0]]

    monkeypatch.setattr("rev_cam.camera.Picamera2Camera", broken_picamera)
    monkeypatch.setattr("rev_cam.camera.OpenCVCamera", DummyOpenCVCamera)
    monkeypatch.delenv("REVCAM_CAMERA", raising=False)

    camera = create_camera()

    assert isinstance(camera, DummyOpenCVCamera)

