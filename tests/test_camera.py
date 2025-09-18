"""Tests for camera selection and error handling."""

from __future__ import annotations

import sys
import types

import pytest

from rev_cam import camera as camera_module
from rev_cam.camera import (
    CameraError,
    OpenCVCamera,
    Picamera2Camera,
    SyntheticCamera,
    create_camera,
    get_camera_status,
)


@pytest.fixture(autouse=True)
def reset_camera_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the cached camera selection is cleared between tests."""

    monkeypatch.setattr(camera_module, "_LAST_SELECTION", None)


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


def test_create_camera_falls_back_when_picamera_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic frames are used when Picamera2 and OpenCV are unavailable."""

    monkeypatch.delenv("REVCAM_CAMERA", raising=False)
    sys.modules.pop("picamera2", None)
    sys.modules.pop("cv2", None)

    camera = create_camera()

    assert isinstance(camera, SyntheticCamera)
    status = get_camera_status()
    assert status["requested"] == "auto"
    assert status["active_backend"] == "synthetic"
    assert status["error"] is None
    assert status["fallbacks"] == [
        "Picamera2: picamera2 is not available",
        "OpenCV: OpenCV is not installed",
    ]


def test_create_camera_uses_opencv_when_picamera_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback to OpenCV when Picamera2 cannot be imported but cv2 is installed."""

    monkeypatch.delenv("REVCAM_CAMERA", raising=False)
    sys.modules.pop("picamera2", None)

    class DummyCapture:
        def __init__(self, index: int) -> None:
            self.index = index

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, str]:
            return True, "frame"

        def release(self) -> None:
            return None

    def dummy_video_capture(index: int) -> DummyCapture:
        return DummyCapture(index)

    def dummy_cvt_color(frame: str, _: object) -> str:
        return frame

    module = types.SimpleNamespace(
        VideoCapture=dummy_video_capture,
        COLOR_BGR2RGB=1,
        cvtColor=dummy_cvt_color,
    )
    monkeypatch.setitem(sys.modules, "cv2", module)

    camera = create_camera()

    assert isinstance(camera, OpenCVCamera)
    status = get_camera_status()
    assert status["requested"] == "auto"
    assert status["active_backend"] == "opencv"
    assert status["error"] is None
    assert status["fallbacks"] == ["Picamera2: picamera2 is not available"]


def test_create_camera_auto_falls_back_when_picamera_initialisation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode should fall back to OpenCV when Picamera2 initialisation fails."""

    class BrokenCamera:
        def __init__(self) -> None:
            raise RuntimeError("picamera unavailable")

    module = types.SimpleNamespace(Picamera2=BrokenCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)
    monkeypatch.delenv("REVCAM_CAMERA", raising=False)

    class DummyCapture:
        def __init__(self, index: int) -> None:
            self.index = index

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, str]:
            return True, "frame"

        def release(self) -> None:
            return None

    def dummy_video_capture(index: int) -> DummyCapture:
        return DummyCapture(index)

    def dummy_cvt_color(frame: str, _: object) -> str:
        return frame

    cv2_module = types.SimpleNamespace(
        VideoCapture=dummy_video_capture,
        COLOR_BGR2RGB=1,
        cvtColor=dummy_cvt_color,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2_module)

    camera = create_camera()

    assert isinstance(camera, OpenCVCamera)
    status = get_camera_status()
    assert status["requested"] == "auto"
    assert status["active_backend"] == "opencv"
    assert status["error"] is None
    assert status["fallbacks"] == [
        "Picamera2: Failed to initialise Picamera2 camera",
    ]


def test_create_camera_raises_when_picamera_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit Picamera2 configuration should surface initialisation failures."""

    class BrokenCamera:
        def __init__(self) -> None:
            raise RuntimeError("picamera unavailable")

    module = types.SimpleNamespace(Picamera2=BrokenCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)
    monkeypatch.setenv("REVCAM_CAMERA", "picamera")

    with pytest.raises(CameraError):
        create_camera()

    status = get_camera_status()
    assert status["requested"] == "picamera"
    assert status["active_backend"] is None
    assert status["error"] == "Failed to initialise Picamera2 camera"
    assert status["fallbacks"] == []


def test_create_camera_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid configuration values surface as camera errors."""

    monkeypatch.setenv("REVCAM_CAMERA", "unknown")

    with pytest.raises(CameraError) as excinfo:
        create_camera()

    assert "unknown" in str(excinfo.value)
    status = get_camera_status()
    assert status["requested"] == "unknown"
    assert status["active_backend"] is None
    assert status["error"] == "Unknown camera backend: 'unknown'"
    assert status["fallbacks"] == []

