"""Tests for camera selection and error handling."""

from __future__ import annotations

import sys
import types

import pytest

from rev_cam.camera import (
    CameraError,
    Picamera2Camera,
    SyntheticCamera,
    create_camera,
    identify_camera,
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


def test_explicit_picamera_failure_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit PiCamera selection should bubble up initialisation errors."""

    class BrokenCamera:
        def __init__(self) -> None:
            raise RuntimeError("picamera unavailable")

    module = types.SimpleNamespace(Picamera2=BrokenCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    with pytest.raises(CameraError):
        create_camera("picamera")


def test_import_failure_surface_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import errors should include the underlying message for diagnostics."""

    import builtins

    original_import = builtins.__import__

    def raising_import(name: str, *args: object, **kwargs: object):
        if name == "picamera2":
            raise ImportError("numpy ABI mismatch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    with pytest.raises(CameraError) as excinfo:
        Picamera2Camera()

    assert "numpy ABI mismatch" in str(excinfo.value)


def test_identify_camera_returns_source() -> None:
    camera = SyntheticCamera()
    assert identify_camera(camera) == "synthetic"

