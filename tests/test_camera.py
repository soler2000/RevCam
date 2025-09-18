"""Tests for camera selection and error handling."""

from __future__ import annotations

import sys
import types

import pytest

from rev_cam.camera import (
    CameraDependencyError,
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


def test_create_camera_falls_back_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Picamera2 dependency should trigger the synthetic fallback."""

    class MissingDependencyCamera:
        def __init__(self) -> None:
            raise CameraDependencyError("picamera2 missing")

    monkeypatch.setattr("rev_cam.camera.Picamera2Camera", MissingDependencyCamera)
    monkeypatch.delenv("REVCAM_CAMERA", raising=False)

    camera = create_camera()

    assert isinstance(camera, SyntheticCamera)


def test_create_camera_propagates_runtime_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Picamera2 runtime errors should not be hidden behind the fallback."""

    class BrokenCamera:
        def __init__(self) -> None:
            raise CameraError("picamera runtime failure")

    monkeypatch.setattr("rev_cam.camera.Picamera2Camera", BrokenCamera)
    monkeypatch.delenv("REVCAM_CAMERA", raising=False)

    with pytest.raises(CameraError):
        create_camera()

