"""Tests for camera selection and error handling."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

pytest.importorskip("numpy")
import numpy as np

import rev_cam.camera as camera_module

from rev_cam.camera import (
    CameraError,
    Picamera2Camera,
    SyntheticCamera,
    create_camera,
    identify_camera,
    diagnose_camera_conflicts,
    _ensure_picamera_allocator,
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


def test_picamera_failure_without_allocator_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing allocator attributes should be patched before closing."""

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
            raise RuntimeError("startup failed")

        def stop(self) -> None:
            self.stop_called = True

        def close(self) -> None:
            cleanup["close_called"] = True
            cleanup["allocator_present"] = hasattr(self, "allocator")

    module = types.SimpleNamespace(Picamera2=DummyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    with pytest.raises(CameraError):
        Picamera2Camera()

    dummy = cleanup["instance"]
    assert getattr(dummy, "stop_called") is False
    assert cleanup.get("close_called") is True
    assert cleanup.get("allocator_present") is True


def test_picamera_allocator_class_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allocator shim should attach to the Picamera2 class when init fails early."""

    class DummyCamera:
        def __init__(self) -> None:
            raise RuntimeError("initialisation failed")

    module = types.SimpleNamespace(Picamera2=DummyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    with pytest.raises(CameraError):
        Picamera2Camera()

    try:
        assert hasattr(DummyCamera, "allocator")
    finally:
        if hasattr(DummyCamera, "allocator"):
            delattr(DummyCamera, "allocator")


def test_picamera_allocator_injection_handles_custom_setattr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allocator patching should bypass restrictive ``__setattr__`` logic."""

    cleanup: dict[str, object] = {}

    class DummyCamera:
        def __init__(self) -> None:
            cleanup["instance"] = self
            object.__setattr__(self, "close_called", False)

        def __setattr__(self, name: str, value: object) -> None:
            if name == "allocator":
                raise AttributeError("allocator attribute disabled")
            object.__setattr__(self, name, value)

        def create_video_configuration(self, **_: object) -> object:
            return object()

        def configure(self, _: object) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("startup failed")

        def stop(self) -> None:
            return None

        def close(self) -> None:
            cleanup["close_called"] = True
            cleanup["allocator_present"] = hasattr(self, "allocator")

    module = types.SimpleNamespace(Picamera2=DummyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    with pytest.raises(CameraError):
        Picamera2Camera()

    assert cleanup.get("close_called") is True
    assert cleanup.get("allocator_present") is True


def test_picamera_allocator_injection_handles_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allocator patching should fall back to class attributes when needed."""

    cleanup: dict[str, object] = {}

    class DummyCamera:
        __slots__ = ("close_called",)

        def __init__(self) -> None:
            cleanup["instance"] = self
            self.close_called = False

        def create_video_configuration(self, **_: object) -> object:
            return object()

        def configure(self, _: object) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("startup failed")

        def stop(self) -> None:
            return None

        def close(self) -> None:
            cleanup["close_called"] = True
            cleanup["allocator_present"] = hasattr(self, "allocator")

    module = types.SimpleNamespace(Picamera2=DummyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    try:
        with pytest.raises(CameraError):
            Picamera2Camera()
    finally:
        if hasattr(DummyCamera, "allocator"):
            delattr(DummyCamera, "allocator")

    assert cleanup.get("close_called") is True
    assert cleanup.get("allocator_present") is True


def test_null_allocator_provides_expected_methods() -> None:
    class Dummy:
        pass

    camera = Dummy()
    _ensure_picamera_allocator(camera)
    allocator = getattr(camera, "allocator", None)
    assert allocator is not None
    assert hasattr(allocator, "sync")
    assert hasattr(allocator, "acquire")
    assert hasattr(allocator, "release")
    sync_context = allocator.sync(object(), object(), False)
    assert hasattr(sync_context, "__enter__")
    assert hasattr(sync_context, "__exit__")
    assert sync_context.__enter__() is sync_context
    assert sync_context.__exit__(None, None, None) is False
    with allocator.sync(object(), object(), False):
        pass
    assert allocator.acquire(object()) is None
    assert allocator.release(object()) is None
    # Unknown attributes should be safely ignored
    assert allocator.some_new_method() is None


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

    message = str(excinfo.value)
    assert "numpy ABI mismatch" in message
    assert "python3-picamera2" in message
    assert "--system-site-packages" in message
    assert "scripts/install.sh --pi" in message


def test_import_failure_numpy_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """NumPy ABI mismatches should provide remediation guidance."""

    import builtins

    original_import = builtins.__import__

    def raising_import(name: str, *args: object, **kwargs: object):
        if name == "picamera2":
            raise ValueError("numpy.dtype size changed, may indicate binary incompatibility")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    with pytest.raises(CameraError) as excinfo:
        Picamera2Camera()

    message = str(excinfo.value)
    assert "numpy.dtype size changed" in message
    assert (
        "sudo apt install --reinstall python3-numpy python3-picamera2 python3-simplejpeg"
        in message
    )
    assert "prefix on SimpleJPEG" in message


def test_picamera_initialisation_error_includes_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested errors should be surfaced when Picamera2 initialisation fails."""

    class ExplodingCamera:
        def __init__(self) -> None:
            raise RuntimeError("outer failure") from ValueError("inner detail")

    module = types.SimpleNamespace(Picamera2=ExplodingCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    with pytest.raises(CameraError) as excinfo:
        Picamera2Camera()

    assert "outer failure" in str(excinfo.value)
    assert "inner detail" in str(excinfo.value)


def test_picamera_busy_error_includes_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Busy camera errors should include guidance for stopping other services."""

    class BusyCamera:
        def __init__(self) -> None:
            raise RuntimeError("Camera __init__ sequence did not complete.") from RuntimeError(
                "Failed to acquire camera: Device or resource busy"
            )

    module = types.SimpleNamespace(Picamera2=BusyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    with pytest.raises(CameraError) as excinfo:
        Picamera2Camera()

    message = str(excinfo.value)
    assert "Device or resource busy" in message
    assert "stop conflicting services" in message


def test_picamera_busy_error_includes_conflicting_process_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Busy camera errors should surface detected conflicting processes."""

    class BusyCamera:
        def __init__(self) -> None:
            raise RuntimeError("Camera __init__ sequence did not complete.") from RuntimeError(
                "Failed to acquire camera: Device or resource busy"
            )

    module = types.SimpleNamespace(Picamera2=BusyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)
    monkeypatch.setattr(
        camera_module,
        "_collect_camera_conflicts",
        lambda: [
            "Processes currently using the camera: 123 (libcamera-hello --timeout). Stop these processes to free the device."
        ],
    )

    with pytest.raises(CameraError) as excinfo:
        Picamera2Camera()

    message = str(excinfo.value)
    assert "libcamera-hello" in message


def test_detect_camera_conflicts_reports_legacy_camera_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel worker processes should add guidance for disabling the legacy stack."""

    monkeypatch.setattr(camera_module, "_is_service_active", lambda name: False)
    monkeypatch.setattr(
        camera_module,
        "_list_camera_processes",
        lambda: ["319 ([kworker/R-mmal-vchiq])"],
    )

    hints = camera_module._collect_camera_conflicts()

    assert any("legacy camera interface" in hint for hint in hints)


def test_picamera_busy_error_includes_legacy_camera_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Busy errors should include legacy camera remediation when detected."""

    class BusyCamera:
        def __init__(self) -> None:
            raise RuntimeError("Camera __init__ sequence did not complete.") from RuntimeError(
                "Failed to acquire camera: Device or resource busy"
            )

    module = types.SimpleNamespace(Picamera2=BusyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)
    legacy_hint = (
        "Kernel threads named kworker/R-mmal-vchiq indicate the legacy camera interface is still enabled."
    )
    monkeypatch.setattr(
        camera_module,
        "_collect_camera_conflicts",
        lambda: [legacy_hint],
    )

    with pytest.raises(CameraError) as excinfo:
        Picamera2Camera()

    assert legacy_hint in str(excinfo.value)


def test_identify_camera_returns_source() -> None:
    camera = SyntheticCamera()
    assert identify_camera(camera) == "synthetic"


def test_diagnose_camera_conflicts_reports_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The diagnostics helper should report detected services and processes."""

    monkeypatch.setattr(camera_module, "_is_service_active", lambda name: True)
    monkeypatch.setattr(
        camera_module,
        "_list_camera_processes",
        lambda: ["123 (libcamera-vid)", "456 (kworker/R-mmal-vchiq)"],
    )

    hints = diagnose_camera_conflicts()

    assert any("libcamera-apps" in hint for hint in hints)
    assert any("Processes currently" in hint for hint in hints)
    assert any("legacy camera" in hint.lower() for hint in hints)


def test_picamera_frames_are_returned_in_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frames captured from Picamera2 should be converted from BGR to RGB order."""

    sample_frame = np.array([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8)

    class DummyCamera:
        def __init__(self) -> None:
            self._closed = False

        def create_video_configuration(self, **_: object) -> object:
            return object()

        def configure(self, _: object) -> None:
            return None

        def start(self) -> None:
            return None

        def capture_array(self) -> np.ndarray:
            return sample_frame

        def stop(self) -> None:
            return None

        def close(self) -> None:
            self._closed = True

    module = types.SimpleNamespace(Picamera2=DummyCamera)
    monkeypatch.setitem(sys.modules, "picamera2", module)

    camera = Picamera2Camera()
    try:
        frame = asyncio.run(camera.get_frame())
        assert frame.shape == sample_frame.shape
        assert np.array_equal(frame[..., 0], sample_frame[..., 2])
        assert np.array_equal(frame[..., 1], sample_frame[..., 1])
        assert np.array_equal(frame[..., 2], sample_frame[..., 0])
    finally:
        asyncio.run(camera.close())

