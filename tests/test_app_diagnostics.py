from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam import app as app_module


def _apply_app_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubBatteryMonitor:
        def read(self):  # pragma: no cover - used indirectly
            return SimpleNamespace(to_dict=lambda: {"available": False})

    class _StubSupervisor:
        def start(self) -> None:  # pragma: no cover - used indirectly
            return None

        async def aclose(self) -> None:  # pragma: no cover - used indirectly
            return None

    class _StubDistanceMonitor:
        def read(self):  # pragma: no cover - used indirectly
            return SimpleNamespace(available=False)

    class _StubCamera(app_module.BaseCamera):
        async def get_frame(self):  # pragma: no cover - not exercised
            import numpy as np

            return np.zeros((1, 1, 3), dtype=np.uint8)

        async def close(self) -> None:  # pragma: no cover - not exercised
            return None

    class _StubStreamer:
        def __init__(
            self,
            *,
            camera,
            pipeline,
            fps: int = 20,
            jpeg_quality: int = 85,
            boundary: str = "frame",
        ) -> None:
            self.camera = camera
            self.pipeline = pipeline
            self.fps = fps
            self.jpeg_quality = jpeg_quality
            self.boundary = boundary

        @property
        def media_type(self) -> str:
            return f"multipart/x-mixed-replace; boundary={self.boundary}"

        async def stream(self):  # pragma: no cover - not exercised
            yield b""

        async def aclose(self) -> None:  # pragma: no cover - not exercised
            return None

        def apply_settings(
            self,
            *,
            fps: int | None = None,
            jpeg_quality: int | None = None,
        ) -> None:
            if fps is not None:
                self.fps = fps
            if jpeg_quality is not None:
                self.jpeg_quality = jpeg_quality

    class _StubWiFiManager:
        def close(self) -> None:  # pragma: no cover - used indirectly
            return None

    def _create_camera(choice: str, *args, **kwargs):
        return _StubCamera()

    monkeypatch.setattr(app_module, "BatteryMonitor", lambda *args, **kwargs: _StubBatteryMonitor())
    monkeypatch.setattr(app_module, "BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr(app_module, "create_battery_overlay", lambda *args, **kwargs: (lambda frame: frame))
    monkeypatch.setattr(app_module, "create_distance_overlay", lambda *args, **kwargs: (lambda frame: frame))
    monkeypatch.setattr(app_module, "DistanceMonitor", lambda *args, **kwargs: _StubDistanceMonitor())
    monkeypatch.setattr(app_module, "create_camera", _create_camera)
    monkeypatch.setattr(app_module, "identify_camera", lambda camera: "stub")
    monkeypatch.setattr(app_module, "MJPEGStreamer", _StubStreamer)
    monkeypatch.setattr(app_module, "WiFiManager", lambda *args, **kwargs: _StubWiFiManager())


def test_diagnostics_endpoint_returns_payload(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_app_stubs(monkeypatch)

    payload = {
        "version": "1.2.3",
        "camera_conflicts": ["motion service", "legacy camera stack"],
        "picamera": {
            "status": "error",
            "details": ["picamera2 module not found."],
            "hints": ["Install Picamera2 packages"],
            "numpy_version": "1.26.2",
        },
    }
    monkeypatch.setattr(app_module, "collect_diagnostics", lambda: payload)

    app = app_module.create_app(tmp_path / "config.json")
    with TestClient(app) as client:
        response = client.get("/api/diagnostics")

    assert response.status_code == 200
    assert response.json() == payload


def test_diagnostics_endpoint_maps_errors(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_app_stubs(monkeypatch)

    def _raise_error() -> None:
        raise RuntimeError("test failure")

    monkeypatch.setattr(app_module, "collect_diagnostics", _raise_error)

    app = app_module.create_app(tmp_path / "config.json")
    with TestClient(app) as client:
        response = client.get("/api/diagnostics")

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "Unable to collect diagnostics" in detail
    assert "test failure" in detail
