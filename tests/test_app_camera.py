"""Integration tests covering camera API error reporting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam.app import create_app
from rev_cam.camera import CameraError
from rev_cam.config import DEFAULT_RESOLUTION_KEY
from rev_cam.version import APP_VERSION


class _BrokenPicamera:
    """Stub Picamera implementation that always fails."""

    def __init__(self) -> None:
        raise CameraError("picamera backend unavailable")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    class _StubBatteryMonitor:
        def read(self):  # pragma: no cover - used indirectly
            return SimpleNamespace(to_dict=lambda: {"available": False})

    class _StubSupervisor:
        def start(self) -> None:  # pragma: no cover - used indirectly
            return None

        async def aclose(self) -> None:  # pragma: no cover - used indirectly
            return None

    monkeypatch.setattr("rev_cam.camera.Picamera2Camera", _BrokenPicamera)
    monkeypatch.setattr("rev_cam.app.BatteryMonitor", lambda *args, **kwargs: _StubBatteryMonitor())
    monkeypatch.setattr("rev_cam.app.BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr("rev_cam.app.create_battery_overlay", lambda *args, **kwargs: (lambda frame: frame))
    config_path = tmp_path / "config.json"
    app = create_app(config_path)
    with TestClient(app) as test_client:
        test_client.config_path = config_path
        yield test_client


def test_camera_endpoint_reports_picamera_error(client: TestClient) -> None:
    response = client.get("/api/camera")
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected"] == "auto"
    assert payload["active"] == "synthetic"
    assert "picamera" in payload["errors"]
    assert "picamera backend unavailable" in payload["errors"]["picamera"]
    assert payload["version"] == APP_VERSION
    stream_info = payload["stream"]
    assert stream_info["enabled"] is True
    assert stream_info["endpoint"] == "/stream/mjpeg"
    assert "multipart/x-mixed-replace" in stream_info["content_type"]
    resolution_info = payload["resolution"]
    assert resolution_info["selected"] == DEFAULT_RESOLUTION_KEY
    assert resolution_info["active"] == DEFAULT_RESOLUTION_KEY
    assert any(opt["value"] == DEFAULT_RESOLUTION_KEY for opt in resolution_info["options"])


def test_camera_update_surfaces_failure(client: TestClient) -> None:
    response = client.post("/api/camera", json={"source": "picamera"})
    assert response.status_code == 400
    payload = response.json()
    assert "picamera backend unavailable" in payload["detail"]

    refreshed = client.get("/api/camera").json()
    assert "picamera backend unavailable" in refreshed["errors"]["picamera"]
    assert refreshed["version"] == APP_VERSION
    stream_info = refreshed["stream"]
    assert stream_info["enabled"] is True
    assert stream_info["endpoint"] == "/stream/mjpeg"
    assert "multipart/x-mixed-replace" in stream_info["content_type"]
    resolution_info = refreshed["resolution"]
    assert resolution_info["selected"] == DEFAULT_RESOLUTION_KEY
    assert resolution_info["active"] == DEFAULT_RESOLUTION_KEY


def test_camera_resolution_update(client: TestClient) -> None:
    response = client.post(
        "/api/camera",
        json={"source": "auto", "resolution": "640x480"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["resolution"]["selected"] == "640x480"
    assert payload["resolution"]["active"] == "640x480"
    config_path = getattr(client, "config_path", None)
    assert config_path is not None
    config_data = json.loads(Path(config_path).read_text())
    assert config_data["resolution"] == {"width": 640, "height": 480}


def test_mjpeg_stream_endpoint(client: TestClient) -> None:
    with client.stream("GET", "/stream/mjpeg") as response:
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "multipart/x-mixed-replace" in content_type
        iterator = response.iter_bytes()
        first_chunk = next(iterator)
        assert b"--frame" in first_chunk


def test_stream_error_surfaces_when_streamer_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenStreamer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("encoder offline")

    monkeypatch.setattr("rev_cam.app.MJPEGStreamer", _BrokenStreamer)
    config_path = tmp_path / "config.json"
    app = create_app(config_path)
    with TestClient(app) as test_client:
        response = test_client.get("/api/camera")
        assert response.status_code == 200
        payload = response.json()
        stream_info = payload["stream"]
        assert stream_info["enabled"] is False
        assert stream_info["endpoint"] is None
        assert stream_info["error"] == "encoder offline"

        stream_response = test_client.get("/stream/mjpeg")
        assert stream_response.status_code == 503
        detail = stream_response.json()["detail"]
        assert "encoder offline" in detail
