"""Integration tests covering camera API error reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam.app import create_app
from rev_cam.camera import CameraError
from rev_cam.version import APP_VERSION


class _BrokenPicamera:
    """Stub Picamera implementation that always fails."""

    def __init__(self) -> None:
        raise CameraError("picamera backend unavailable")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("rev_cam.camera.Picamera2Camera", _BrokenPicamera)
    app = create_app(tmp_path / "config.json")
    with TestClient(app) as test_client:
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


def test_mjpeg_stream_endpoint(client: TestClient) -> None:
    with client.stream("GET", "/stream/mjpeg") as response:
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "multipart/x-mixed-replace" in content_type
        iterator = response.iter_bytes()
        first_chunk = next(iterator)
        assert b"--frame" in first_chunk
