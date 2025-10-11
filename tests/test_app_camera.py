"""Integration tests covering camera API error reporting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam.app import create_app
from rev_cam.camera import BaseCamera, CameraError
from rev_cam.config import DEFAULT_RESOLUTION_KEY
from rev_cam.version import APP_VERSION


def _apply_common_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubBatteryMonitor:
        def read(self):  # pragma: no cover - used indirectly
            return SimpleNamespace(to_dict=lambda: {"available": False})

    class _StubSupervisor:
        def start(self) -> None:  # pragma: no cover - used indirectly
            return None

        async def aclose(self) -> None:  # pragma: no cover - used indirectly
            return None

    monkeypatch.setattr(
        "rev_cam.camera.Picamera2Camera",
        _BrokenPicamera,
    )
    monkeypatch.setattr(
        "rev_cam.app.BatteryMonitor",
        lambda *args, **kwargs: _StubBatteryMonitor(),
    )
    monkeypatch.setattr(
        "rev_cam.app.BatterySupervisor",
        lambda *args, **kwargs: _StubSupervisor(),
    )
    monkeypatch.setattr(
        "rev_cam.app.create_battery_overlay",
        lambda *args, **kwargs: (lambda frame: frame),
    )
    _StubWebRTCManager.instances.clear()
    monkeypatch.setattr("rev_cam.app.WebRTCManager", _StubWebRTCManager)


class _BrokenPicamera:
    """Stub Picamera implementation that always fails."""

    def __init__(self) -> None:
        raise CameraError("picamera backend unavailable")


class _RecorderStreamer:
    instances: list["_RecorderStreamer"] = []

    def __init__(
        self,
        *,
        camera: BaseCamera,
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
        self.apply_calls: list[dict[str, int | None]] = []
        _RecorderStreamer.instances.append(self)

    @property
    def media_type(self) -> str:
        return f"multipart/x-mixed-replace; boundary={self.boundary}"

    async def stream(self):  # pragma: no cover - not exercised in tests
        yield b""

    async def aclose(self) -> None:  # pragma: no cover - not exercised in tests
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
        self.apply_calls.append({"fps": fps, "jpeg_quality": jpeg_quality})


class _StubWebRTCManager:
    instances: list["_StubWebRTCManager"] = []

    def __init__(
        self,
        *,
        camera: BaseCamera,
        pipeline,
        fps: int = 20,
        **_: object,
    ) -> None:
        self.camera = camera
        self.pipeline = pipeline
        self.fps = fps
        self.apply_calls: list[dict[str, int | None]] = []
        self.sessions: list[dict[str, str]] = []
        _StubWebRTCManager.instances.append(self)

    async def create_session(self, sdp: str, offer_type: str):
        self.sessions.append({"sdp": sdp, "type": offer_type})
        return SimpleNamespace(sdp="answer", type="answer")

    def apply_settings(
        self,
        *,
        fps: int | None = None,
        jpeg_quality: int | None = None,
    ) -> None:
        if fps is not None:
            self.fps = fps
        self.apply_calls.append({"fps": fps, "jpeg_quality": jpeg_quality})

    async def aclose(self) -> None:  # pragma: no cover - not exercised in tests
        return None

@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _apply_common_stubs(monkeypatch)
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
    assert stream_info["settings"] == {"fps": 20, "jpeg_quality": 85}
    assert stream_info["active"] == {"fps": 20, "jpeg_quality": 85}
    assert stream_info["webrtc"]["enabled"] is True
    assert stream_info["webrtc"]["endpoint"] == "/stream/webrtc"
    assert stream_info["webrtc"]["fps"] == 20
    assert stream_info["mjpeg"]["enabled"] is True
    resolution_info = payload["resolution"]
    assert resolution_info["selected"] == DEFAULT_RESOLUTION_KEY
    assert resolution_info["active"] == DEFAULT_RESOLUTION_KEY
    assert any(opt["value"] == DEFAULT_RESOLUTION_KEY for opt in resolution_info["options"])


def test_stream_status_endpoint_reports_capabilities(client: TestClient) -> None:
    response = client.get("/api/stream")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["settings"] == {"fps": 20, "jpeg_quality": 85}
    assert payload["webrtc"]["enabled"] is True
    assert payload["webrtc"]["error"] is None
    assert payload["mjpeg"]["enabled"] is True


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
    assert stream_info["settings"] == {"fps": 20, "jpeg_quality": 85}
    assert stream_info["active"] == {"fps": 20, "jpeg_quality": 85}
    assert stream_info["webrtc"]["enabled"] is True
    assert stream_info["webrtc"]["endpoint"] == "/stream/webrtc"
    resolution_info = refreshed["resolution"]
    assert resolution_info["selected"] == DEFAULT_RESOLUTION_KEY
    assert resolution_info["active"] == DEFAULT_RESOLUTION_KEY


def test_camera_adjustments_persist_without_switching(client: TestClient) -> None:
    initial_response = client.get("/api/camera")
    assert initial_response.status_code == 200
    initial = initial_response.json()
    assert initial["image_adjustments"] == {"brightness": 100.0, "saturation": 100.0, "hue": 0.0}

    desired_adjustments = {"brightness": 135, "saturation": 80, "hue": -25}
    update_response = client.post(
        "/api/camera",
        json={
            "source": initial["selected"],
            "resolution": initial["resolution"]["selected"],
            "image_adjustments": desired_adjustments,
        },
    )
    assert update_response.status_code == 200
    update_payload = update_response.json()
    saved_adjustments = update_payload["image_adjustments"]
    assert saved_adjustments["brightness"] == desired_adjustments["brightness"]
    assert saved_adjustments["saturation"] == desired_adjustments["saturation"]
    assert saved_adjustments["hue"] == desired_adjustments["hue"]

    refreshed = client.get("/api/camera")
    assert refreshed.status_code == 200
    refreshed_payload = refreshed.json()
    assert refreshed_payload["image_adjustments"] == saved_adjustments

    config_path = getattr(client, "config_path", None)
    assert config_path is not None
    config_data = json.loads(Path(config_path).read_text())
    assert config_data["camera_image_adjustments"] == saved_adjustments


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


def test_snapshot_endpoint_returns_jpeg(client: TestClient) -> None:
    response = client.get("/api/camera/snapshot")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("image/jpeg")
    assert response.content


def test_snapshot_endpoint_handles_camera_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingCamera(BaseCamera):
        async def get_frame(self):  # pragma: no cover - exercised in test
            raise CameraError("capture failed")

        async def close(self) -> None:  # pragma: no cover - exercised indirectly
            return None

    def _create_camera_override(choice: str, *args, **kwargs):
        if choice == "picamera":
            raise CameraError("picamera backend unavailable")
        if choice == "synthetic":
            return _FailingCamera()
        raise AssertionError(f"Unexpected camera choice {choice}")

    _apply_common_stubs(monkeypatch)
    monkeypatch.setattr("rev_cam.app.create_camera", _create_camera_override)

    config_path = tmp_path / "config.json"
    app = create_app(config_path)
    with TestClient(app) as test_client:
        response = test_client.get("/api/camera/snapshot")
        assert response.status_code == 503
        payload = response.json()
        assert payload["detail"] == "capture failed"

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
        assert stream_info["enabled"] is True
        assert stream_info["endpoint"] is None
        assert stream_info["error"] == "encoder offline"
        assert stream_info["settings"] == {"fps": 20, "jpeg_quality": 85}
        assert stream_info["active"] == {"fps": 20, "jpeg_quality": None}
        assert stream_info["webrtc"]["enabled"] is True
        assert stream_info["mjpeg"]["enabled"] is False
        assert stream_info["mjpeg"]["error"] == "encoder offline"

        stream_response = test_client.get("/stream/mjpeg")
        assert stream_response.status_code == 503
        detail = stream_response.json()["detail"]
        assert "encoder offline" in detail


def test_stream_status_endpoint_reports_webrtc_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_common_stubs(monkeypatch)

    class _BrokenWebRTCManager:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("aiortc unavailable")

    monkeypatch.setattr("rev_cam.app.WebRTCManager", _BrokenWebRTCManager)

    config_path = tmp_path / "config.json"
    app = create_app(config_path)
    with TestClient(app) as test_client:
        response = test_client.get("/api/stream")
        assert response.status_code == 200
        payload = response.json()
        assert payload["enabled"] is True
        assert payload["webrtc"]["enabled"] is False
        assert payload["webrtc"]["error"] == "aiortc unavailable"
        assert payload["mjpeg"]["enabled"] is True


def test_stream_settings_endpoint_updates_config_and_streamer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_common_stubs(monkeypatch)
    _RecorderStreamer.instances.clear()
    monkeypatch.setattr("rev_cam.app.MJPEGStreamer", _RecorderStreamer)
    config_path = tmp_path / "config.json"
    app = create_app(config_path)
    with TestClient(app) as test_client:
        initial = test_client.get("/api/camera")
        assert initial.status_code == 200
        initial_payload = initial.json()
        assert initial_payload["stream"]["settings"] == {"fps": 20, "jpeg_quality": 85}
        assert initial_payload["stream"]["active"] == {"fps": 20, "jpeg_quality": 85}
        assert _RecorderStreamer.instances
        streamer = _RecorderStreamer.instances[-1]
        assert _StubWebRTCManager.instances
        webrtc_manager = _StubWebRTCManager.instances[-1]

        response = test_client.post(
            "/api/stream/settings",
            json={"fps": 12, "jpeg_quality": 65},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["fps"] == 12
        assert payload["jpeg_quality"] == 65
        assert payload["active"] == {"fps": 12, "jpeg_quality": 65}
        assert streamer.apply_calls
        assert streamer.apply_calls[-1] == {"fps": 12, "jpeg_quality": 65}
        assert streamer.fps == 12
        assert streamer.jpeg_quality == 65
        assert webrtc_manager.apply_calls
        assert webrtc_manager.apply_calls[-1]["fps"] == 12
        assert webrtc_manager.fps == 12

        config_data = json.loads(Path(config_path).read_text())
        assert config_data["stream"] == {"fps": 12, "jpeg_quality": 65}

        refreshed = test_client.get("/api/camera")
        assert refreshed.status_code == 200
        stream_info = refreshed.json()["stream"]
        assert stream_info["settings"] == {"fps": 12, "jpeg_quality": 65}
        assert stream_info["active"] == {"fps": 12, "jpeg_quality": 65}
        assert stream_info["webrtc"]["fps"] == 12


def test_stream_settings_endpoint_validates_input(client: TestClient) -> None:
    response = client.post("/api/stream/settings", json={"fps": 0})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "fps" in detail.lower()

    invalid_quality = client.post("/api/stream/settings", json={"jpeg_quality": 150})
    assert invalid_quality.status_code == 400
    quality_detail = invalid_quality.json()["detail"]
    assert "quality" in quality_detail.lower()

    camera = client.get("/api/camera").json()
    assert camera["stream"]["settings"] == {"fps": 20, "jpeg_quality": 85}


def test_webrtc_endpoint_returns_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_common_stubs(monkeypatch)
    _RecorderStreamer.instances.clear()
    monkeypatch.setattr("rev_cam.app.MJPEGStreamer", _RecorderStreamer)
    config_path = tmp_path / "config.json"
    app = create_app(config_path)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/stream/webrtc",
            json={"sdp": "offer", "type": "offer"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload == {"sdp": "answer", "type": "answer"}
        assert _StubWebRTCManager.instances
        manager = _StubWebRTCManager.instances[-1]
        assert manager.sessions
        assert manager.sessions[-1] == {"sdp": "offer", "type": "offer"}


def test_webrtc_error_logging_endpoint(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    response = client.post(
        "/api/log/webrtc-error",
        json={
            "name": "OperationError",
            "message": "ICE connection failed",
            "stack": "Error: ICE failed\n    at line",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "logged"}
    warning_messages = [
        record.message
        for record in caplog.records
        if record.name == "rev_cam.app" and record.levelname == "WARNING"
    ]
    assert any("Client reported WebRTC error: OperationError – ICE connection failed" in msg for msg in warning_messages)
    debug_messages = [
        record.message
        for record in caplog.records
        if record.name == "rev_cam.app" and record.levelname == "DEBUG"
    ]
    assert any("Client WebRTC error stack trace" in msg for msg in debug_messages)
