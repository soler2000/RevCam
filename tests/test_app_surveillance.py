"""Tests for the surveillance settings API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam import app as app_module
from rev_cam.config import ConfigManager, SURVEILLANCE_STANDARD_PRESETS


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    class _StubBatteryMonitor:
        def __init__(self, *args, **kwargs) -> None:
            self.capacity_mah = kwargs.get("capacity_mah", 1000)

        def read(self):  # pragma: no cover - indirect use
            return SimpleNamespace(to_dict=lambda: {})

    class _StubSupervisor:
        def start(self) -> None:  # pragma: no cover - indirect use
            return None

        async def aclose(self) -> None:  # pragma: no cover - indirect use
            return None

    class _StubDistanceMonitor:
        def __init__(self, *args, **kwargs) -> None:
            self.i2c_bus = kwargs.get("i2c_bus")

    def _noop_overlay(*args, **kwargs):
        return lambda frame: frame

    monkeypatch.setattr(app_module, "BatteryMonitor", lambda *args, **kwargs: _StubBatteryMonitor(*args, **kwargs))
    monkeypatch.setattr(app_module, "BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr(app_module, "DistanceMonitor", lambda *args, **kwargs: _StubDistanceMonitor(*args, **kwargs))
    monkeypatch.setattr(app_module, "create_battery_overlay", lambda *args, **kwargs: _noop_overlay())
    monkeypatch.setattr(app_module, "create_wifi_overlay", lambda *args, **kwargs: _noop_overlay())
    monkeypatch.setattr(app_module, "create_distance_overlay", lambda *args, **kwargs: _noop_overlay())
    monkeypatch.setattr(app_module, "create_reversing_aids_overlay", lambda *args, **kwargs: _noop_overlay())

    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    monkeypatch.setattr(app_module, "RECORDINGS_DIR", recordings_dir)

    config_path = tmp_path / "config.json"
    app = app_module.create_app(config_path)
    with TestClient(app) as test_client:
        test_client.config_path = config_path
        test_client.recordings_dir = recordings_dir
        yield test_client


def test_get_surveillance_settings(client: TestClient) -> None:
    response = client.get("/api/surveillance/settings")
    assert response.status_code == 200
    payload = response.json()
    assert "settings" in payload
    settings = payload["settings"]
    assert settings["profile"] == "standard"
    assert settings["preset"] in SURVEILLANCE_STANDARD_PRESETS
    assert settings["overlays_enabled"] is True
    assert settings["remember_recording_state"] is False
    assert settings["storage_threshold_percent"] == 10
    assert settings["motion_detection_enabled"] is False
    assert settings["motion_frame_decimation"] == 1
    assert settings["motion_post_event_seconds"] == 2.0
    assert "presets" in payload
    assert any(item["name"] == settings["preset"] for item in payload["presets"])


def test_update_surveillance_settings_standard(client: TestClient) -> None:
    response = client.post(
        "/api/surveillance/settings",
        json={"profile": "standard", "preset": "detail"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["settings"]["profile"] == "standard"
    assert payload["settings"]["preset"] == "detail"
    expected = SURVEILLANCE_STANDARD_PRESETS["detail"]
    assert payload["settings"]["fps"] == expected[0]
    assert payload["settings"]["jpeg_quality"] == expected[1]

    manager = ConfigManager(client.config_path)
    stored = manager.get_surveillance_settings()
    assert stored.preset == "detail"
    assert stored.resolved_fps == expected[0]


def test_update_surveillance_settings_advanced(client: TestClient) -> None:
    response = client.post(
        "/api/surveillance/settings",
        json={
            "profile": "expert",
            "expert_fps": 8,
            "expert_jpeg_quality": 88,
            "chunk_duration_seconds": 300,
            "overlays_enabled": False,
            "remember_recording_state": True,
            "motion_detection_enabled": True,
            "motion_sensitivity": 65,
            "motion_frame_decimation": 2,
            "motion_post_event_seconds": 0.5,
            "auto_purge_days": 10,
            "storage_threshold_percent": 20,
        },
    )
    assert response.status_code == 200
    payload = response.json()["settings"]
    assert payload["profile"] == "expert"
    assert payload["fps"] == 8
    assert payload["chunk_duration_seconds"] == 300
    assert payload["overlays_enabled"] is False
    assert payload["remember_recording_state"] is True
    assert payload["motion_detection_enabled"] is True
    assert payload["motion_sensitivity"] == 65
    assert payload["motion_frame_decimation"] == 2
    assert payload["motion_post_event_seconds"] == 0.5
    assert payload["auto_purge_days"] == 10
    assert payload["storage_threshold_percent"] == 20

    status_response = client.get("/api/surveillance/status")
    assert status_response.status_code == 200
    status_payload = status_response.json()["settings"]
    assert status_payload["motion_frame_decimation"] == 2
    assert status_payload["motion_post_event_seconds"] == 0.5


def test_update_surveillance_settings_expert_validation(client: TestClient) -> None:
    response = client.post(
        "/api/surveillance/settings",
        json={"profile": "expert", "expert_fps": "fast"},
    )
    assert response.status_code == 400
    detail = response.json()
    assert detail["detail"]


def test_delete_surveillance_recording(client: TestClient) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20230101-010101"
    chunk_file = recordings_dir / f"{name}.chunk001.json"
    chunk_file.write_text(json.dumps({"frames": []}), encoding="utf-8")
    meta = recordings_dir / f"{name}.meta.json"
    meta.write_text(
        json.dumps({
            "name": name,
            "chunks": [{"file": chunk_file.name, "frame_count": 0, "size_bytes": 2}],
            "ended_at": "2023-01-01T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    response = client.delete(f"/api/surveillance/recordings/{name}")
    assert response.status_code == 200
    assert not meta.exists()
    assert not chunk_file.exists()


def test_surveillance_status_storage(client: TestClient) -> None:
    response = client.get("/api/surveillance/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] in {"revcam", "surveillance"}
    assert payload.get("recording_mode") in {"idle", "continuous", "motion"}
    storage = payload.get("storage")
    assert isinstance(storage, dict)
    assert "free_percent" in storage
    assert "threshold_percent" in storage
    assert "processing" in payload
    assert payload["processing"] in {True, False}
    assert "processing_recording" in payload
    assert "motion" in payload
    motion = payload.get("motion")
    if motion is not None:
        assert isinstance(motion, dict)
        assert "enabled" in motion
        assert "session_state" in motion
        assert "session_active" in motion
        assert "session_override" in motion
        assert "post_event_record_seconds" in motion
    resume_state = payload.get("resume_state")
    assert isinstance(resume_state, dict)
    assert resume_state.get("mode") in {"revcam", "surveillance"}


def test_surveillance_timeline_page(client: TestClient) -> None:
    response = client.get("/surveillance/timeline")
    assert response.status_code == 200
    assert "Recording timeline" in response.text
