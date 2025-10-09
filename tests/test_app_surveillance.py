"""Tests for the surveillance settings API."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam import app as app_module
from rev_cam.config import ConfigManager, SURVEILLANCE_STANDARD_PRESETS
import rev_cam.recording as recording
from rev_cam.recording import load_recording_payload


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
    media_dir = recordings_dir / "media"
    preview_dir = recordings_dir / "previews"
    media_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True)
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
    video_file = recordings_dir / "media" / f"{name}.mp4"
    video_file.write_bytes(b"mp4")
    meta = recordings_dir / f"{name}.meta.json"
    meta.write_text(
        json.dumps({
            "name": name,
            "file": f"media/{video_file.name}",
            "media_type": "video/mp4",
            "frame_count": 0,
            "size_bytes": video_file.stat().st_size,
            "ended_at": "2023-01-01T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    response = client.delete(f"/api/surveillance/recordings/{name}")
    assert response.status_code == 200
    assert not meta.exists()
    assert not video_file.exists()


def test_fetch_surveillance_recording_metadata_and_media(client: TestClient) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20230102-020202"
    video_file = recordings_dir / "media" / f"{name}.mp4"
    video_payload = b"fake-mp4-data"
    video_file.write_bytes(video_payload)
    preview_file = recordings_dir / "previews" / f"{name}.jpg"
    preview_file.write_bytes(b"jpg")
    meta = recordings_dir / f"{name}.meta.json"
    meta.write_text(
        json.dumps(
            {
                "name": name,
                "fps": 5,
                "file": f"media/{video_file.name}",
                "frame_count": 10,
                "size_bytes": video_file.stat().st_size,
                "duration_seconds": 2.5,
                "media_type": "video/mp4",
                "preview_file": f"previews/{name}.jpg",
            }
        ),
        encoding="utf-8",
    )

    response = client.get(
        f"/api/surveillance/recordings/{name}", params={"include_frames": "false"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == name
    assert payload["file"] == f"media/{video_file.name}"
    assert payload.get("preview_file") == f"previews/{name}.jpg"
    assert payload["media_type"] == "video/mp4"
    assert payload["duration_seconds"] == 2.5
    assert "frames" not in payload

    media_response = client.get(f"/api/surveillance/recordings/{name}/media")
    assert media_response.status_code == 200
    assert media_response.headers["content-type"].startswith("video/mp4")
    assert "inline" in media_response.headers["content-disposition"].lower()
    assert media_response.content == video_payload


def test_load_recording_payload_migrates_legacy_assets(client: TestClient) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20251009-legacy"
    legacy_video = recordings_dir / f"{name}.mp4"
    legacy_preview = recordings_dir / f"{name}.jpg"
    video_payload = b"legacy-video"
    preview_payload = b"legacy-preview"
    legacy_video.write_bytes(video_payload)
    legacy_preview.write_bytes(preview_payload)

    metadata = {
        "name": name,
        "file": legacy_video.name,
        "media_type": "video/mp4",
        "duration_seconds": 1.23,
        "frame_count": 12,
        "chunks": [
            {
                "file": legacy_video.name,
                "media_type": "video/mp4",
                "codec": "h264",
                "duration_seconds": 1.23,
                "frame_count": 12,
                "size_bytes": len(video_payload),
            }
        ],
        "preview_file": legacy_preview.name,
    }
    meta_path = recordings_dir / f"{name}.meta.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    payload = load_recording_payload(recordings_dir, name, include_frames=False)

    assert payload["file"] == f"media/{legacy_video.name}"
    media_path = recordings_dir / payload["file"]
    assert media_path.exists()
    assert media_path.read_bytes() == video_payload
    assert not legacy_video.exists()
    assert payload.get("preview_file") == f"previews/{legacy_preview.name}"
    preview_path = recordings_dir / payload["preview_file"]
    assert preview_path.exists()
    assert preview_path.read_bytes() == preview_payload

    stored_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert stored_metadata["file"] == payload["file"]
    assert stored_metadata.get("preview_file") == payload["preview_file"]


def test_fetch_surveillance_recording_media_for_legacy_mjpeg(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20230102-020203"
    chunk_file = recordings_dir / "media" / f"{name}.chunk001.mjpeg"
    chunk_payload = b"fake-avi-data"
    chunk_file.write_bytes(chunk_payload)
    meta = recordings_dir / f"{name}.meta.json"
    meta.write_text(
        json.dumps(
            {
                "name": name,
                "fps": 5,
                "chunks": [
                    {
                        "file": f"media/{chunk_file.name}",
                        "frame_count": 10,
                        "size_bytes": chunk_file.stat().st_size,
                        "duration_seconds": 2.5,
                        "media_type": "multipart/x-mixed-replace; boundary=chunk001",
                        "codec": "jpeg-fallback",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_remux(
        source_path: Path,
        *,
        target_path: Path,
        name: str,
        index: int,
        fps: float,
        jpeg_quality: int,
        boundary: str,
        start_offset: float | None = None,
    ) -> dict[str, object]:
        target_path.write_bytes(chunk_payload)
        return {
            "file": target_path.name,
            "media_type": f"multipart/x-mixed-replace; boundary={boundary}",
            "size_bytes": target_path.stat().st_size,
            "codec": "jpeg-fallback",
        }

    monkeypatch.setattr(recording, "_remux_mjpeg_video_chunk", _fake_remux)

    response = client.get(
        f"/api/surveillance/recordings/{name}", params={"include_frames": "false"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["file"].endswith(".mjpeg")
    assert payload["media_type"].startswith("multipart/x-mixed-replace")

    media_response = client.get(f"/api/surveillance/recordings/{name}/media")
    assert media_response.status_code == 200
    assert media_response.headers["content-type"].startswith(
        "multipart/x-mixed-replace"
    )
    assert media_response.content == chunk_payload


def test_fetch_surveillance_recording_media_not_found(client: TestClient) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20230103-030303"
    meta = recordings_dir / f"{name}.meta.json"
    meta.write_text(json.dumps({"name": name}), encoding="utf-8")

    response = client.get(f"/api/surveillance/recordings/{name}/media")
    assert response.status_code == 404


def test_fetch_surveillance_recording_media_uses_file_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20240101-010101"
    video_file = recordings_dir / "media" / f"{name}.mp4"
    payload = b"video-data"
    video_file.write_bytes(payload)

    def _fake_load(
        directory: Path, requested_name: str, *, include_frames: bool = False
    ) -> dict[str, object]:
        assert directory == recordings_dir
        assert requested_name == name
        return {
            "name": name,
            "file": "media/missing.mp4",
            "file_path": str(video_file),
            "media_type": "video/mp4",
        }

    monkeypatch.setattr(app_module, "load_recording_payload", _fake_load)

    response = client.get(f"/api/surveillance/recordings/{name}/media")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content == payload


def test_fetch_surveillance_recording_media_recovers_missing_metadata(
    client: TestClient,
) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20240202-020202"
    legacy_chunk = recordings_dir / f"{name}.chunk001.mp4"
    legacy_chunk.write_bytes(b"legacy")
    final_path = recordings_dir / "media" / f"{name}.mp4"
    payload = b"modern"
    final_path.write_bytes(payload)

    meta = recordings_dir / f"{name}.meta.json"
    meta.write_text(
        json.dumps(
            {
                "name": name,
                "file": legacy_chunk.name,
                "chunks": [
                    {
                        "file": legacy_chunk.name,
                        "media_type": "video/mp4",
                        "size_bytes": len(payload),
                    }
                ],
                "media_type": "video/mp4",
            }
        ),
        encoding="utf-8",
    )

    response = client.get(f"/api/surveillance/recordings/{name}/media")
    assert response.status_code == 200
    assert response.content == payload
    stored = json.loads(meta.read_text(encoding="utf-8"))
    assert stored["file"].startswith("media/")


def test_surveillance_recording_codec_failure_surfaces_to_clients(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20240505-050505"
    codec_message = recording._compose_codec_failure_message(None, [])

    class _StubCamera:
        async def get_frame(self):  # pragma: no cover - not exercised
            return None

    class _StubPipeline:
        def process(self, frame):  # pragma: no cover - not exercised
            return frame

    def _failing_chunk_writer(self, *_args, **_kwargs):
        self._video_encoding_disabled = True
        self._last_video_initialisation_error = codec_message
        self._persist_codec_state()
        return None

    monkeypatch.setattr(
        recording.RecordingManager, "_create_chunk_writer", _failing_chunk_writer
    )

    manager = recording.RecordingManager(
        camera=_StubCamera(),
        pipeline=_StubPipeline(),
        directory=recordings_dir,
    )
    manager._recording_active = True
    manager._recording_name = name
    manager._recording_started_at = datetime.now(timezone.utc)
    manager._recording_started_monotonic = time.perf_counter()
    manager._total_frame_count = 3

    manager._create_chunk_writer(name, 1, None)
    context = manager._capture_finalise_context_locked(
        stop_reason=None, deactivate_session=True, notify_stop=True
    )
    assert context is not None

    async def _finalise() -> dict[str, object] | None:
        await manager._process_finalise_context(context, track=False)
        metadata = await manager.wait_for_processing()
        await manager.aclose()
        return metadata

    metadata = asyncio.run(_finalise())
    assert metadata is not None
    assert metadata.get("processing_error") == codec_message
    assert metadata.get("media_available") is False
    assert "file" not in metadata
    assert "No video file was created." in codec_message
    assert "Install FFmpeg with H.264 encoder support" in codec_message

    meta_path = recordings_dir / f"{name}.meta.json"
    assert meta_path.exists()
    stored_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert stored_metadata.get("processing_error") == codec_message
    assert stored_metadata.get("media_available") is False
    assert "file" not in stored_metadata

    media_file = recordings_dir / "media" / f"{name}.mp4"
    assert not media_file.exists()

    media_response = client.get(f"/api/surveillance/recordings/{name}/media")
    assert media_response.status_code == 409
    assert media_response.json()["detail"] == codec_message

    download_response = client.get(
        f"/api/surveillance/recordings/{name}/download"
    )
    assert download_response.status_code == 409
    assert download_response.json()["detail"] == codec_message

    export_response = client.post(
        f"/api/surveillance/recordings/{name}/export"
    )
    assert export_response.status_code == 409
    assert export_response.json()["detail"] == codec_message


def test_finalise_marks_processing_error_when_media_missing(tmp_path: Path) -> None:
    class _StubCamera:
        async def get_frame(self):  # pragma: no cover - not exercised
            return None

    class _StubPipeline:
        def process(self, frame):  # pragma: no cover - not exercised
            return frame

    manager = recording.RecordingManager(
        camera=_StubCamera(),
        pipeline=_StubPipeline(),
        directory=tmp_path,
    )
    name = "20240101-010101"
    base_metadata = {
        "name": name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": 1.0,
        "fps": manager.fps,
        "frame_count": 5,
        "thumbnail": None,
    }
    chunk_entries = [{"file": "media/does-not-exist.mp4", "media_type": "video/mp4"}]
    manager._video_encoding_disabled = True

    async def _finalise() -> dict[str, object]:
        result = await manager._run_recording_finalise(
            name=name,
            base_metadata=base_metadata,
            chunk_entries=chunk_entries,
            invoke_callback=False,
        )
        await manager.aclose()
        return result

    metadata = asyncio.run(_finalise())
    assert metadata["media_available"] is False
    error_message = metadata["processing_error"]
    assert "No video file was created." in error_message
    assert "Install FFmpeg with H.264 encoder support" in error_message
    assert "file" not in metadata


def test_export_surveillance_recording_to_desktop(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20240303-030303"
    video_file = recordings_dir / "media" / f"{name}.mp4"
    payload = b"export-me"
    video_file.write_bytes(payload)
    meta = recordings_dir / f"{name}.meta.json"
    meta.write_text(
        json.dumps({"name": name, "file": f"media/{video_file.name}", "media_type": "video/mp4"}),
        encoding="utf-8",
    )

    desktop_dir = tmp_path / "Desktop"
    monkeypatch.setattr(app_module, "_resolve_desktop_directory", lambda: desktop_dir)

    response = client.post(f"/api/surveillance/recordings/{name}/export")
    assert response.status_code == 200
    payload_json = response.json()
    exported_path = desktop_dir / video_file.name
    assert exported_path.exists()
    assert exported_path.read_bytes() == payload
    assert payload_json["path"].endswith(video_file.name)


def test_download_surveillance_recording_streams_file_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_dir: Path = client.recordings_dir
    name = "20240102-020202"
    video_file = recordings_dir / "media" / f"{name}.mp4"
    payload = b"download-video"
    video_file.write_bytes(payload)

    def _fake_load(
        directory: Path, requested_name: str, *, include_frames: bool = False
    ) -> dict[str, object]:
        assert directory == recordings_dir
        assert requested_name == name
        return {
            "name": name,
            "file": "",
            "file_path": str(video_file),
            "media_type": "video/mp4",
        }

    monkeypatch.setattr(recording, "load_recording_payload", _fake_load)

    response = client.get(f"/api/surveillance/recordings/{name}/download")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content == payload


def test_surveillance_status_storage(client: TestClient) -> None:
    response = client.get("/api/surveillance/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] in {"revcam", "surveillance"}
    assert payload.get("recording_mode") in {"idle", "continuous", "motion"}
    assert "recording_started_at" in payload
    started_at = payload.get("recording_started_at")
    assert started_at is None or isinstance(started_at, str)
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
        assert "session_recording" in motion
        assert "event_active" in motion
        assert "post_event_record_seconds" in motion
    resume_state = payload.get("resume_state")
    assert isinstance(resume_state, dict)
    assert resume_state.get("mode") in {"revcam", "surveillance"}


def test_surveillance_timeline_page(client: TestClient) -> None:
    response = client.get("/surveillance/timeline")
    assert response.status_code == 200
    assert "Recording timeline" in response.text


def test_compose_codec_failure_message_notes_attempts() -> None:
    message = recording._compose_codec_failure_message(
        None, ["h264_v4l2m2m", "libx264", "h264_v4l2m2m"]
    )
    assert "attempted codecs: h264_v4l2m2m, libx264" in message
    assert message.endswith(
        "No video file was created. Install FFmpeg with H.264 encoder support (e.g. sudo apt install ffmpeg libavcodec-extra) and ensure Raspberry Pi hardware video encoding is enabled. Confirm that FFmpeg exposes one of: h264_v4l2m2m, libx264."
    )
