from __future__ import annotations

import json
import types
from pathlib import Path

import asyncio
from fractions import Fraction

import av
import numpy as np
import pytest

import rev_cam.recording as recording
from rev_cam.camera import BaseCamera
from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline
from rev_cam.recording import (
    MotionDetector,
    RecordingManager,
    build_recording_video,
    load_recording_metadata,
    load_recording_payload,
)


class _StaticCamera(BaseCamera):
    async def get_frame(self) -> np.ndarray:  # pragma: no cover - not exercised in tests
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _pipeline() -> FramePipeline:
    return FramePipeline(lambda: Orientation(rotation=0, flip_horizontal=False, flip_vertical=False))


def test_motion_detector_pauses_and_resumes() -> None:
    detector = MotionDetector(enabled=True, sensitivity=75, inactivity_timeout=0.1)
    still = np.zeros((16, 16, 3), dtype=np.uint8)
    assert detector.should_record(still, 0.0) is True
    # Still frames before inactivity threshold should continue recording.
    assert detector.should_record(still, 0.05) is True
    # After inactivity threshold with no change we pause.
    assert detector.should_record(still, 0.2) is False
    meta = detector.snapshot()
    assert meta["pause_count"] == 1
    assert meta["active"] is False
    # Motion resumes recording immediately.
    moving = still.copy()
    moving[:4, :4, :] = 255
    assert detector.should_record(moving, 0.25) is True
    meta = detector.snapshot()
    assert meta["active"] is True


def test_motion_detector_decimates_frames() -> None:
    detector = MotionDetector(
        enabled=True,
        sensitivity=75,
        inactivity_timeout=1.0,
        frame_decimation=3,
    )
    baseline = np.zeros((16, 16, 3), dtype=np.uint8)
    motion_a = baseline.copy()
    motion_a[:4, :4, :] = 255
    motion_b = baseline.copy()
    motion_b[4:, 4:, :] = 255
    motion_c = baseline.copy()
    motion_c[:, :4, :] = 128

    assert detector.should_record(baseline, 0.0) is True
    # Next two frames should be dropped by decimation while motion is active.
    assert detector.should_record(motion_a, 0.1) is False
    assert detector.should_record(motion_b, 0.2) is False
    # Third frame after initial capture is recorded again.
    assert detector.should_record(motion_c, 0.3) is True
    snapshot = detector.snapshot()
    assert snapshot["frame_decimation"] == 3
    assert snapshot["decimation_drops"] >= 2


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_recording_manager_records_only_when_motion_detected(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        motion_detection_enabled=True,
        motion_sensitivity=80,
        motion_inactivity_timeout_seconds=0.05,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)
    await manager.start_recording()

    still = np.zeros((32, 32, 3), dtype=np.uint8)
    await manager._record_frame(b"frame1", 0.0, still)
    await manager._record_frame(b"frame2", 0.02, still)
    # Past inactivity threshold with no motion, frame is skipped.
    await manager._record_frame(b"frame3", 0.2, still)
    snapshot = manager.get_motion_status()
    assert snapshot["recorded_frames"] == 2
    assert snapshot["session_state"] == "monitoring"
    assert snapshot["session_active"] is True
    assert snapshot["session_recording"] is False
    assert "post_event_record_seconds" in snapshot

    moving = still.copy()
    moving[:4, :4, :] = 255
    await manager._record_frame(b"frame4", 0.25, moving)
    snapshot = manager.get_motion_status()
    assert snapshot["recorded_frames"] == 3
    assert snapshot["session_state"] == "recording"
    assert snapshot["session_active"] is True
    assert snapshot["session_recording"] is True
    assert snapshot["post_event_record_seconds"] >= 0

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True
    metadata = await manager.wait_for_processing()
    assert metadata is not None
    motion = metadata["motion_detection"]
    assert motion["enabled"] is True
    assert motion["pause_count"] >= 1
    assert motion["skipped_frames"] >= 1
    assert motion["recorded_frames"] == 3
    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_start_recording_survives_cancellation(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        fps=2,
    )

    original_ensure = manager._ensure_producer_running

    async def _delayed_ensure(self):
        await asyncio.sleep(0.05)
        await original_ensure()

    manager._ensure_producer_running = types.MethodType(_delayed_ensure, manager)

    task = asyncio.create_task(manager.start_recording())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert manager.is_recording is True
    assert manager.recording_started_at is not None

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] in {True, False}
    await manager.wait_for_processing()
    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_chunk_mp4_generation(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        fps=2,
        chunk_duration_seconds=1,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)
    await manager.start_recording()

    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    frame[:8, :8, :] = 255
    jpeg_payload = manager._encode_frame(frame)

    await manager._record_frame(jpeg_payload, 0.0, frame)
    await manager._record_frame(jpeg_payload, 0.5, frame)
    await manager._record_frame(jpeg_payload, 1.1, frame)

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True
    metadata = await manager.wait_for_processing()
    assert metadata is not None
    assert metadata.get("media_type") == "video/mp4"
    assert metadata.get("size_bytes", 0) > 0
    file_name = metadata.get("file")
    assert isinstance(file_name, str)
    video_path = tmp_path / file_name
    assert video_path.exists()
    with av.open(str(video_path), mode="r") as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        assert video_streams
        first_frame = next(container.decode(video=0), None)
        assert first_frame is not None
    assert metadata.get("video_codec")
    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_chunk_encoder_handles_odd_frame_size(
    tmp_path: Path, anyio_backend
) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        fps=2,
        chunk_duration_seconds=1,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)

    await manager.start_recording()

    frame = np.zeros((25, 37, 3), dtype=np.uint8)
    frame[:8, :8, :] = 255
    jpeg_payload = manager._encode_frame(frame)

    await manager._record_frame(jpeg_payload, 0.0, frame)
    await manager._record_frame(jpeg_payload, 0.6, frame)

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True
    metadata = await manager.wait_for_processing()
    assert metadata is not None
    codec_name = metadata.get("video_codec")
    assert isinstance(codec_name, str)
    assert "264" in codec_name or codec_name.startswith("h264")
    file_name = metadata.get("file")
    assert isinstance(file_name, str)
    assert file_name.endswith(".mp4")
    video_path = tmp_path / file_name
    assert video_path.exists()
    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_even_dimension_frame_encoding(
    tmp_path: Path, monkeypatch, anyio_backend
) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        fps=2,
        chunk_duration_seconds=1,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)

    monkeypatch.setattr(recording, "_VIDEO_CODEC_CANDIDATES", ("h264",))
    monkeypatch.setattr(
        recording,
        "_CODECS_REQUIRE_EVEN_DIMENSIONS",
        frozenset({"h264"}),
    )

    captured: dict[str, object] = {}

    class _DummyStream:
        def __init__(self, codec_name: str) -> None:
            self.codec_context = types.SimpleNamespace(
                codec=types.SimpleNamespace(name=codec_name)
            )
            self.width: int = 0
            self.height: int = 0
            self.pix_fmt: str | None = None
            self.time_base = None

        def encode(self, frame=None):  # pragma: no cover - deterministic stub
            return []

    class _DummyContainer:
        def __init__(self, path: str, mode: str = "w", format: str | None = None) -> None:
            self.path = Path(path)
            self.streams: list[_DummyStream] = []
            self._handle = self.path.open("wb")

        def add_stream(self, codec_name: str, rate):
            stream = _DummyStream(codec_name)
            self.streams.append(stream)
            captured["stream"] = stream
            return stream

        def mux(self, packet) -> None:  # pragma: no cover - deterministic stub
            return None

        def close(self) -> None:
            try:
                self._handle.write(b"final")
            finally:
                self._handle.close()

    monkeypatch.setattr(
        recording.RecordingManager, "_select_video_codec", lambda self: "h264"
    )
    monkeypatch.setattr(
        av, "open", lambda path, mode="w", format=None: _DummyContainer(path, mode, format)
    )

    await manager.start_recording()

    frame = np.zeros((25, 37, 3), dtype=np.uint8)
    frame[:8, :8, :] = 255
    jpeg_payload = manager._encode_frame(frame)

    await manager._record_frame(jpeg_payload, 0.0, frame)
    await manager._record_frame(jpeg_payload, 0.6, frame)

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True
    await manager.wait_for_processing()
    await manager.aclose()

    stream = captured.get("stream")
    assert stream is not None
    assert stream.width % 2 == 0
    assert stream.height % 2 == 0
    assert stream.time_base == Fraction(1, 2000)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_manual_recording_disables_motion_when_requested(
    tmp_path: Path, anyio_backend
) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        motion_detection_enabled=True,
        motion_inactivity_timeout_seconds=0.05,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)

    await manager.start_recording(motion_mode=False)
    assert manager.recording_mode == "continuous"

    status = manager.get_motion_status()
    assert status["session_active"] is False
    assert status["session_override"] is True
    assert status["session_state"] is None
    assert status["session_recording"] is False

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True
    await manager.wait_for_processing()
    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_manual_override_persists_when_settings_change(
    tmp_path: Path, anyio_backend
) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        motion_detection_enabled=True,
        motion_inactivity_timeout_seconds=0.05,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)

    await manager.start_recording(motion_mode=False)
    assert manager.recording_mode == "continuous"

    await manager.apply_settings(motion_sensitivity=65)

    status = manager.get_motion_status()
    assert status["enabled"] is False
    assert status["session_override"] is True
    assert status["session_state"] is None

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True
    await manager.wait_for_processing()
    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_recording_manager_motion_override_session(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        motion_detection_enabled=False,
        motion_inactivity_timeout_seconds=0.05,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)
    await manager.start_recording(motion_mode=True)

    assert manager.recording_mode == "motion"

    status = manager.get_motion_status()
    assert status["session_override"] is True
    assert status["session_active"] is True
    assert status["session_state"] == "monitoring"
    assert status["session_recording"] is False

    still = np.zeros((32, 32, 3), dtype=np.uint8)
    await manager._record_frame(b"frame0", 0.0, still)

    status = manager.get_motion_status()
    assert status["session_active"] is True
    assert status["session_state"] in {"monitoring", "recording"}
    assert status["session_recording"] in {True, False}
    assert "post_event_record_seconds" in status

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is False
    assert placeholder.get("motion_events") == 0
    metadata = await manager.wait_for_processing()
    assert metadata is None

    assert manager.recording_mode == "idle"
    status = manager.get_motion_status()
    assert status["session_active"] is False
    assert status["session_state"] is None
    assert status["session_recording"] is False
    assert "post_event_record_seconds" in status
    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_manual_recording_missing_name_finalises(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        motion_detection_enabled=True,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)
    await manager.start_recording(motion_mode=False)

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    await manager._record_frame(b"frame", 0.0, frame)

    async with manager._state_lock:
        manager._recording_name = None

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True
    assert placeholder["name"]

    metadata = await manager.wait_for_processing()
    assert metadata is not None
    assert metadata.get("name")
    assert (tmp_path / f"{metadata['name']}.meta.json").exists()

    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_motion_recording_creates_event_clips(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        motion_detection_enabled=True,
        motion_sensitivity=70,
        motion_inactivity_timeout_seconds=0.05,
        motion_post_event_seconds=0.1,
        fps=6,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)
    await manager.start_recording(motion_mode=True)

    still = np.zeros((32, 32, 3), dtype=np.uint8)
    motion_a = still.copy()
    motion_a[:8, :8, :] = 255
    motion_b = still.copy()
    motion_b[8:, 8:, :] = 255

    await manager._record_frame(b"baseline", 0.0, still)
    await manager._record_frame(b"event1", 0.01, motion_a)
    await manager._record_frame(b"event1-hold", 0.08, still)
    await manager._record_frame(b"event1-end", 0.3, still)
    await manager._record_frame(b"gap", 0.45, still)

    for _ in range(20):
        if manager.is_processing:
            break
        await asyncio.sleep(0.01)
    for _ in range(40):
        if not manager.is_processing:
            break
        await asyncio.sleep(0.02)

    await manager._record_frame(b"event2", 0.6, motion_b)
    await manager._record_frame(b"event2-hold", 0.7, still)
    await manager._record_frame(b"event2-end", 1.0, still)

    for _ in range(20):
        if manager.is_processing:
            break
        await asyncio.sleep(0.01)
    for _ in range(40):
        if not manager.is_processing:
            break
        await asyncio.sleep(0.02)

    result = await manager.stop_recording()
    assert result.get("processing") is True
    await manager.wait_for_processing()
    await manager.aclose()

    metadata_entries = load_recording_metadata(tmp_path)
    assert metadata_entries, "expected motion recordings to be saved"
    names = {
        entry.get("name")
        for entry in metadata_entries
        if isinstance(entry, dict) and entry.get("name")
    }
    indices = {
        entry.get("motion_event_index")
        for entry in metadata_entries
        if isinstance(entry, dict) and entry.get("motion_event_index")
    }
    assert 1 in indices and 2 in indices
    assert any(str(name).endswith(".motion001") for name in names)
    assert any(str(name).endswith(".motion002") for name in names)
    for entry in metadata_entries:
        if not isinstance(entry, dict):
            continue
        motion_meta = entry.get("motion_detection")
        if isinstance(motion_meta, dict) and entry.get("motion_event_index"):
            assert motion_meta.get("event_index") == entry.get("motion_event_index")


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_motion_status_reports_recording_state(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        motion_detection_enabled=True,
        motion_inactivity_timeout_seconds=0.05,
        motion_post_event_seconds=0.2,
        fps=8,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)
    await manager.start_recording(motion_mode=True)

    still = np.zeros((32, 32, 3), dtype=np.uint8)
    motion = still.copy()
    motion[::2, ::2, :] = 255

    await manager._record_frame(b"baseline", 0.0, still)
    status = manager.get_motion_status()
    assert status["session_state"] == "monitoring"

    await manager._record_frame(b"motion", 0.1, motion)
    status = manager.get_motion_status()
    assert status["session_state"] == "recording"

    await manager._record_frame(b"hold", 0.25, still)
    status = manager.get_motion_status()
    assert status["session_state"] == "recording"

    await manager._record_frame(b"post", 0.6, still)
    status = manager.get_motion_status()
    assert status["session_state"] == "recording"

    await manager._record_frame(b"after", 0.9, still)
    status = manager.get_motion_status()
    assert status["session_state"] in {"monitoring", None}

    await manager.stop_recording()
    await manager.wait_for_processing()
    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_recording_manager_writes_compressed_chunks(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        fps=4,
        chunk_duration_seconds=1,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)
    await manager.start_recording()

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for index in range(5):
        await manager._record_frame(f"frame{index}".encode(), index * 0.2, frame)

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True
    metadata = await manager.wait_for_processing()
    assert metadata is not None
    await manager.aclose()

    assert metadata.get("media_type") == "video/mp4"
    file_name = metadata.get("file")
    assert isinstance(file_name, str)
    path = tmp_path / file_name
    assert path.exists()
    assert path.suffix == ".mp4"
    size_bytes = path.stat().st_size
    assert metadata.get("size_bytes") == size_bytes
    assert metadata.get("duration_seconds", 0) > 0

    payload = load_recording_payload(tmp_path, metadata["name"], include_frames=True)
    frames = payload.get("frames", [])
    assert isinstance(frames, list)
    assert frames == []


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_recording_manager_streams_chunk_writes(
    tmp_path: Path, anyio_backend
) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        fps=4,
        chunk_duration_seconds=1,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)

    await manager.start_recording()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for index in range(10):
        await manager._record_frame(f"frame{index}".encode(), index * 0.25, frame)

    await manager.stop_recording()
    await manager.wait_for_processing()
    await manager.aclose()

    mp4_files = sorted((tmp_path / "media").glob("*.mp4"))
    assert mp4_files
    legacy_chunks = sorted((tmp_path / "media").glob("*.chunk*.json*"))
    assert not legacy_chunks


def test_load_recording_payload_upgrades_legacy_mjpeg(tmp_path: Path, monkeypatch) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    metadata = {
        "name": "legacy",
        "chunks": [
            {
                "file": "media/legacy.chunk001.avi",
                "media_type": "video/x-motion-jpeg",
                "size_bytes": 128,
                "codec": "mjpeg",
            }
        ],
        "fps": 5,
    }
    meta_path = tmp_path / "legacy.meta.json"
    meta_path.write_text(json.dumps(metadata))
    avi_path = media_dir / "legacy.chunk001.avi"
    avi_path.write_bytes(b"avi")

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
        target_path.write_bytes(b"mjpeg")
        return {
            "file": target_path.name,
            "media_type": f"multipart/x-mixed-replace; boundary={boundary}",
            "size_bytes": target_path.stat().st_size,
            "codec": "jpeg-fallback",
            "frame_count": 3,
            "duration_seconds": 0.6,
        }

    monkeypatch.setattr(recording, "_remux_mjpeg_video_chunk", _fake_remux)

    payload = recording.load_recording_payload(tmp_path, "legacy", include_frames=False)
    assert payload["media_type"].startswith("multipart/x-mixed-replace")
    assert payload["file"].endswith(".mjpeg")
    assert payload["file"].startswith("media/")
    assert (tmp_path / payload["file"]).exists()

    updated = json.loads(meta_path.read_text())
    assert updated["chunks"][0]["file"].endswith(".mjpeg")
    assert updated["chunks"][0]["codec"] == "jpeg-fallback"


def test_load_recording_payload_upgrades_single_chunk(tmp_path: Path) -> None:
    name = "single"
    legacy_file = f"media/{name}.chunk001.mp4"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    mp4_path = media_dir / f"{name}.mp4"
    mp4_path.write_bytes(b"mp4-data")
    metadata = {
        "name": name,
        "file": legacy_file,
        "chunks": [
            {
                "file": legacy_file,
                "media_type": "video/mp4",
                "size_bytes": len(b"mp4-data"),
            }
        ],
    }
    meta_path = tmp_path / f"{name}.meta.json"
    meta_path.write_text(json.dumps(metadata))

    payload = recording.load_recording_payload(tmp_path, name, include_frames=False)

    assert payload["file"] == f"media/{name}.mp4"
    assert payload.get("chunk_count") == 1
    assert payload["chunks"][0]["file"] == f"media/{name}.mp4"
    assert json.loads(meta_path.read_text())["file"] == f"media/{name}.mp4"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_recording_finalise_cancellation(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        fps=4,
        chunk_duration_seconds=1,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)

    await manager.start_recording()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for index in range(5):
        await manager._record_frame(f"frame{index}".encode(), index * 0.25, frame)

    placeholder = await manager.stop_recording()
    assert placeholder["processing"] is True

    finalise_task = manager._finalise_task
    assert finalise_task is not None
    finalise_task.cancel()

    await asyncio.sleep(0)

    metadata = await manager.wait_for_processing()
    assert metadata is None

    await asyncio.sleep(0)

    processing = await manager.get_processing_metadata()
    assert processing is not None
    if "processing_error" in processing:
        assert processing.get("processing_error") == "cancelled"

    await manager.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"], indirect=True)
async def test_build_recording_video_exports_mp4(tmp_path: Path, anyio_backend) -> None:
    camera = _StaticCamera()
    pipeline = _pipeline()
    manager = RecordingManager(
        camera=camera,
        pipeline=pipeline,
        directory=tmp_path,
        fps=6,
        chunk_duration_seconds=1,
    )

    async def _noop(self):
        return None

    manager._ensure_producer_running = types.MethodType(_noop, manager)

    await manager.start_recording()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for index in range(6):
        frame[:] = (index * 20) % 255
        payload = manager._encode_frame(frame)
        await manager._record_frame(payload, index * (1 / 6), frame)

    metadata = await manager.stop_recording()
    assert metadata["processing"] is True
    finalised = await manager.wait_for_processing()
    assert finalised is not None
    await manager.aclose()

    safe_name, handle = build_recording_video(tmp_path, finalised["name"])
    assert safe_name == finalised["name"]

    header = handle.read(8)
    assert header.startswith(b"\x00\x00\x00")
    handle.seek(0)

    with av.open(handle, mode="r") as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        assert video_streams
        decoded_frames = 0
        for packet in container.demux(video_streams[0]):
            for frame in packet.decode():
                decoded_frames += 1
        assert decoded_frames >= finalised["frame_count"]

    handle.close()


def test_build_recording_video_uses_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "20240103-030303"
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    video_file = media_dir / f"{name}.mp4"
    payload = b"file-path-video"
    video_file.write_bytes(payload)

    def _fake_load(
        directory: Path, requested_name: str, *, include_frames: bool = False
    ) -> dict[str, object]:
        assert directory == tmp_path
        assert requested_name == name
        return {
            "name": name,
            "file": "",
            "file_path": str(video_file),
            "media_type": "video/mp4",
        }

    monkeypatch.setattr(recording, "load_recording_payload", _fake_load)

    safe_name, handle = build_recording_video(tmp_path, name)
    assert safe_name == name
    try:
        assert handle.read() == payload
    finally:
        handle.close()
