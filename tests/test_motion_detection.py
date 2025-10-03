from __future__ import annotations

import types

from pathlib import Path

import numpy as np
import pytest

from rev_cam.camera import BaseCamera
from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline
from rev_cam.recording import MotionDetector, RecordingManager, load_recording_payload


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

    moving = still.copy()
    moving[:4, :4, :] = 255
    await manager._record_frame(b"frame4", 0.25, moving)
    snapshot = manager.get_motion_status()
    assert snapshot["recorded_frames"] == 3

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

    assert metadata["chunk_count"] >= 1
    assert metadata.get("chunk_compression") == "gzip"

    chunk_sizes = []
    for chunk in metadata["chunks"]:
        path = tmp_path / chunk["file"]
        assert path.exists()
        assert path.suffix == ".gz"
        assert chunk.get("compression") == "gzip"
        size_bytes = path.stat().st_size
        assert chunk.get("size_bytes") == size_bytes
        chunk_sizes.append(size_bytes)

    assert metadata["size_bytes"] == sum(chunk_sizes)

    payload = load_recording_payload(tmp_path, metadata["name"])
    frames = payload.get("frames", [])
    assert isinstance(frames, list)
    assert len(frames) == metadata["frame_count"]
    assert frames and "jpeg" in frames[0]
