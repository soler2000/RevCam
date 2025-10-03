from __future__ import annotations

import types

from pathlib import Path

import numpy as np
import pytest

from rev_cam.camera import BaseCamera
from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline
from rev_cam.recording import MotionDetector, RecordingManager


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
    assert len(manager._recording_frames) == 2

    moving = still.copy()
    moving[:4, :4, :] = 255
    await manager._record_frame(b"frame4", 0.25, moving)
    assert len(manager._recording_frames) == 3

    metadata = await manager.stop_recording()
    motion = metadata["motion_detection"]
    assert motion["enabled"] is True
    assert motion["pause_count"] >= 1
    assert motion["skipped_frames"] >= 1
    assert motion["recorded_frames"] == 3
    await manager.aclose()
