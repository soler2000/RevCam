from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from rev_cam.surveillance import ClipFilters, MotionRecorder, SurveillanceManager


def test_motion_recorder_emits_clip_on_motion(tmp_path):
    manager = SurveillanceManager(base_path=tmp_path)
    settings = manager.save_settings(
        {
            "record_on_motion": True,
            "pre_roll_s": 0,
            "post_motion_gap_s": 0.1,
            "min_motion_duration_ms": 0,
            "clip_max_length_s": 2,
            "min_changed_area_percent": 0.05,
            "sensitivity": 1.0,
            "detection_fps": 30,
        }
    )

    recorder = MotionRecorder(manager)
    recorder.update_settings(settings)

    now = datetime.now(timezone.utc)
    base_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    motion_frame = base_frame.copy()
    motion_frame[:, :80] = 255

    # Prime the background with still frames.
    for index in range(3):
        recorder.process_frame(base_frame, timestamp=now + timedelta(seconds=index * 0.05))

    result = None
    # Feed motion frames to trigger recording.
    for index in range(5):
        recorder.process_frame(
            motion_frame,
            timestamp=now + timedelta(seconds=0.2 + index * 0.05),
        )

    # Supply idle frames so the recorder finalises the clip.
    for index in range(5):
        result = recorder.process_frame(
            base_frame,
            timestamp=now + timedelta(seconds=0.5 + index * 0.1),
        )
        if result is not None:
            break

    assert result is not None
    assert Path(result.path).exists()

    clips, total = manager.list_clips(ClipFilters())
    assert total == 1
    assert clips[0].id == result.id
