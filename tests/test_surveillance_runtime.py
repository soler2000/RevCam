from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from rev_cam.surveillance import ClipFilters, SurveillanceManager, SurveillanceRuntime


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

    runtime = SurveillanceRuntime(manager)
    runtime.notify_settings_updated(settings)
    runtime.set_active(True)

    now = datetime.now(timezone.utc)
    base_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    motion_frame = base_frame.copy()
    motion_frame[:, :80] = 255

    for index in range(10):
        runtime.ingest_frame(base_frame, timestamp=now + timedelta(seconds=index * 0.05))

    motion_start = now + timedelta(seconds=0.5)
    for index in range(10):
        runtime.ingest_frame(
            motion_frame,
            timestamp=motion_start + timedelta(seconds=index * 0.05),
        )

    settle_start = motion_start + timedelta(seconds=0.5)
    for index in range(20):
        runtime.ingest_frame(
            base_frame,
            timestamp=settle_start + timedelta(seconds=index * 0.05),
        )

    runtime.set_active(False)

    clips, total = manager.list_clips(ClipFilters())
    assert total == 1
    clip = clips[0]
    assert Path(clip.path).exists()
    assert Path(clip.path).stat().st_size > 0
