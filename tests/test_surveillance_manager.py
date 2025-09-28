from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rev_cam.surveillance import ClipFilters, SurveillanceManager, SurveillanceSettings


def test_surveillance_manager_creates_and_lists_clips(tmp_path):
    manager = SurveillanceManager(base_path=tmp_path)

    # Defaults are loaded when the settings file does not yet exist.
    settings = manager.load_settings()
    assert isinstance(settings, SurveillanceSettings)
    assert settings.pre_roll_s == 3

    # Persist updated settings and ensure they round-trip.
    updated = manager.save_settings({"storage_max_days": 2, "audio_enabled": True})
    assert updated.storage_max_days == 2
    assert updated.audio_enabled is True

    # Generate a placeholder clip and ensure it is discoverable.
    clip = manager.create_test_clip()
    assert Path(clip.path).exists()
    assert clip.duration_s > 0

    clips, total = manager.list_clips(ClipFilters())
    assert total == 1
    assert clips[0].id == clip.id

    # Export the clip to a ZIP archive for download.
    export_path = manager.export_clips([clip.id])
    assert export_path.exists()
    assert export_path.suffix == ".zip"

    removed = manager.delete_clips([clip.id])
    assert removed == 1

    clips, total = manager.list_clips(ClipFilters())
    assert total == 0
    assert clips == []


def test_manual_record_request_creates_command_file(tmp_path):
    manager = SurveillanceManager(base_path=tmp_path)

    request = manager.request_manual_record(duration_s=12)
    assert request["duration_s"] == 12
    assert isinstance(request["requested_at"], datetime)

    command_path = Path(request["path"])
    assert command_path.exists()
    payload = json.loads(command_path.read_text(encoding="utf-8"))
    assert payload["id"] == request["id"]
    assert payload["duration_s"] == 12

    # Request without explicit duration defaults to None and writes another file.
    request2 = manager.request_manual_record()
    assert request2["duration_s"] is None
    assert Path(request2["path"]).exists()


def test_surveillance_retention(tmp_path):
    manager = SurveillanceManager(base_path=tmp_path)
    earlier = datetime.now(timezone.utc) - timedelta(days=10)

    # Force a short retention window so the generated clip is purged.
    manager.save_settings({"storage_max_days": 1})

    folder = manager.ensure_day_folder(earlier)
    clip_path = folder / "clip-old.mp4"
    clip_path.write_text("old clip", encoding="utf-8")
    thumb_path = folder / "clip-old.jpg"
    thumb_path.write_bytes(b"\xff\xd8\xff\xd9")

    clip = manager.register_clip(
        start_ts=earlier - timedelta(seconds=5),
        end_ts=earlier,
        path=clip_path,
        thumb_path=thumb_path,
        has_audio=False,
        motion_score=0.2,
        settings=manager.load_settings(),
    )

    removed = manager.apply_retention()
    assert removed >= 1
    assert not Path(clip.path).exists()
    clips, total = manager.list_clips(ClipFilters())
    assert total == 0
    assert clips == []
