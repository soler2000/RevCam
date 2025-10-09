"""Tests for recording cleanup helpers."""

from __future__ import annotations

import json
from pathlib import Path

from rev_cam.recording import remove_recording_files


def _write(path: Path, data: bytes | str = b"data") -> None:
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def test_remove_recording_files_removes_known_artifacts(tmp_path: Path) -> None:
    name = "example"
    video_name = f"{name}.mp4"
    media_dir = tmp_path / "media"
    preview_dir = tmp_path / "previews"
    media_dir.mkdir()
    preview_dir.mkdir()
    metadata = {
        "name": name,
        "file": f"media/{video_name}",
        "frame_count": 1,
        "size_bytes": 2,
        "media_type": "video/mp4",
        "preview_file": f"previews/{name}.jpg",
    }

    meta_path = tmp_path / f"{name}.meta.json"
    _write(meta_path, json.dumps(metadata))
    _write(media_dir / video_name)
    _write(media_dir / f"{name}.chunk001.mp4")
    _write(preview_dir / f"{name}.jpg")

    remove_recording_files(tmp_path, name)

    assert not meta_path.exists()
    assert not (media_dir / video_name).exists()
    assert not (media_dir / f"{name}.chunk001.mp4").exists()
    assert not (preview_dir / f"{name}.jpg").exists()


def test_remove_recording_files_handles_missing_metadata(tmp_path: Path) -> None:
    name = "legacy"
    orphan_chunk = tmp_path / f"{name}.chunk005.json"
    _write(orphan_chunk)

    remove_recording_files(tmp_path, name)

    assert not orphan_chunk.exists()
