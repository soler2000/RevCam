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
    chunk_name = f"{name}.chunk001.mp4"
    metadata = {
        "name": name,
        "chunks": [
            {
                "file": chunk_name,
                "frame_count": 1,
                "size_bytes": 2,
                "media_type": "video/mp4",
            },
        ],
    }

    meta_path = tmp_path / f"{name}.meta.json"
    _write(meta_path, json.dumps(metadata))
    _write(tmp_path / chunk_name)
    _write(tmp_path / f"{name}.mp4")
    _write(tmp_path / f"{name}.thumbnail.jpeg")

    remove_recording_files(tmp_path, name)

    for suffix in (".meta.json", ".mp4", ".thumbnail.jpeg"):
        assert not (tmp_path / f"{name}{suffix}").exists()
    assert not (tmp_path / chunk_name).exists()


def test_remove_recording_files_handles_missing_metadata(tmp_path: Path) -> None:
    name = "legacy"
    orphan_chunk = tmp_path / f"{name}.chunk005.json"
    _write(orphan_chunk)

    remove_recording_files(tmp_path, name)

    assert not orphan_chunk.exists()
