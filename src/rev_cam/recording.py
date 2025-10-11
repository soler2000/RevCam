"""Recording helpers for surveillance mode."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import time
import math
from fractions import Fraction
from copy import deepcopy
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    IO,
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
)

import av
import simplejpeg

try:  # pragma: no cover - optional dependency on numpy
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency fallback
    _np = None

from .camera import BaseCamera
from .pipeline import FramePipeline


logger = logging.getLogger(__name__)


class RecordingProcessingError(RuntimeError):
    """Raised when a recording failed to produce usable media."""


_VIDEO_CODEC_CANDIDATES: tuple[str, ...] = (
    "libx264",
    "h264",
    "h264_v4l2m2m",
    "h264_omx",
)


_CODECS_REQUIRE_EVEN_DIMENSIONS: frozenset[str] = frozenset(
    {
        "h264",
        "libx264",
        "h264_v4l2m2m",
        "h264_omx",
    }
)


def _normalise_pixel_format(value: object) -> str | None:
    """Return a lower-cased pixel format string when ``value`` is usable."""

    if isinstance(value, str):
        value = value.strip()
        return value.lower() or None
    if isinstance(value, bytes):
        try:
            decoded = value.decode("ascii", errors="ignore").strip()
        except Exception:  # pragma: no cover - defensive decode
            return None
        return decoded.lower() or None
    return None


def _select_stream_pixel_format(
    stream: av.video.stream.VideoStream,
    requested: str | None,
) -> str:
    """Choose a pixel format compatible with ``stream`` and the requested value."""

    requested_format = _normalise_pixel_format(requested) or "yuv420p"
    codec_context = getattr(stream, "codec_context", None)
    available: tuple[str, ...] = ()
    if codec_context is not None:
        pix_fmts = getattr(codec_context, "pix_fmts", None)
        if isinstance(pix_fmts, (list, tuple)):
            collected: list[str] = []
            for item in pix_fmts:
                normalised = _normalise_pixel_format(item)
                if normalised:
                    collected.append(normalised)
            available = tuple(collected)
    if not available:
        return requested_format
    preferred_order: list[str] = [requested_format, "yuv420p", "nv12", "nv21", "p010le"]
    for candidate in preferred_order:
        if candidate and candidate in available:
            return candidate
    return available[0]


def _prepare_frame_for_encoding(array: "_np.ndarray") -> "_np.ndarray | None":
    """Normalise frame arrays before handing them to the encoder."""

    if _np is None:
        return None

    frame_array = _np.asarray(array)
    if frame_array.ndim == 2:
        frame_array = _np.repeat(frame_array[:, :, _np.newaxis], 3, axis=2)
    elif frame_array.ndim == 3:
        if frame_array.shape[2] == 1:
            frame_array = _np.repeat(frame_array, 3, axis=2)
        elif frame_array.shape[2] > 3:
            frame_array = frame_array[:, :, :3]
    else:
        return None

    if frame_array.dtype != _np.uint8:
        frame_array = _np.clip(frame_array, 0, 255).astype(_np.uint8)

    if not bool(getattr(frame_array.flags, "c_contiguous", False)):
        frame_array = _np.ascontiguousarray(frame_array)

    return frame_array


def _codec_profile(codec: str) -> Mapping[str, object]:
    base: dict[str, object] = {
        "format": "mp4",
        "extension": ".mp4",
        "pixel_format": "yuv420p",
        "media_type": "video/mp4",
    }
    if codec.startswith("h264") or codec == "libx264":
        base["container_options"] = {"movflags": "+faststart"}
    return base


def _compose_codec_failure_message(
    last_error: Exception | str | None, attempted_codecs: Sequence[str]
) -> str:
    """Return a detailed codec initialisation failure message with guidance."""

    detail: str | None
    if isinstance(last_error, Exception):
        detail = str(last_error).strip() or None
    elif isinstance(last_error, str):
        detail = last_error.strip() or None
    else:
        detail = None

    if detail is None:
        detail = "No usable surveillance video codec available"

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for codec in attempted_codecs:
        if not codec or codec in seen:
            continue
        seen.add(codec)
        ordered_unique.append(codec)

    attempt_suffix = ""
    if ordered_unique:
        attempt_suffix = f" (attempted codecs: {', '.join(ordered_unique)})"

    message = f"{detail}{attempt_suffix}".rstrip()
    if message.endswith("."):
        message = f"{message} No video file was created."
    else:
        message = f"{message}. No video file was created."

    guidance_parts = [
        (
            "Install FFmpeg with H.264 encoder support (e.g. sudo apt install ffmpeg"
            " libavcodec-extra) and ensure Raspberry Pi hardware video encoding is enabled."
        )
    ]
    if ordered_unique:
        guidance_parts.append(
            f"Confirm that FFmpeg exposes one of: {', '.join(ordered_unique)}."
        )

    guidance = " ".join(guidance_parts).strip()
    if guidance:
        message = f"{message} {guidance}".rstrip()
    return message


def _select_time_base(frame_rate: Fraction) -> Fraction:
    """Choose a container time base that mirrors the frame duration."""

    numerator = frame_rate.numerator
    denominator = frame_rate.denominator
    if numerator <= 0:
        numerator = 1
    if denominator <= 0:
        denominator = 1
    frame_rate_value = Fraction(numerator, denominator)
    if frame_rate_value <= 0:
        return Fraction(1, 30)
    # Prefer a high-resolution reciprocal for very low frame rates so encoded
    # packets advance in millisecond-scale increments. This avoids accelerated
    # playback and improves timing precision when working with slow capture
    # intervals.
    if frame_rate_value <= 5:
        ticks_per_second = frame_rate_value * 1000
        if ticks_per_second.numerator <= 0:
            return Fraction(1, 30)
        time_base = Fraction(ticks_per_second.denominator, ticks_per_second.numerator)
    else:
        time_base = Fraction(denominator, numerator)
    if time_base <= 0:
        return Fraction(1, 30)
    return time_base.limit_denominator(1_000_000)


def _apply_stream_timing(
    stream: av.video.stream.VideoStream,
    frame_rate: Fraction,
    time_base: Fraction,
) -> None:
    """Synchronise stream and codec timing so playback honours real duration."""

    try:
        stream.time_base = time_base
    except Exception:  # pragma: no cover - property may be read-only
        pass

    try:
        stream.rate = frame_rate
    except Exception:  # pragma: no cover - legacy PyAV attribute
        pass

    try:
        stream.average_rate = frame_rate
    except Exception:  # pragma: no cover - best-effort hint
        pass

    codec_context = getattr(stream, "codec_context", None)
    if codec_context is None:
        return

    try:
        codec_context.time_base = time_base
    except Exception:  # pragma: no cover - codec contexts vary
        pass

    try:
        codec_context.framerate = frame_rate
    except Exception:  # pragma: no cover - optional property
        pass

    try:
        codec_context.ticks_per_frame = 1
    except Exception:  # pragma: no cover - property may be absent
        pass


def _extract_multipart_boundary(media_type: str) -> str | None:
    if not isinstance(media_type, str):
        return None
    parts = [segment.strip() for segment in media_type.split(";")]
    if not parts or not parts[0].lower().startswith("multipart/x-mixed-replace"):
        return None
    for segment in parts[1:]:
        if not segment.lower().startswith("boundary="):
            continue
        boundary = segment.split("=", 1)[1].strip().strip('"')
        if boundary:
            return boundary
    return None


def _iter_mjpeg_frames(path: Path, boundary: str) -> Iterable[bytes]:
    marker = f"--{boundary}".encode("ascii", errors="ignore")
    try:
        data = path.read_bytes()
    except OSError:  # pragma: no cover - best-effort read
        return []
    if not marker or marker not in data:
        return []
    payloads: list[bytes] = []
    for part in data.split(marker):
        part = part.strip()
        if not part or part == b"--":
            continue
        header, _, body = part.partition(b"\r\n\r\n")
        if not _:
            continue
        payload = body.rstrip(b"\r\n")
        if payload:
            payloads.append(payload)
    return payloads


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_recording_name(name: str) -> str:
    safe_name = Path(name).name
    if not safe_name:
        raise FileNotFoundError("Recording not found")
    return safe_name
def _load_metadata(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - best-effort parsing
        return None
    return payload if isinstance(payload, dict) else None


def _write_metadata(path: Path, payload: Mapping[str, object]) -> None:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    path.write_text(data, encoding="utf-8")


def _ensure_centralised_asset(
    directory: Path, value: str, *, subdir: str
) -> tuple[str, Path, bool]:
    """Ensure ``value`` points at ``directory/subdir`` and return its relative path.

    If the referenced file exists outside of the target directory, it is moved into
    ``directory/subdir``. When the file cannot be located the function still returns
    the normalised relative path so metadata can be updated consistently.
    """

    if not isinstance(value, str) or not value:
        target_dir = directory / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        return value, target_dir, False

    raw_path = Path(value)
    name = raw_path.name
    target_dir = directory / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / name if name else target_dir

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(directory / raw_path)
        if raw_path.parent != Path(subdir):
            candidates.append(directory / raw_path.name)
    candidates.append(target_path)

    seen: set[Path] = set()
    resolved_path = target_path
    found = False

    for candidate in candidates:
        if not isinstance(candidate, Path):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists() or candidate.is_dir():
            continue
        found = True
        if candidate.parent == target_dir:
            resolved_path = candidate
            break
        target_candidate = target_dir / candidate.name
        if target_candidate == candidate:
            resolved_path = candidate
            break
        try:
            if target_candidate.exists() and target_candidate != candidate:
                target_candidate.unlink()
        except OSError:
            pass
        try:
            candidate.replace(target_candidate)
            resolved_path = target_candidate
            break
        except OSError:
            try:
                shutil.copy2(candidate, target_candidate)
                candidate.unlink()
            except Exception:
                resolved_path = candidate
                break
            else:
                resolved_path = target_candidate
                break

    if not found and name:
        resolved_path = target_dir / name

    try:
        relative = resolved_path.relative_to(directory).as_posix()
    except ValueError:
        relative = (Path(subdir) / name).as_posix() if name else str(value)

    return relative, resolved_path, found


def resolve_recording_media_path(
    directory: Path, payload: Mapping[str, object]
) -> Path:
    """Return the on-disk media path for ``payload`` within ``directory``.

    The helper inspects the ``file_path`` and ``file`` metadata entries, falling
    back to any chunk references and known centralised media locations. When a
    file is located the payload is updated in-place (if mutable) so downstream
    callers observe the normalised relative path.
    """

    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path instance")

    candidate_paths: list[Path] = []
    seen: set[Path] = set()
    target: MutableMapping[str, object] | None = (
        payload if isinstance(payload, MutableMapping) else None
    )

    def _register(path: Path) -> None:
        if path in seen:
            return
        seen.add(path)
        candidate_paths.append(path)

    def _coerce(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else directory / path

    def _add_media_candidates(value: str) -> None:
        if not value:
            return
        base_path = _coerce(value)
        _register(base_path)
        path_obj = Path(value)
        if not path_obj.is_absolute():
            _register(directory / "media" / path_obj.name)

    desktop_value = payload.get("desktop_file") if isinstance(payload, Mapping) else None
    if isinstance(desktop_value, str) and desktop_value:
        _register(_coerce(desktop_value))

    file_path_value = payload.get("file_path") if isinstance(payload, Mapping) else None
    if isinstance(file_path_value, str) and file_path_value:
        _register(_coerce(file_path_value))

    file_value = payload.get("file") if isinstance(payload, Mapping) else None
    if isinstance(file_value, str) and file_value:
        _add_media_candidates(file_value)

    chunks = payload.get("chunks") if isinstance(payload, Mapping) else None
    if isinstance(chunks, Sequence):
        for entry in chunks:
            if not isinstance(entry, Mapping):
                continue
            chunk_file = entry.get("file")
            if isinstance(chunk_file, str) and chunk_file:
                _add_media_candidates(chunk_file)

    name_value = payload.get("name") if isinstance(payload, Mapping) else None
    if isinstance(name_value, str) and name_value:
        base_name = Path(name_value)
        stems: set[str] = {base_name.name}
        if base_name.suffix:
            stems.add(base_name.stem)
        for stem in stems:
            if not stem:
                continue
            stem_path = Path(stem)
            suffix = stem_path.suffix.lower()
            candidates: list[str] = []
            if suffix:
                candidates.append(stem_path.name)
                candidates.append(stem_path.stem)
            else:
                candidates.append(stem)
            for candidate_name in list(candidates):
                base_stem = Path(candidate_name).stem or candidate_name
                for extension in (".mp4", ".mjpeg", ".avi", ".mov"):
                    composed = f"{base_stem}{extension}"
                    _register(directory / "media" / composed)
                    _register(directory / composed)
            for candidate_name in candidates:
                _register(directory / "media" / candidate_name)
                _register(directory / candidate_name)

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            _update_payload_media_fields(target, candidate, directory)
            return candidate
        if ".chunk001" in candidate.name:
            base_name = candidate.name.replace(".chunk001", "", 1)
            alternate = candidate.with_name(base_name)
            if alternate.exists() and alternate.is_file():
                _update_payload_media_fields(target, alternate, directory)
                return alternate
            alt_media = directory / "media" / base_name
            if alt_media.exists() and alt_media.is_file():
                _update_payload_media_fields(target, alt_media, directory)
                return alt_media

    # As a final fallback, probe the central media directory using the recording
    # name and common file extensions even when metadata has not yet been
    # normalised.
    if isinstance(name_value, str) and name_value:
        stem = Path(name_value).stem or name_value
        for extension in (".mp4", ".mjpeg", ".avi", ".mov"):
            candidate = directory / "media" / f"{stem}{extension}"
            if candidate.exists() and candidate.is_file():
                _update_payload_media_fields(target, candidate, directory)
                return candidate
            candidate = directory / f"{stem}{extension}"
            if candidate.exists() and candidate.is_file():
                _update_payload_media_fields(target, candidate, directory)
                return candidate

    raise FileNotFoundError("Recording media file not found")


def _update_payload_media_fields(
    payload: MutableMapping[str, object] | None, path: Path, directory: Path
) -> None:
    if payload is None:
        return
    payload["file_path"] = str(path)
    try:
        relative = path.relative_to(directory).as_posix()
    except ValueError:
        relative = path.name
    payload["file"] = relative


def load_recording_metadata(directory: Path) -> list[dict[str, object]]:
    """Load recording metadata files from ``directory`` synchronously."""

    items: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.meta.json"), reverse=True):
        data = _load_metadata(path)
        if data is None:
            continue
        if isinstance(data, dict):
            data.setdefault("name", path.stem.replace(".meta", ""))
            file_name = data.get("file")
            file_exists = False
            if isinstance(file_name, str):
                file_path = directory / file_name
                if file_path.exists() and file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                    except OSError:
                        size = None
                    else:
                        data.setdefault("size_bytes", int(size))
                        file_exists = True
            if "media_available" not in data:
                data["media_available"] = bool(file_exists)
            elif not file_exists:
                data["media_available"] = False
            processing_error = data.get("processing_error")
            if isinstance(processing_error, str) and processing_error.strip():
                data["media_available"] = False
            chunks = data.get("chunks")
            if isinstance(chunks, list):
                total = 0
                for entry in chunks:
                    if isinstance(entry, Mapping):
                        size = entry.get("size_bytes")
                        if isinstance(size, (int, float)):
                            total += int(size)
                if total and "size_bytes" not in data:
                    data["size_bytes"] = total
            items.append(data)
    return items


def _collect_chunk_sources(
    directory: Path, safe_name: str, metadata: Mapping[str, object] | None
) -> tuple[list[dict[str, object]], list[tuple[dict[str, object], Path]]]:
    chunk_entries: list[dict[str, object]] = []
    sources: list[tuple[dict[str, object], Path]] = []
    if not metadata:
        return chunk_entries, sources
    chunks = metadata.get("chunks")
    if not isinstance(chunks, list):
        return chunk_entries, sources
    for entry in chunks:
        if not isinstance(entry, Mapping):
            continue
        filename = entry.get("file")
        if not isinstance(filename, str):
            continue
        chunk_entry = dict(entry)
        chunk_entries.append(chunk_entry)
        sources.append((chunk_entry, directory / filename))
    return chunk_entries, sources
def load_recording_payload(
    directory: Path, name: str, *, include_frames: bool = True
) -> dict[str, object]:
    """Load a recording payload by ``name`` from ``directory`` synchronously."""

    safe_name = _safe_recording_name(name)
    meta_path = directory / f"{safe_name}.meta.json"
    if not meta_path.exists() or not meta_path.is_file():
        raise FileNotFoundError("Recording not found")
    metadata = _load_metadata(meta_path)
    if metadata is None:
        raise FileNotFoundError("Recording not found")
    if not isinstance(metadata, dict):
        metadata = dict(metadata)
    metadata_dirty = False
    chunk_entries, chunk_sources = _collect_chunk_sources(directory, safe_name, metadata)

    payload: dict[str, object] = dict(metadata)
    payload.setdefault("name", safe_name)
    media_available = bool(payload.get("media_available"))

    if chunk_sources:
        total_size = 0
        normalised_chunks: list[dict[str, object]] = []
        updated_metadata_entries: dict[int, dict[str, object]] = {}
        fps_value = metadata.get("fps") if isinstance(metadata, Mapping) else None
        default_fps = (
            float(fps_value)
            if isinstance(fps_value, (int, float)) and fps_value > 0
            else 10.0
        )
        jpeg_quality_value = metadata.get("jpeg_quality") if isinstance(metadata, Mapping) else None
        jpeg_quality = (
            int(jpeg_quality_value)
            if isinstance(jpeg_quality_value, (int, float)) and jpeg_quality_value > 0
            else 80
        )
        for index, (entry, chunk_path) in enumerate(chunk_sources, start=1):
            chunk_entry = dict(entry)
            file_name_value = chunk_entry.get("file")
            if isinstance(file_name_value, str) and file_name_value:
                normalised_file, resolved_path, _ = _ensure_centralised_asset(
                    directory, file_name_value, subdir="media"
                )
                if normalised_file != file_name_value:
                    chunk_entry["file"] = normalised_file
                    updated_metadata_entries[index - 1] = dict(chunk_entry)
                    metadata_dirty = True
                chunk_path = resolved_path
            media_type_value = chunk_entry.get("media_type")
            if (
                isinstance(media_type_value, str)
                and media_type_value.lower().startswith("video/x-motion-jpeg")
                and chunk_path.suffix.lower() == ".avi"
                and chunk_path.exists()
                and chunk_path.is_file()
            ):
                start_offset_raw = chunk_entry.get("start_offset_seconds")
                start_offset_value = (
                    float(start_offset_raw)
                    if isinstance(start_offset_raw, (int, float))
                    else None
                )
                target_path = chunk_path.with_suffix(".mjpeg")
                try:
                    converted_entry = _remux_mjpeg_video_chunk(
                        chunk_path,
                        target_path=target_path,
                        name=safe_name,
                        index=index,
                        fps=default_fps,
                        jpeg_quality=jpeg_quality,
                        boundary=f"chunk{index:03d}",
                        start_offset=start_offset_value,
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Failed to convert legacy MJPEG chunk %s for %s",
                        index,
                        safe_name,
                    )
                else:
                    try:
                        relative_target = target_path.relative_to(directory)
                        relative_name = relative_target.as_posix()
                    except Exception:
                        relative_name = converted_entry.get("file", target_path.name)
                    else:
                        converted_entry["file"] = relative_name
                    chunk_entry["file"] = str(converted_entry.get("file", target_path.name))
                    chunk_entry.update(converted_entry)
                    chunk_path = directory / chunk_entry["file"]
                    updated_metadata_entries[index - 1] = dict(chunk_entry)
                    metadata_dirty = True
            if not chunk_path.exists() or not chunk_path.is_file():
                file_name = chunk_entry.get("file")
                if isinstance(file_name, str) and ".chunk001" in file_name:
                    base_name = file_name.replace(".chunk001", "", 1)
                    fallback_path = directory / base_name
                    if fallback_path.exists() and fallback_path.is_file():
                        chunk_entry["file"] = base_name
                        chunk_path = fallback_path
                        updated_metadata_entries[index - 1] = dict(chunk_entry)
                        metadata_dirty = True
                if not chunk_path.exists() or not chunk_path.is_file():
                    if isinstance(file_name, str):
                        media_candidate = directory / "media" / Path(file_name).name
                        if media_candidate.exists() and media_candidate.is_file():
                            relative_media = (Path("media") / media_candidate.name).as_posix()
                            chunk_entry["file"] = relative_media
                            chunk_path = media_candidate
                            updated_metadata_entries[index - 1] = dict(chunk_entry)
                            metadata_dirty = True
            if chunk_path.exists() and chunk_path.is_file():
                try:
                    size = chunk_path.stat().st_size
                except OSError:  # pragma: no cover - defensive stat
                    size = chunk_entry.get("size_bytes")
                else:
                    chunk_entry["size_bytes"] = int(size)
                    total_size += int(size)
                media_available = True
            normalised_chunks.append(chunk_entry)
        if primary_chunk := (normalised_chunks[0] if normalised_chunks else None):
            file_name = primary_chunk.get("file")
            if isinstance(file_name, str):
                normalised_file, resolved_path, _ = _ensure_centralised_asset(
                    directory, file_name, subdir="media"
                )
                if normalised_file != file_name:
                    primary_chunk["file"] = normalised_file
                    normalised_chunks[0] = dict(primary_chunk)
                    if isinstance(metadata, dict):
                        raw_chunks = metadata.get("chunks")
                        if isinstance(raw_chunks, list) and raw_chunks:
                            first = raw_chunks[0]
                            if isinstance(first, dict):
                                first["file"] = normalised_file
                        metadata_dirty = True
                if payload.get("file") != normalised_file:
                    payload["file"] = normalised_file
                    if isinstance(metadata, dict):
                        metadata["file"] = normalised_file
                        metadata_dirty = True
                if resolved_path.exists() and resolved_path.is_file():
                    payload["file_path"] = str(resolved_path)
            media_type = primary_chunk.get("media_type")
            if isinstance(media_type, str) and payload.get("media_type") != media_type:
                payload["media_type"] = media_type
                if isinstance(metadata, dict):
                    metadata["media_type"] = media_type
                    metadata_dirty = True
            codec_name = primary_chunk.get("codec")
            if isinstance(codec_name, str) and payload.get("video_codec") != codec_name:
                payload["video_codec"] = codec_name
                if isinstance(metadata, dict):
                    metadata["video_codec"] = codec_name
                    metadata_dirty = True
        if updated_metadata_entries and isinstance(metadata, dict):
            raw_chunks = metadata.get("chunks")
            if isinstance(raw_chunks, list):
                for idx, updated_entry in updated_metadata_entries.items():
                    if 0 <= idx < len(raw_chunks):
                        raw_chunks[idx] = dict(updated_entry)
                metadata_dirty = True
        payload["chunks"] = normalised_chunks
        payload["chunk_count"] = len(normalised_chunks)
        if total_size and "size_bytes" not in payload:
            payload["size_bytes"] = total_size
        if include_frames:
            payload["frames"] = []
        if payload.get("processing_error"):
            media_available = False
        payload["media_available"] = media_available
        if metadata_dirty:
            try:
                _write_metadata(meta_path, metadata)
            except OSError:  # pragma: no cover - best-effort persistence
                pass
        return payload
    file_value = payload.get("file")
    if isinstance(file_value, str):
        normalised_file, resolved_path, found_asset = _ensure_centralised_asset(
            directory, file_value, subdir="media"
        )
        if normalised_file != file_value:
            payload["file"] = normalised_file
            if isinstance(metadata, dict):
                metadata["file"] = normalised_file
                metadata_dirty = True
        file_path = resolved_path
        if not file_path.exists() or not file_path.is_file():
            if ".chunk001" in normalised_file:
                base_name = normalised_file.replace(".chunk001", "", 1)
                candidate = directory / base_name
                if candidate.exists() and candidate.is_file():
                    normalised_file = base_name
                    payload["file"] = base_name
                    file_path = candidate
                    if isinstance(metadata, dict):
                        metadata["file"] = base_name
                        raw_chunks = metadata.get("chunks")
                        if isinstance(raw_chunks, list) and raw_chunks:
                            first = raw_chunks[0]
                            if isinstance(first, dict):
                                first["file"] = base_name
                        metadata_dirty = True
            if not file_path.exists() or not file_path.is_file():
                media_candidate = directory / "media" / Path(normalised_file).name
                if media_candidate.exists() and media_candidate.is_file():
                    normalised_file = (Path("media") / media_candidate.name).as_posix()
                    payload["file"] = normalised_file
                    file_path = media_candidate
                    if isinstance(metadata, dict):
                        metadata["file"] = normalised_file
                        raw_chunks = metadata.get("chunks")
                        if isinstance(raw_chunks, list) and raw_chunks:
                            first = raw_chunks[0]
                            if isinstance(first, dict):
                                first["file"] = normalised_file
                        metadata_dirty = True
        if file_path.exists() and file_path.is_file():
            payload["file_path"] = str(file_path)
            media_available = True
        try:
            size = file_path.stat().st_size
        except OSError:
            size = payload.get("size_bytes")
        else:
            payload["size_bytes"] = int(size)
        preview_file = metadata.get("preview_file") if isinstance(metadata, Mapping) else None
        if isinstance(preview_file, str):
            normalised_preview, preview_path, _ = _ensure_centralised_asset(
                directory, preview_file, subdir="previews"
            )
            if normalised_preview != preview_file and isinstance(metadata, dict):
                metadata["preview_file"] = normalised_preview
                metadata_dirty = True
            if preview_path.exists() and preview_path.is_file():
                payload["preview_file"] = normalised_preview
        media_type = payload.get("media_type")
        if not isinstance(media_type, str):
            payload["media_type"] = "video/mp4"
        if metadata_dirty:
            try:
                _write_metadata(meta_path, metadata)
            except OSError:  # pragma: no cover - best-effort persistence
                pass
    if include_frames:
        payload["frames"] = []
    if payload.get("processing_error"):
        media_available = False
    payload["media_available"] = media_available
    return payload


def iter_recording_frames(
    directory: Path, name: str, metadata: Mapping[str, object] | None = None
) -> Iterable[Mapping[str, object]]:
    """Yield frames for ``name`` sequentially without loading them all at once."""

    safe_name = _safe_recording_name(name)
    base_metadata: Mapping[str, object] | None
    if metadata is None:
        base_metadata = load_recording_payload(
            directory, safe_name, include_frames=False
        )
    else:
        base_metadata = metadata

    file_value = base_metadata.get("file") if isinstance(base_metadata, Mapping) else None
    if isinstance(file_value, str):
        file_path = directory / file_value
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError("Recording not found")
        try:
            container = av.open(str(file_path), mode="r")
        except Exception as exc:  # pragma: no cover - defensive open
            raise FileNotFoundError("Recording not found") from exc
        with container:
            video_stream = None
            for stream in container.streams.video:
                video_stream = stream
                break
            if video_stream is None:
                raise RuntimeError("Recording does not contain a video stream")
            chunk_info = {"file": file_value, "media_type": base_metadata.get("media_type", "video/mp4")}
            for frame in container.decode(video_stream):
                try:
                    array = frame.to_ndarray(format="rgb24")
                except Exception:  # pragma: no cover - defensive decode
                    continue
                yield {"array": array, "chunk": dict(chunk_info)}
        return

    chunk_entries, chunk_sources = _collect_chunk_sources(
        directory, safe_name, base_metadata
    )
    if chunk_sources:
        for entry, chunk_path in chunk_sources:
            if not chunk_path.exists() or not chunk_path.is_file():
                continue
            boundary = None
            if isinstance(entry, Mapping):
                media_type = entry.get("media_type")
                if isinstance(media_type, str):
                    boundary = _extract_multipart_boundary(media_type)
            if boundary:
                for payload in _iter_mjpeg_frames(chunk_path, boundary):
                    try:
                        array = simplejpeg.decode_jpeg(payload, colorspace="RGB")
                    except Exception:  # pragma: no cover - skip invalid frames
                        continue
                    yield {"array": array, "chunk": dict(entry)}
                continue
            try:
                container = av.open(str(chunk_path), mode="r")
            except Exception:  # pragma: no cover - defensive open
                continue
            with container:
                video_stream = None
                for stream in container.streams.video:
                    video_stream = stream
                    break
                if video_stream is None:
                    continue
                for frame in container.decode(video_stream):
                    try:
                        array = frame.to_ndarray(format="rgb24")
                    except Exception:  # pragma: no cover - defensive decode
                        continue
                    yield {"array": array, "chunk": dict(entry)}
        return

    raise FileNotFoundError("Recording not found")


def build_recording_video(
    directory: Path, name: str
) -> tuple[str, IO[bytes]]:
    """Return the recorded MP4 file for ``name`` as a readable handle."""

    safe_name = _safe_recording_name(name)
    payload = load_recording_payload(directory, safe_name, include_frames=False)
    processing_error = payload.get("processing_error") if isinstance(payload, Mapping) else None
    if isinstance(processing_error, str):
        processing_error = processing_error.strip()
        if processing_error:
            raise RecordingProcessingError(processing_error)
    try:
        file_path = resolve_recording_media_path(directory, payload)
    except FileNotFoundError as exc:
        raise FileNotFoundError("Recording media file not found") from exc
    except Exception as exc:  # pragma: no cover - defensive failure
        raise FileNotFoundError("Recording media file not found") from exc
    handle = file_path.open("rb")
    return safe_name, handle


def export_recording_media(
    directory: Path,
    name: str,
    target_directory: Path | None = None,
) -> Path:
    """Copy the recording media for ``name`` into ``target_directory``.

    Returns the destination path of the copied file. ``target_directory`` defaults
    to the user's desktop directory when omitted.
    """

    safe_name = _safe_recording_name(name)
    payload = load_recording_payload(directory, safe_name, include_frames=False)
    processing_error = payload.get("processing_error") if isinstance(payload, Mapping) else None
    if isinstance(processing_error, str):
        processing_error = processing_error.strip()
        if processing_error:
            raise RecordingProcessingError(processing_error)
    file_path = resolve_recording_media_path(directory, payload)
    destination_root = target_directory
    if destination_root is None:
        destination_root = Path.home() / "Desktop"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / file_path.name
    if destination.resolve() == file_path.resolve():
        # Ensure the file exists at the destination even if it's already in the
        # requested directory.
        return destination
    shutil.copy2(file_path, destination)
    return destination


def remove_recording_files(directory: Path, name: str) -> None:
    """Delete recording data, metadata, and derived artefacts for ``name``."""

    safe_name = _safe_recording_name(name)
    candidates: set[Path] = {
        directory / f"{safe_name}.json",
        directory / f"{safe_name}.json.gz",
        directory / f"{safe_name}.meta.json",
        directory / f"{safe_name}.meta.json.gz",
    }
    media_dir = directory / "media"
    preview_dir = directory / "previews"
    search_roots = (directory, media_dir, preview_dir)

    metadata: Mapping[str, object] | None = None
    for meta_path in (
        directory / f"{safe_name}.meta.json",
        directory / f"{safe_name}.meta.json.gz",
        directory / f"{safe_name}.json.gz",
        directory / f"{safe_name}.json",
    ):
        if not meta_path.exists():
            continue
        try:
            payload = _load_json_from_path(meta_path)
        except Exception:  # pragma: no cover - defensive parsing
            continue
        if isinstance(payload, Mapping):
            metadata = payload
            break

    if metadata is not None:
        file_entry = metadata.get("file")
        if isinstance(file_entry, str):
            candidates.add(directory / file_entry)
        chunks = metadata.get("chunks")
        if isinstance(chunks, list):
            for entry in chunks:
                if isinstance(entry, Mapping):
                    filename = entry.get("file")
                    if isinstance(filename, str):
                        candidates.add(directory / filename)

    for pattern in (
        f"{safe_name}.chunk*.mp4",
        f"{safe_name}.chunk*.avi",
        f"{safe_name}.chunk*.mjpeg",
        f"{safe_name}.chunk*.json",
        f"{safe_name}.chunk*.json.gz",
    ):
        for root in search_roots:
            for chunk_path in root.glob(pattern):
                candidates.add(chunk_path)

    for pattern in (
        f"{safe_name}*.mp4",
        f"{safe_name}*.jpg",
        f"{safe_name}*.jpeg",
    ):
        for root in search_roots:
            for artefact in root.glob(pattern):
                candidates.add(artefact)

    for candidate in candidates:
        try:
            candidate.unlink()
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            continue


def purge_recordings(directory: Path, older_than: datetime) -> list[str]:
    """Remove recordings that finished before ``older_than``."""

    removed: list[str] = []
    for meta_path in list(directory.glob("*.meta.json")):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive parsing
            continue
        if not isinstance(metadata, Mapping):
            continue
        ended_at_raw = metadata.get("ended_at")
        if not isinstance(ended_at_raw, str):
            continue
        try:
            ended_at = datetime.fromisoformat(ended_at_raw)
        except ValueError:
            continue
        if ended_at.tzinfo is None:
            ended_at = ended_at.replace(tzinfo=timezone.utc)
        if ended_at >= older_than:
            continue
        name = metadata.get("name") or meta_path.stem.replace(".meta", "")
        remove_recording_files(directory, str(name))
        removed.append(str(name))
    return removed


@dataclass
class MotionDetector:
    """Simple frame differencing motion detector with inactivity pauses."""

    enabled: bool = False
    sensitivity: int = 50
    inactivity_timeout: float = 2.5
    frame_decimation: int = 1
    _previous_frame: "_np.ndarray" | None = field(init=False, default=None)
    _last_active_monotonic: float | None = field(init=False, default=None)
    _paused: bool = field(init=False, default=False)
    _pause_events: int = field(init=False, default=0)
    _skipped_frames: int = field(init=False, default=0)
    _recorded_frames: int = field(init=False, default=0)
    _decimation_drops: int = field(init=False, default=0)
    _decimation_remaining: int = field(init=False, default=0)
    _last_trigger_was_motion: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.sensitivity = self._normalise_sensitivity(self.sensitivity)
        self.inactivity_timeout = self._normalise_timeout(self.inactivity_timeout)
        self.frame_decimation = self._normalise_frame_decimation(self.frame_decimation)

    def configure(
        self,
        *,
        enabled: bool | None = None,
        sensitivity: int | float | None = None,
        inactivity_timeout: float | int | None = None,
        frame_decimation: int | float | None = None,
    ) -> None:
        """Update detector settings and reset state when switching mode."""

        changed = False
        if enabled is not None and bool(enabled) != self.enabled:
            self.enabled = bool(enabled)
            changed = True
        if sensitivity is not None:
            new_sensitivity = self._normalise_sensitivity(sensitivity)
            if new_sensitivity != self.sensitivity:
                self.sensitivity = new_sensitivity
                changed = True
        if inactivity_timeout is not None:
            timeout = self._normalise_timeout(inactivity_timeout)
            if timeout != self.inactivity_timeout:
                self.inactivity_timeout = timeout
                changed = True
        if frame_decimation is not None:
            decimation = self._normalise_frame_decimation(frame_decimation)
            if decimation != self.frame_decimation:
                self.frame_decimation = decimation
                changed = True
        if changed:
            self.reset()

    def reset(self) -> None:
        """Clear cached frame history and counters."""

        self._previous_frame = None
        self._last_active_monotonic = None
        self._paused = False
        self._pause_events = 0
        self._skipped_frames = 0
        self._recorded_frames = 0
        self._decimation_drops = 0
        self._decimation_remaining = 0
        self._last_trigger_was_motion = False

    def should_record(self, frame, timestamp: float) -> bool:
        """Return ``True`` when the frame should be persisted."""

        if not self.enabled or _np is None:
            self._last_active_monotonic = timestamp
            self._paused = False
            self._last_trigger_was_motion = True
            return True

        try:
            array = _np.asarray(frame)
        except Exception:  # pragma: no cover - defensive guard
            self._last_trigger_was_motion = False
            return True

        if array.size == 0:
            self._last_trigger_was_motion = False
            return True

        if array.ndim == 3:
            # Convert to grayscale to reduce noise and processing cost.
            sample = _np.mean(array.astype(_np.float32), axis=2)
        elif array.ndim == 2:
            sample = array.astype(_np.float32)
        else:  # pragma: no cover - defensive guard
            return True

        # Downsample to reduce noise and CPU usage.
        sample = sample[::4, ::4]
        if sample.size == 0:
            return True

        previous = self._previous_frame
        self._previous_frame = sample
        if previous is None or previous.shape != sample.shape:
            self._last_active_monotonic = timestamp
            self._paused = False
            self._decimation_remaining = max(0, self.frame_decimation - 1)
            self._last_trigger_was_motion = False
            return True

        difference = _np.abs(sample - previous)
        activity = float(difference.mean())
        threshold = self._calculate_threshold()
        if activity >= threshold:
            self._last_active_monotonic = timestamp
            if self._paused:
                self._paused = False
                self._decimation_remaining = 0
            self._last_trigger_was_motion = True
        else:
            last_active = self._last_active_monotonic
            if last_active is None:
                self._last_active_monotonic = timestamp
            elif timestamp - last_active >= self.inactivity_timeout:
                if not self._paused:
                    self._paused = True
                    self._pause_events += 1
                self._decimation_remaining = 0
                self._skipped_frames += 1
                self._last_trigger_was_motion = False
                return False
            self._last_trigger_was_motion = False

        if self.frame_decimation > 1:
            if self._decimation_remaining > 0:
                self._decimation_remaining -= 1
                self._decimation_drops += 1
                return False
            self._decimation_remaining = self.frame_decimation - 1

        return True

    def notify_recorded(self) -> None:
        """Record that a frame was persisted while motion detection was enabled."""

        if self.enabled:
            self._recorded_frames += 1

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the detector state."""

        return {
            "enabled": bool(self.enabled),
            "sensitivity": int(self.sensitivity),
            "inactivity_timeout_seconds": float(self.inactivity_timeout),
            "frame_decimation": int(self.frame_decimation),
            "active": (not self.enabled) or (not self._paused),
            "pause_count": int(self._pause_events),
            "skipped_frames": int(self._skipped_frames),
            "recorded_frames": int(self._recorded_frames),
            "decimation_drops": int(self._decimation_drops),
        }

    def triggered_motion(self) -> bool:
        """Return ``True`` when the most recent frame triggered motion detection."""

        return bool(self._last_trigger_was_motion)

    def _calculate_threshold(self) -> float:
        # Larger sensitivity lowers the activity threshold.
        minimum = 1.0
        maximum = 12.0
        scale = (100 - self.sensitivity) / 100.0
        return minimum + (maximum - minimum) * scale

    @staticmethod
    def _normalise_sensitivity(value: int | float | None) -> int:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return 50
        if numeric < 0:
            numeric = 0
        elif numeric > 100:
            numeric = 100
        return numeric

    @staticmethod
    def _normalise_timeout(value: float | int | None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            numeric = 2.5
        if numeric < 0.1:
            numeric = 0.1
        elif numeric > 30.0:
            numeric = 30.0
        return float(numeric)

    @staticmethod
    def _normalise_frame_decimation(value: int | float | None) -> int:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return 1
        if numeric < 1:
            numeric = 1
        elif numeric > 30:
            numeric = 30
        return numeric


class _ChunkWriter(Protocol):
    """Protocol implemented by active chunk writers."""

    name: str
    index: int
    path: Path
    fps: float
    codec: str
    media_type: str
    frame_count: int
    bytes_written: int

    def add_frame(self, array: "_np.ndarray", timestamp: float) -> None:
        ...

    def finalise(self) -> dict[str, object]:
        ...

    def abort(self) -> None:
        ...


@dataclass
class _ActiveChunkWriter:
    """Active FFmpeg-backed chunk that receives frames incrementally."""

    name: str
    index: int
    path: Path
    container: av.container.OutputContainer
    stream: av.video.stream.VideoStream
    time_base: Fraction
    fps: float
    codec: str
    media_type: str
    relative_file: str
    target_width: int
    target_height: int
    pixel_format: str
    frame_count: int = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    bytes_written: int = 0
    last_pts: int = field(init=False, default=-1)
    next_pts: int = field(init=False, default=0)
    _tick_increment: Fraction | None = field(init=False, default=None, repr=False)
    _tick_cursor: Fraction = field(init=False, default=Fraction(0, 1), repr=False)
    _next_tick_cursor: Fraction = field(init=False, default=Fraction(0, 1), repr=False)
    _frame_duration: Fraction | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        resolved_time_base = self._resolve_time_base(self.time_base)
        if resolved_time_base is not None:
            self.time_base = resolved_time_base
        self._initialise_tick_state()

    @staticmethod
    def _coerce_fraction(value: object | None) -> Fraction | None:
        if value is None:
            return None
        try:
            if isinstance(value, Fraction):
                fraction = value
            elif isinstance(value, (int, float)):
                fraction = Fraction(str(value)).limit_denominator(1_000_000)
            else:
                numerator = getattr(value, "numerator", None)
                denominator = getattr(value, "denominator", None)
                if numerator is None or denominator is None:
                    return None
                fraction = Fraction(int(numerator), int(denominator))
        except Exception:  # pragma: no cover - defensive conversion
            return None
        return fraction if fraction > 0 else None

    def _resolve_time_base(
        self, candidate: Fraction | float | int | None
    ) -> Fraction | None:
        values: list[object | None] = []
        values.append(getattr(self.stream, "time_base", None))
        codec_context = getattr(self.stream, "codec_context", None)
        if codec_context is not None:
            values.append(getattr(codec_context, "time_base", None))
        if candidate is not None:
            values.append(candidate)
        for value in values:
            fraction = self._coerce_fraction(value)
            if fraction is not None:
                return fraction
        return None

    def _initialise_tick_state(self) -> None:
        tick_increment = self._determine_tick_increment()
        self._tick_increment = tick_increment
        self._frame_duration = self._determine_frame_duration()
        self._tick_cursor = Fraction(0, 1)
        self._next_tick_cursor = Fraction(0, 1)

    def _determine_tick_increment(self) -> Fraction | None:
        if not isinstance(self.fps, (int, float)) or self.fps <= 0:
            return None
        try:
            frame_rate = Fraction(str(self.fps)).limit_denominator(1_000_000)
        except Exception:  # pragma: no cover - defensive conversion
            return None
        if frame_rate <= 0:
            return None
        time_base_fraction = None
        try:
            time_base_fraction = Fraction(self.time_base)
        except Exception:  # pragma: no cover - defensive conversion
            pass
        if time_base_fraction is None or time_base_fraction <= 0:
            return None
        frame_duration = Fraction(1, 1) / frame_rate
        self._frame_duration = frame_duration
        ticks_per_frame = frame_duration / time_base_fraction
        if ticks_per_frame <= 0:
            return None
        return ticks_per_frame

    def _determine_frame_duration(self) -> Fraction | None:
        if not isinstance(self.fps, (int, float)) or self.fps <= 0:
            return None
        try:
            frame_rate = Fraction(str(self.fps)).limit_denominator(1_000_000)
        except Exception:  # pragma: no cover - defensive conversion
            return None
        if frame_rate <= 0:
            return None
        return Fraction(1, 1) / frame_rate

    def _propagate_time_base(self, value: Fraction) -> None:
        stream = getattr(self, "stream", None)
        if stream is not None:
            try:
                stream.time_base = value
            except Exception:  # pragma: no cover - property may be read-only
                pass
            codec_context = getattr(stream, "codec_context", None)
            if codec_context is not None:
                try:
                    codec_context.time_base = value
                except Exception:  # pragma: no cover - defensive assignment
                    pass

    def _update_time_base(self, new_time_base: Fraction | None) -> None:
        if new_time_base is None or new_time_base <= 0:
            return
        try:
            current_time_base = Fraction(self.time_base)
        except Exception:  # pragma: no cover - defensive conversion
            current_time_base = None
        if current_time_base is None:
            self.time_base = new_time_base
            self._tick_increment = self._determine_tick_increment()
            self._propagate_time_base(new_time_base)
            return
        if new_time_base == current_time_base:
            return
        self._retime_tick_state(current_time_base, new_time_base)
        self._propagate_time_base(new_time_base)

    def _maybe_update_time_base(self) -> None:
        resolved_time_base = self._resolve_time_base(self.time_base)
        self._update_time_base(resolved_time_base)

    def _maybe_update_time_base_from_packet(self, packet: object) -> None:
        packet_time_base = self._coerce_fraction(getattr(packet, "time_base", None))
        if packet_time_base is not None:
            self._update_time_base(packet_time_base)

    def _retime_tick_state(
        self, current_time_base: Fraction, new_time_base: Fraction
    ) -> None:
        if new_time_base <= 0:
            return
        self.time_base = new_time_base
        try:
            cursor_seconds = self._tick_cursor * current_time_base
        except Exception:  # pragma: no cover - defensive conversion
            cursor_seconds = None
        try:
            next_cursor_seconds = self._next_tick_cursor * current_time_base
        except Exception:  # pragma: no cover - defensive conversion
            next_cursor_seconds = None
        try:
            last_pts_seconds = (
                Fraction(self.last_pts, 1) * current_time_base
                if self.last_pts >= 0
                else None
            )
        except Exception:  # pragma: no cover - defensive conversion
            last_pts_seconds = None
        try:
            next_pts_seconds = (
                Fraction(self.next_pts, 1) * current_time_base
                if self.next_pts > 0
                else None
            )
        except Exception:  # pragma: no cover - defensive conversion
            next_pts_seconds = None

        if cursor_seconds is not None:
            self._tick_cursor = cursor_seconds / new_time_base
        if next_cursor_seconds is not None:
            self._next_tick_cursor = next_cursor_seconds / new_time_base
        if last_pts_seconds is not None:
            self.last_pts = self._round_fraction_to_int(
                last_pts_seconds / new_time_base
            )
        if next_pts_seconds is not None:
            self.next_pts = self._round_fraction_to_int(
                next_pts_seconds / new_time_base
            )

        self._tick_increment = self._determine_tick_increment()

    def _compute_pts(
        self, timestamp: float, previous_timestamp: float | None
    ) -> int:
        self._maybe_update_time_base()
        increment = self._tick_increment
        if increment is not None and increment > 0:
            step_multiplier = 1
            frame_duration_fraction = self._frame_duration
            if (
                frame_duration_fraction is not None
                and isinstance(previous_timestamp, (int, float))
            ):
                try:
                    current_value = float(timestamp)
                    previous_value = float(previous_timestamp)
                except Exception:  # pragma: no cover - defensive conversion
                    try:
                        current_value = float(timestamp)
                    except Exception:
                        current_value = 0.0
                    previous_value = current_value
                delta_seconds = current_value - previous_value
                frame_duration_seconds = float(frame_duration_fraction)
                if (
                    frame_duration_seconds > 0
                    and delta_seconds > 0
                    and math.isfinite(delta_seconds)
                ):
                    threshold = frame_duration_seconds * 1.95
                    if delta_seconds >= threshold:
                        approx_frames = int(round(delta_seconds / frame_duration_seconds))
                        if approx_frames < 1:
                            approx_frames = 1
                        step_multiplier = approx_frames
            current_cursor = self._tick_cursor
            if step_multiplier > 1:
                current_cursor = current_cursor + (increment * (step_multiplier - 1))
            pts = self._round_fraction_to_int(current_cursor)
            if pts <= self.last_pts:
                pts = self.last_pts + 1
                current_cursor = Fraction(pts, 1)
            next_cursor = current_cursor + increment
            next_pts = self._round_fraction_to_int(next_cursor)
            if next_pts <= pts:
                next_pts = pts + 1
                next_cursor = Fraction(next_pts, 1)
            self._tick_cursor = next_cursor
            self._next_tick_cursor = next_cursor
            self.next_pts = next_pts
        else:
            pts = self.frame_count
            if pts <= self.last_pts:
                pts = self.last_pts + 1
            self.next_pts = pts + 1
        self.last_pts = pts
        return pts

    @staticmethod
    def _round_fraction_to_int(value: Fraction) -> int:
        if value <= 0:
            return int(value + Fraction(1, 2))
        numerator, denominator = value.numerator, value.denominator
        quotient, remainder = divmod(numerator, denominator)
        if remainder == 0:
            return quotient
        doubled = remainder * 2
        if doubled < denominator:
            return quotient
        if doubled > denominator:
            return quotient + 1
        # Tie: prefer even to minimise drift.
        if quotient % 2 == 0:
            return quotient
        return quotient + 1

    def _refresh_size(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:  # pragma: no cover - best-effort stat
            return
        self.bytes_written = max(self.bytes_written, int(size))

    def add_frame(self, array: "_np.ndarray", timestamp: float) -> None:
        previous_timestamp = self.last_timestamp
        if self.first_timestamp is None:
            self.first_timestamp = float(timestamp)
        self.last_timestamp = float(timestamp)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        target_width = self.target_width or frame.width
        target_height = self.target_height or frame.height
        pixel_format = self.pixel_format or "yuv420p"
        frame = frame.reformat(
            width=target_width, height=target_height, format=pixel_format
        )
        frame.pts = self._compute_pts(timestamp, previous_timestamp)
        frame.time_base = self.time_base
        for packet in self.stream.encode(frame):
            self._maybe_update_time_base_from_packet(packet)
            self.container.mux(packet)
            packet_size = getattr(packet, "size", None)
            if isinstance(packet_size, int) and packet_size > 0:
                self.bytes_written += packet_size
        self.frame_count += 1
        self._refresh_size()

    def finalise(self) -> dict[str, object]:
        for packet in self.stream.encode():
            self._maybe_update_time_base_from_packet(packet)
            self.container.mux(packet)
            packet_size = getattr(packet, "size", None)
            if isinstance(packet_size, int) and packet_size > 0:
                self.bytes_written += packet_size
        self.container.close()
        size = self.bytes_written
        try:
            size = max(size, self.path.stat().st_size)
        except OSError:  # pragma: no cover - best-effort stat
            pass
        self.bytes_written = int(size)
        duration = 0.0
        fps_minimum = 0.0
        if self.first_timestamp is not None and self.last_timestamp is not None:
            try:
                duration = max(0.0, float(self.last_timestamp - self.first_timestamp))
            except Exception:  # pragma: no cover - defensive conversion
                duration = 0.0
        if self.frame_count and self.fps > 0:
            try:
                fps_minimum = float(self.frame_count) / float(self.fps)
            except Exception:  # pragma: no cover - defensive conversion
                fps_minimum = 0.0
        if self._tick_increment and self._tick_increment > 0:
            try:
                time_base_fraction = Fraction(self.time_base).limit_denominator(1_000_000)
                tick_fraction = self._next_tick_cursor * time_base_fraction
                pts_duration = float(tick_fraction)
            except Exception:  # pragma: no cover - defensive conversion
                pts_duration = 0.0
            duration = max(duration, pts_duration)
        elif self.last_pts >= 0:
            try:
                pts_duration = float(self.last_pts + 1) * float(self.time_base)
            except Exception:  # pragma: no cover - defensive conversion
                pts_duration = 0.0
            duration = max(duration, pts_duration)
        if fps_minimum > 0:
            duration = max(duration, fps_minimum)
        entry: dict[str, object] = {
            "file": self.relative_file,
            "frame_count": int(self.frame_count),
            "size_bytes": int(self.bytes_written),
            "duration_seconds": round(duration, 3),
            "media_type": self.media_type,
            "codec": self.codec,
        }
        if self.first_timestamp is not None:
            entry["start_offset_seconds"] = round(float(self.first_timestamp), 3)
        return entry

    def abort(self) -> None:
        """Abort the writer and clean up the partially written file."""

        try:
            self.stream.encode()  # flush buffered packets to avoid PyAV warnings
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
        try:
            self.container.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass
        self.bytes_written = 0
        self.frame_count = 0


def _remux_mjpeg_video_chunk(
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
    """Convert an MJPEG container file into a multipart MJPEG stream."""

    effective_fps = float(fps) if fps and fps > 0 else 10.0
    frame_base = float(start_offset) if isinstance(start_offset, (int, float)) else 0.0
    boundary_bytes = boundary.encode("ascii")
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    frame_count = 0
    with target_path.open("wb") as handle:
        def _write_part(payload: bytes) -> None:
            handle.write(b"--" + boundary_bytes + b"\r\n")
            handle.write(b"Content-Type: image/jpeg\r\n")
            handle.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
            handle.write(payload)
            handle.write(b"\r\n")

        with av.open(str(source_path), mode="r") as container:
            video_stream = None
            for stream in container.streams:
                if getattr(stream, "type", "") == "video":
                    video_stream = stream
                    break
            if video_stream is None:
                raise RuntimeError("MJPEG chunk does not contain a video stream")
            time_base = getattr(video_stream, "time_base", None)
            for frame_index, frame in enumerate(container.decode(video_stream)):
                try:
                    array = frame.to_ndarray(format="rgb24")
                except Exception as exc:
                    raise RuntimeError("Unable to decode MJPEG frame") from exc
                timestamp = None
                pts = getattr(frame, "pts", None)
                if pts is not None and time_base:
                    try:
                        timestamp = float(pts * time_base)
                    except Exception:
                        timestamp = None
                if timestamp is None:
                    frame_time = getattr(frame, "time", None)
                    if frame_time is not None:
                        try:
                            timestamp = float(frame_time)
                        except Exception:
                            timestamp = None
                if timestamp is None:
                    timestamp = frame_index / effective_fps
                absolute_time = frame_base + float(timestamp)
                if first_timestamp is None:
                    first_timestamp = float(absolute_time)
                last_timestamp = float(absolute_time)
                jpeg = simplejpeg.encode_jpeg(
                    array,
                    quality=max(1, min(100, int(jpeg_quality))),
                    colorspace="RGB",
                )
                _write_part(jpeg)
                frame_count += 1
        handle.write(b"--" + boundary_bytes + b"--\r\n")

    if frame_count <= 0:
        try:
            target_path.unlink()
        except OSError:
            pass
        raise RuntimeError("MJPEG chunk did not contain any frames")

    duration = 0.0
    if first_timestamp is not None and last_timestamp is not None:
        duration = max(0.0, float(last_timestamp - first_timestamp))
    if frame_count and effective_fps > 0:
        duration = max(duration, frame_count / effective_fps)

    return {
        "file": target_path.name,
        "frame_count": frame_count,
        "size_bytes": target_path.stat().st_size,
        "duration_seconds": round(duration, 3),
        "media_type": f"multipart/x-mixed-replace; boundary={boundary}",
        "codec": "jpeg-fallback",
        "start_offset_seconds": (
            round(float(first_timestamp), 3)
            if isinstance(first_timestamp, (int, float))
            else None
        ),
    }


@dataclass
class RecordingManager:
    """Capture frames for surveillance recordings with MJPEG preview support."""

    camera: BaseCamera
    pipeline: FramePipeline
    directory: Path
    media_directory: Path = field(init=False)
    preview_directory: Path = field(init=False)
    fps: int = 10
    jpeg_quality: int = 80
    boundary: str = "recording"
    max_frames: int | None = None
    chunk_duration_seconds: int | None = None
    chunk_data_limit_bytes: int | None = None
    storage_threshold_percent: float = 10.0
    motion_detection_enabled: bool = False
    motion_sensitivity: int = 50
    motion_inactivity_timeout_seconds: float = 2.5
    motion_frame_decimation: int = 1
    motion_post_event_seconds: float = 2.0
    on_stop: Callable[[dict[str, object]], Awaitable[None]] | None = None
    _subscribers: set[asyncio.Queue[bytes]] = field(init=False, default_factory=set)
    _producer_task: asyncio.Task[None] | None = field(init=False, default=None)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _state_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _frame_interval: float = field(init=False)
    _recording_active: bool = field(init=False, default=False)
    _active_chunk: _ChunkWriter | None = field(init=False, default=None)
    _chunk_entries: list[dict[str, object]] = field(init=False, default_factory=list)
    _chunk_index: int = field(init=False, default=0)
    _chunk_frame_limit: int = field(init=False, default=0)
    _chunk_byte_limit: int = field(init=False, default=0)
    _total_frame_count: int = field(init=False, default=0)
    _recording_started_at: datetime | None = field(init=False, default=None)
    _recording_started_monotonic: float | None = field(init=False, default=None)
    _recording_name: str | None = field(init=False, default=None)
    _thumbnail: str | None = field(init=False, default=None)
    _next_stop_reason: str | None = field(init=False, default=None)
    _auto_stop_task: asyncio.Task[None] | None = field(init=False, default=None)
    _last_storage_status: dict[str, float | int] | None = field(init=False, default=None)
    _motion_detector: MotionDetector = field(init=False)
    _finalise_task: asyncio.Task[dict[str, object]] | None = field(
        init=False, default=None
    )
    _processing_metadata: dict[str, object] | None = field(
        init=False, default=None
    )
    _last_finalised_metadata: dict[str, object] | None = field(
        init=False, default=None
    )
    _active_finalise_tasks: set[asyncio.Task[dict[str, object]]] = field(
        init=False, default_factory=set
    )
    _session_motion_override: bool = field(init=False, default=False)
    _session_motion_override_enabled: bool | None = field(init=False, default=None)
    _active_motion_enabled: bool = field(init=False, default=False)
    _motion_state: str | None = field(init=False, default=None)
    _motion_event_base: str | None = field(init=False, default=None)
    _motion_event_index: int = field(init=False, default=0)
    _current_motion_event_index: int = field(init=False, default=0)
    _motion_event_active: bool = field(init=False, default=False)
    _motion_event_pending_stop: float | None = field(init=False, default=None)
    _motion_state_hold_until: float | None = field(init=False, default=None)
    _active_chunk_bytes: int = field(init=False, default=0)
    _preferred_video_codec: str | None = field(init=False, default=None)
    _codec_failures: set[str] = field(init=False, default_factory=set)
    _video_encoding_disabled: bool = field(init=False, default=False)
    _last_video_initialisation_error: str | None = field(init=False, default=None)
    _last_failed_codecs: tuple[str, ...] = field(init=False, default=())
    _codec_state_path: Path | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.media_directory = self.directory / "media"
        self.preview_directory = self.directory / "previews"
        self.media_directory.mkdir(parents=True, exist_ok=True)
        self.preview_directory.mkdir(parents=True, exist_ok=True)
        self._frame_interval = 1.0 / float(self.fps)
        self.chunk_duration_seconds = self._normalise_chunk_duration(self.chunk_duration_seconds)
        self.storage_threshold_percent = self._normalise_storage_threshold(self.storage_threshold_percent)
        self.motion_detection_enabled = bool(self.motion_detection_enabled)
        self.motion_sensitivity = MotionDetector._normalise_sensitivity(self.motion_sensitivity)
        self.motion_inactivity_timeout_seconds = MotionDetector._normalise_timeout(
            self.motion_inactivity_timeout_seconds
        )
        self.motion_frame_decimation = MotionDetector._normalise_frame_decimation(
            self.motion_frame_decimation
        )
        self.motion_post_event_seconds = self._normalise_post_event_duration(
            self.motion_post_event_seconds
        )
        self._motion_detector = MotionDetector(
            enabled=self.motion_detection_enabled,
            sensitivity=self.motion_sensitivity,
            inactivity_timeout=self.motion_inactivity_timeout_seconds,
            frame_decimation=self.motion_frame_decimation,
        )
        # Single-file recordings do not rotate, so disable chunk byte/frame limits
        self.chunk_data_limit_bytes = None
        self._chunk_byte_limit = 0
        self._chunk_frame_limit = 0
        self._codec_state_path = self.directory / ".codec_state.json"
        self._load_codec_state()

    def _reset_video_encoding_state(self) -> None:
        """Reset codec failure tracking when a new session begins."""

        previous_error = (
            str(self._last_video_initialisation_error).strip()
            if isinstance(self._last_video_initialisation_error, str)
            else None
        )
        self._video_encoding_disabled = False
        self._last_failed_codecs = ()
        self._last_video_initialisation_error = None
        if previous_error:
            if self._codec_failures:
                self._codec_failures.clear()
            # Persist so the state file drops the stored error message while
            # keeping any preferred codec that previously worked.
            self._persist_codec_state()

    @property
    def media_type(self) -> str:
        return f"multipart/x-mixed-replace; boundary={self.boundary}"

    @property
    def is_recording(self) -> bool:
        return self._recording_active

    @property
    def is_processing(self) -> bool:
        if not self._active_finalise_tasks:
            return False
        return any(not task.done() for task in list(self._active_finalise_tasks))

    @property
    def motion_session_active(self) -> bool:
        return self._recording_active and self._active_motion_enabled

    @property
    def recording_mode(self) -> str:
        if not self._recording_active:
            return "idle"
        return "motion" if self._active_motion_enabled else "continuous"

    @property
    def recording_started_at(self) -> datetime | None:
        return self._recording_started_at

    async def stream(self) -> AsyncGenerator[bytes, None]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        async with self._subscriber(queue):
            while True:
                chunk = await queue.get()
                yield self._render_chunk(chunk)

    async def start_recording(
        self, *, motion_mode: bool | None = None
    ) -> dict[str, object]:
        async with self._state_lock:
            if self._recording_active:
                raise RuntimeError("Recording already in progress")
            started_at = _utcnow()
            monotonic = time.perf_counter()
            name = started_at.strftime("%Y%m%d-%H%M%S")
            if motion_mode is None:
                self._session_motion_override = False
                self._session_motion_override_enabled = None
                motion_enabled = bool(self.motion_detection_enabled)
            else:
                desired_motion = bool(motion_mode)
                self._session_motion_override = True
                self._session_motion_override_enabled = desired_motion
                motion_enabled = desired_motion
            self._active_motion_enabled = motion_enabled
            self._motion_detector.configure(
                enabled=motion_enabled,
                sensitivity=self.motion_sensitivity,
                inactivity_timeout=self.motion_inactivity_timeout_seconds,
                frame_decimation=self.motion_frame_decimation,
            )
            self._motion_detector.reset()
            self._motion_state = "monitoring" if motion_enabled else None
            self._recording_active = True
            self._active_chunk = None
            self._active_chunk_bytes = 0
            self._chunk_entries.clear()
            self._chunk_index = 0
            self._total_frame_count = 0
            self._reset_video_encoding_state()
            if motion_enabled and self._session_motion_override:
                self._motion_event_base = name
                self._motion_event_index = 0
                self._current_motion_event_index = 0
                self._motion_event_active = False
                self._motion_event_pending_stop = None
                self._recording_started_at = None
                self._recording_started_monotonic = None
                self._recording_name = None
            else:
                self._motion_event_base = None
                self._motion_event_index = 0
                self._current_motion_event_index = 0
                self._motion_event_active = False
                self._motion_event_pending_stop = None
                self._recording_started_at = started_at
                self._recording_started_monotonic = monotonic
                self._recording_name = name
            self._thumbnail = None
            self._next_stop_reason = None
            self._cancel_auto_stop_task()
            self._chunk_frame_limit = self._calculate_chunk_frame_limit()
            self._motion_state_hold_until = None

        ensure_task = asyncio.create_task(self._ensure_producer_running())
        try:
            await asyncio.shield(ensure_task)
        except asyncio.CancelledError:
            await ensure_task
            raise
        return {"name": name, "started_at": started_at.isoformat()}

    async def stop_recording(self) -> dict[str, object]:
        base_name: str | None = None
        motion_events_recorded = 0
        context: _FinaliseContext | None
        stop_reason: str | None = None
        async with self._state_lock:
            if not self._recording_active:
                raise RuntimeError("No recording in progress")
            base_name = self._recording_name or self._motion_event_base
            motion_events_recorded = self._motion_event_index
            stop_reason = self._next_stop_reason
            context = self._capture_finalise_context_locked(
                stop_reason=stop_reason,
                deactivate_session=True,
                notify_stop=True,
            )
        if context is None:
            await self._maybe_stop_producer()
            stub: dict[str, object] = {
                "name": base_name,
                "processing": False,
                "frame_count": 0,
                "motion_events": motion_events_recorded,
            }
            if stop_reason:
                stub["stop_reason"] = stop_reason
            return stub

        finalise_task = asyncio.create_task(
            self._process_finalise_context(context, track=True)
        )
        try:
            placeholder = await asyncio.shield(finalise_task)
        except asyncio.CancelledError:
            try:
                await finalise_task
            finally:
                await self._maybe_stop_producer()
            raise
        except Exception:
            await self._maybe_stop_producer()
            raise
        await self._maybe_stop_producer()
        return placeholder

    async def list_recordings(self) -> list[dict[str, object]]:
        records = await asyncio.to_thread(load_recording_metadata, self.directory)
        processing = await self.get_processing_metadata()
        if processing:
            names = {
                str(item.get("name"))
                for item in records
                if isinstance(item, Mapping) and item.get("name")
            }
            if str(processing.get("name")) not in names:
                records.insert(0, processing)
        return records

    async def get_recording(
        self, name: str, *, include_frames: bool = True
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            load_recording_payload, self.directory, name, include_frames=include_frames
        )

    async def remove_recording(self, name: str) -> None:
        await asyncio.to_thread(remove_recording_files, self.directory, name)

    async def _run_recording_finalise(
        self,
        *,
        name: str,
        base_metadata: dict[str, object],
        chunk_entries: list[dict[str, object]],
        invoke_callback: bool = True,
    ) -> dict[str, object]:
        metadata = dict(base_metadata)
        existing_error = metadata.pop("processing_error", None)
        processing_error = self._resolve_processing_error_message(
            existing_error, allow_fallback=False
        )
        entries: list[dict[str, object]] = []
        for entry in chunk_entries:
            if isinstance(entry, Mapping):
                entries.append(dict(entry))

        primary_entry: dict[str, object] | None = entries[0] if entries else None
        size_bytes = 0
        media_available = bool(entries)
        primary_entry_found = False
        if primary_entry is not None:
            file_name = primary_entry.get("file")
            if isinstance(file_name, str):
                normalised_file, resolved_path, found_asset = _ensure_centralised_asset(
                    self.directory, file_name, subdir="media"
                )
                if normalised_file != file_name:
                    primary_entry["file"] = normalised_file
                    entries[0] = dict(primary_entry)
                metadata["file"] = normalised_file
                primary_entry_found = bool(found_asset)
                if resolved_path.exists() and resolved_path.is_file():
                    primary_entry_found = True
                    metadata["file_path"] = str(resolved_path)
            media_type = primary_entry.get("media_type")
            if isinstance(media_type, str):
                metadata["media_type"] = media_type
            codec_name = primary_entry.get("codec")
            if isinstance(codec_name, str):
                metadata["video_codec"] = codec_name
            frame_total = primary_entry.get("frame_count")
            if isinstance(frame_total, (int, float)):
                metadata["frame_count"] = int(frame_total)
            duration_value = primary_entry.get("duration_seconds")
            if isinstance(duration_value, (int, float)):
                metadata["duration_seconds"] = max(
                    float(metadata.get("duration_seconds", 0.0)),
                    float(duration_value),
                )
            size_value = primary_entry.get("size_bytes")
            if isinstance(size_value, (int, float)):
                size_bytes = int(size_value)
        if media_available and not primary_entry_found:
            media_available = False
        if size_bytes:
            metadata["size_bytes"] = size_bytes
        thumbnail_data = metadata.get("thumbnail")
        if isinstance(thumbnail_data, str) and thumbnail_data:
            try:
                preview_bytes = base64.b64decode(thumbnail_data, validate=False)
            except Exception:
                preview_bytes = b""
            if preview_bytes:
                preview_name = f"{name}.jpg"
                preview_path = self.preview_directory / preview_name
                try:
                    preview_path.write_bytes(preview_bytes)
                except OSError:  # pragma: no cover - best-effort write
                    pass
                else:
                    metadata["preview_file"] = (Path("previews") / preview_name).as_posix()
        if processing_error is None and not entries:
            processing_error = self._resolve_processing_error_message(
                None, allow_fallback=True
            )
        if processing_error is None and not media_available:
            processing_error = self._resolve_processing_error_message(
                None, allow_fallback=True
            )
        if processing_error:
            media_available = False
            metadata["processing_error"] = processing_error
        metadata["media_available"] = media_available
        if not media_available:
            metadata.pop("file", None)
            metadata.pop("file_path", None)
            metadata.pop("desktop_file", None)
            self._cleanup_failed_recording_media(name)
        meta_path = self.directory / f"{name}.meta.json"
        try:
            await asyncio.to_thread(_write_metadata, meta_path, metadata)
        except asyncio.CancelledError:  # pragma: no cover - propagate cancellation
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to write surveillance metadata for %s", name)
            processing_error = str(exc)
            metadata["processing_error"] = processing_error
            metadata["media_available"] = False

        if invoke_callback and callable(self.on_stop):
            try:
                await self.on_stop(metadata)
            except Exception:  # pragma: no cover - defensive callback guard
                logger.exception("Surveillance on_stop callback failed for %s", name)

        return metadata

    async def wait_for_processing(self) -> dict[str, object] | None:
        task = self._finalise_task
        if task is not None:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:  # pragma: no cover - differentiate cancellation sources
                if task.cancelled():
                    return None
                raise
            except Exception:  # pragma: no cover - defensive guard
                return None
        async with self._state_lock:
            if self._last_finalised_metadata is None:
                return None
            return deepcopy(self._last_finalised_metadata)

    async def get_processing_metadata(self) -> dict[str, object] | None:
        async with self._state_lock:
            if self._processing_metadata is None:
                return None
            return deepcopy(self._processing_metadata)

    async def apply_settings(
        self,
        *,
        fps: int | None = None,
        jpeg_quality: int | None = None,
        chunk_duration_seconds: int | None = None,
        storage_threshold_percent: float | int | None = None,
        motion_detection_enabled: bool | None = None,
        motion_sensitivity: int | float | None = None,
        motion_frame_decimation: int | float | None = None,
        motion_post_event_seconds: float | int | None = None,
    ) -> None:
        flush_request: _ChunkWriter | None = None
        finalise_context: _FinaliseContext | None = None
        async with self._state_lock:
            chunk_limit_needs_update = False
            if fps is not None and fps > 0 and fps != self.fps:
                self.fps = int(fps)
                self._frame_interval = 1.0 / float(self.fps)
                chunk_limit_needs_update = True
            if jpeg_quality is not None:
                value = int(jpeg_quality)
                if value < 1:
                    value = 1
                elif value > 100:
                    value = 100
                self.jpeg_quality = value
            if chunk_duration_seconds is not None:
                new_duration = self._normalise_chunk_duration(chunk_duration_seconds)
                if new_duration != self.chunk_duration_seconds:
                    self.chunk_duration_seconds = new_duration
                    chunk_limit_needs_update = True
            if storage_threshold_percent is not None:
                self.storage_threshold_percent = self._normalise_storage_threshold(storage_threshold_percent)
            detector_updates: dict[str, object] = {}
            if motion_detection_enabled is not None:
                flag = bool(motion_detection_enabled)
                self.motion_detection_enabled = flag
                detector_updates["enabled"] = flag
            if motion_sensitivity is not None:
                sensitivity = MotionDetector._normalise_sensitivity(motion_sensitivity)
                self.motion_sensitivity = sensitivity
                detector_updates["sensitivity"] = sensitivity
            if motion_frame_decimation is not None:
                decimation = MotionDetector._normalise_frame_decimation(motion_frame_decimation)
                self.motion_frame_decimation = decimation
                detector_updates["frame_decimation"] = decimation
            if motion_post_event_seconds is not None:
                linger = self._normalise_post_event_duration(motion_post_event_seconds)
                self.motion_post_event_seconds = linger
                if self._motion_event_active and linger <= 0:
                    finalise_context = self._finalise_motion_event_locked(
                        stop_reason="motion_inactive"
                    )
                elif self._motion_event_active:
                    self._motion_event_pending_stop = None
            session_override = (
                self._session_motion_override
                and self._session_motion_override_enabled is not None
                and self._recording_active
            )
            effective_enabled = (
                self._session_motion_override_enabled
                if session_override
                else self.motion_detection_enabled
            )
            self._active_motion_enabled = effective_enabled if self._recording_active else False
            if self._recording_active and self._active_motion_enabled and self._motion_state is None:
                self._motion_state = "monitoring"
            elif not self._active_motion_enabled:
                self._motion_state = None
                self._motion_state_hold_until = None
            if detector_updates or session_override:
                updates = dict(detector_updates)
                updates["enabled"] = effective_enabled
                updates.setdefault("sensitivity", self.motion_sensitivity)
                updates.setdefault("frame_decimation", self.motion_frame_decimation)
                self._motion_detector.configure(
                    enabled=updates["enabled"],
                    sensitivity=updates["sensitivity"],
                    inactivity_timeout=self.motion_inactivity_timeout_seconds,
                    frame_decimation=updates["frame_decimation"],
                )
            if chunk_limit_needs_update or not self._chunk_frame_limit:
                self._chunk_frame_limit = self._calculate_chunk_frame_limit()
                if (
                    self._recording_active
                    and self._active_chunk is not None
                    and (
                        (
                            self._chunk_byte_limit
                            and self._active_chunk_bytes >= self._chunk_byte_limit
                        )
                        or (
                            self._chunk_frame_limit
                            and self._active_chunk.frame_count >= self._chunk_frame_limit
                        )
                    )
                ):
                    flush_request = self._pop_active_chunk_locked()

        persisted_chunk: dict[str, object] | None = None
        if flush_request is not None:
            persisted_chunk = await self._persist_chunk(flush_request)
        if finalise_context is not None:
            if persisted_chunk is not None:
                finalise_context.chunk_entries.append(persisted_chunk)
            await self._process_finalise_context(finalise_context, track=False)

    async def aclose(self) -> None:
        await self.wait_for_processing()
        async with self._state_lock:
            pending_tasks = [
                task for task in self._active_finalise_tasks if not task.done()
            ]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        await self._maybe_stop_producer(force=True)
        async with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
            task = self._producer_task
            self._producer_task = None
        for queue in subscribers:
            self._drain_queue(queue)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover
                pass

    async def _maybe_stop_producer(self, force: bool = False) -> None:
        async with self._lock:
            if self._producer_task is None:
                return
            async with self._state_lock:
                active = self._recording_active
            if self._subscribers or active:
                if not force:
                    return
            task = self._producer_task
            self._producer_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover
                pass

    async def _ensure_producer_running(self) -> None:
        async with self._lock:
            if self._producer_task is None or self._producer_task.done():
                self._producer_task = asyncio.create_task(self._produce_frames())

    @asynccontextmanager
    async def _subscriber(self, queue: asyncio.Queue[bytes]):
        await self._register(queue)
        try:
            yield
        finally:
            await self._unregister(queue)

    async def _register(self, queue: asyncio.Queue[bytes]) -> None:
        should_start = False
        async with self._lock:
            self._subscribers.add(queue)
            should_start = self._producer_task is None or self._producer_task.done()
        if should_start:
            await self._ensure_producer_running()

    async def _unregister(self, queue: asyncio.Queue[bytes]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)
            should_stop = not self._subscribers
        if should_stop:
            await self._maybe_stop_producer()

    async def _produce_frames(self) -> None:
        try:
            while True:
                iteration_start = time.perf_counter()
                try:
                    frame = await self.camera.get_frame()
                except asyncio.CancelledError:  # pragma: no cover
                    raise
                except Exception:  # pragma: no cover
                    await asyncio.sleep(self._frame_interval)
                    continue

                try:
                    processed = self.pipeline.process(frame)
                    jpeg = await asyncio.to_thread(self._encode_frame, processed)
                except asyncio.CancelledError:  # pragma: no cover
                    raise
                except Exception:  # pragma: no cover
                    await asyncio.sleep(self._frame_interval)
                    continue

                await self._record_frame(jpeg, iteration_start, processed)
                self._broadcast(jpeg)

                elapsed = time.perf_counter() - iteration_start
                delay = self._frame_interval - elapsed
                if delay > 0:
                    try:
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:  # pragma: no cover
                        raise
        except asyncio.CancelledError:  # pragma: no cover
            pass

    async def _record_frame(self, payload: bytes, frame_time: float, frame) -> None:
        storage_limit_reached = False
        storage_status: dict[str, float | int] | None = None
        should_record = False
        flush_request: _ChunkWriter | None = None
        finalise_context: _FinaliseContext | None = None
        async with self._state_lock:
            if not self._recording_active:
                return
            motion_should_record = self._motion_detector.should_record(frame, frame_time)
            motion_triggered = self._motion_detector.triggered_motion()
            should_record = motion_should_record
            if self._active_motion_enabled:
                now = frame_time
                motion_state_recording = False
                if motion_should_record:
                    motion_state_recording = motion_triggered or motion_state_recording
                    if self._motion_event_base is not None:
                        start_event = False
                        if self._motion_event_active or not self._session_motion_override:
                            start_event = True
                        elif motion_triggered:
                            start_event = True
                        if start_event:
                            if not self._motion_event_active:
                                self._begin_motion_event_locked(now)
                            self._motion_event_pending_stop = None
                            motion_state_recording = True
                        else:
                            should_record = False
                    elif motion_triggered:
                        motion_state_recording = True
                else:
                    if self._motion_event_base is not None and self._motion_event_active:
                        linger = self.motion_post_event_seconds
                        if linger > 0:
                            deadline = self._motion_event_pending_stop
                            if deadline is None:
                                deadline = now + linger
                                self._motion_event_pending_stop = deadline
                            if now <= deadline:
                                should_record = True
                                motion_state_recording = True
                            else:
                                finalise_context = self._finalise_motion_event_locked(
                                    stop_reason="motion_inactive"
                                )
                                should_record = False
                        else:
                            finalise_context = self._finalise_motion_event_locked(
                                stop_reason="motion_inactive"
                            )
                            should_record = False
                    elif motion_triggered:
                        motion_state_recording = True
                if not should_record and not self._motion_event_active:
                    deadline = self._motion_event_pending_stop
                    if deadline is not None and now > deadline:
                        self._motion_event_pending_stop = None
                if (
                    not motion_state_recording
                    and (
                        self._motion_event_active
                        or (
                            self._motion_event_pending_stop is not None
                            and now <= self._motion_event_pending_stop
                        )
                    )
                    ):
                    motion_state_recording = True
                recording_now = motion_state_recording
                if recording_now and self._motion_event_base is None:
                    linger = max(0.5, float(self.motion_post_event_seconds))
                    hold_until = now + linger
                    if (
                        self._motion_state_hold_until is None
                        or hold_until > self._motion_state_hold_until
                    ):
                        self._motion_state_hold_until = hold_until
                elif not recording_now and self._motion_event_base is None:
                    hold_until = self._motion_state_hold_until
                    if hold_until is not None and now <= hold_until:
                        motion_state_recording = True
                    else:
                        self._motion_state_hold_until = None
                self._motion_state = "recording" if motion_state_recording else "monitoring"
            else:
                self._motion_state = None
                self._motion_event_pending_stop = None
                self._motion_state_hold_until = None
            storage_status = self._compute_storage_status()
            if self.storage_threshold_percent > 0:
                free_percent = float(storage_status.get("free_percent", 100.0))
                if free_percent <= self.storage_threshold_percent:
                    storage_limit_reached = True
            if should_record:
                start = self._recording_started_monotonic
                if start is None:
                    timestamp = 0.0
                else:
                    delta = frame_time - start
                    if delta < 0 and frame_time >= 0:
                        timestamp = float(frame_time)
                    else:
                        timestamp = max(0.0, delta)
                if self._thumbnail is None:
                    self._thumbnail = base64.b64encode(payload).decode("ascii")
                self._total_frame_count += 1
                self._motion_detector.notify_recorded()
                chunk_frame_count = 0
                chunk_writer: _ChunkWriter | None = None
                if _np is not None:
                    chunk_writer = self._append_frame_to_chunk_locked(frame, timestamp)
                if chunk_writer is not None:
                    self._active_chunk_bytes = chunk_writer.bytes_written
                    chunk_frame_count = chunk_writer.frame_count
                elif self._active_chunk is not None:
                    self._active_chunk_bytes = self._active_chunk.bytes_written
                    chunk_frame_count = self._active_chunk.frame_count
                if (
                    self._chunk_byte_limit
                    and self._active_chunk_bytes >= self._chunk_byte_limit
                ):
                    flush_request = self._pop_active_chunk_locked()
                elif (
                    self._chunk_frame_limit
                    and chunk_frame_count >= self._chunk_frame_limit
                ):
                    flush_request = self._pop_active_chunk_locked()
                if (
                    self.max_frames is not None
                    and self._total_frame_count >= int(self.max_frames)
                ):
                    self._recording_active = False
            elif storage_limit_reached and self._next_stop_reason is None:
                self._next_stop_reason = "storage_low"
        if storage_status is None:
            storage_status = self._compute_storage_status()
        if storage_limit_reached:
            self._schedule_auto_stop("storage_low")
        persisted_chunk: dict[str, object] | None = None
        if flush_request is not None:
            persisted_chunk = await self._persist_chunk(flush_request)
        if finalise_context is not None:
            if persisted_chunk is not None:
                finalise_context.chunk_entries.append(persisted_chunk)
            await self._process_finalise_context(finalise_context, track=False)
        if not should_record:
            return

    def _broadcast(self, payload: bytes) -> None:
        for queue in list(self._subscribers):
            self._offer(queue, payload)

    def _begin_motion_event_locked(self, frame_time: float) -> None:
        base = self._motion_event_base
        if not base:
            candidate = self._recording_name
            if not candidate:
                candidate = _utcnow().strftime("%Y%m%d-%H%M%S")
            base = candidate
            self._motion_event_base = base
        self._motion_event_index += 1
        self._current_motion_event_index = self._motion_event_index
        event_name = f"{base}.motion{self._motion_event_index:03d}"
        started_at = _utcnow()
        self._recording_name = event_name
        self._recording_started_at = started_at
        self._recording_started_monotonic = frame_time
        self._active_chunk = None
        self._active_chunk_bytes = 0
        self._chunk_entries.clear()
        self._chunk_index = 0
        self._total_frame_count = 0
        self._thumbnail = None
        self._motion_event_active = True
        self._motion_event_pending_stop = None
        self._motion_state_hold_until = None

    def _finalise_motion_event_locked(
        self, *, stop_reason: str | None
    ) -> _FinaliseContext | None:
        if not self._motion_event_active:
            return None
        return self._capture_finalise_context_locked(
            stop_reason=stop_reason,
            deactivate_session=False,
            notify_stop=False,
        )

    def _capture_finalise_context_locked(
        self,
        *,
        stop_reason: str | None,
        deactivate_session: bool,
        notify_stop: bool,
    ) -> _FinaliseContext | None:
        name = self._recording_name
        if not name:
            if not self._active_motion_enabled:
                started_at = self._recording_started_at or _utcnow()
                monotonic = self._recording_started_monotonic or time.perf_counter()
                fallback_name = started_at.strftime("%Y%m%d-%H%M%S")
                self._recording_started_at = started_at
                self._recording_started_monotonic = monotonic
                self._recording_name = fallback_name
                name = fallback_name
            else:
                if deactivate_session:
                    self._recording_active = False
                    self._recording_started_at = None
                    self._recording_started_monotonic = None
                    self._recording_name = None
                    self._thumbnail = None
                    self._next_stop_reason = None
                    self._session_motion_override = False
                    self._session_motion_override_enabled = None
                    self._active_motion_enabled = False
                    self._motion_state = None
                    self._cancel_auto_stop_task()
                    self._total_frame_count = 0
                    self._chunk_index = 0
                    self._active_chunk_bytes = 0
                    self._motion_event_base = None
                    self._motion_event_index = 0
                    self._current_motion_event_index = 0
                    self._motion_event_active = False
                    self._motion_event_pending_stop = None
                return None

        pending_flush = self._pop_active_chunk_locked()
        chunk_entries = list(self._chunk_entries)
        self._chunk_entries.clear()
        storage_status = (
            dict(self._last_storage_status)
            if isinstance(self._last_storage_status, Mapping)
            else None
        )
        processing_error: str | None = None
        if isinstance(self._last_video_initialisation_error, str):
            candidate = self._last_video_initialisation_error.strip()
            if candidate:
                processing_error = candidate
        context = _FinaliseContext(
            name=name,
            started_at=self._recording_started_at or _utcnow(),
            total_frames=self._total_frame_count,
            thumbnail=self._thumbnail,
            stop_reason=stop_reason,
            storage_status=storage_status,
            motion_snapshot=self._motion_detector.snapshot(),
            motion_event_index=self._current_motion_event_index or None,
            notify_stop_callback=notify_stop,
            pending_flush=pending_flush,
            chunk_entries=chunk_entries,
            processing_error=processing_error,
        )
        self._total_frame_count = 0
        self._chunk_index = 0
        self._active_chunk_bytes = 0
        self._recording_started_at = None
        self._recording_started_monotonic = None
        self._recording_name = None
        self._thumbnail = None
        self._motion_event_active = False
        self._motion_event_pending_stop = None
        self._current_motion_event_index = 0
        if deactivate_session:
            self._recording_active = False
            self._session_motion_override = False
            self._session_motion_override_enabled = None
            self._active_motion_enabled = False
            self._motion_state = None
            self._next_stop_reason = None
            self._motion_event_base = None
            self._motion_event_index = 0
            self._cancel_auto_stop_task()
        else:
            self._motion_state = "monitoring" if self._active_motion_enabled else None
        return context

    async def _process_finalise_context(
        self, context: _FinaliseContext, *, track: bool
    ) -> dict[str, object]:
        pending_flush = context.pending_flush
        if pending_flush is not None:
            await self._persist_chunk(pending_flush, task_list=context.chunk_entries)
        finished_at = _utcnow()
        duration = max(0.0, (finished_at - context.started_at).total_seconds())
        base_metadata: dict[str, object] = {
            "name": context.name,
            "started_at": context.started_at.isoformat(),
            "ended_at": finished_at.isoformat(),
            "duration_seconds": duration,
            "fps": self.fps,
            "frame_count": context.total_frames,
            "thumbnail": context.thumbnail,
        }
        if context.stop_reason:
            base_metadata["stop_reason"] = context.stop_reason
        if context.storage_status:
            base_metadata["storage_status"] = context.storage_status
        motion_snapshot = dict(context.motion_snapshot)
        if context.motion_event_index:
            base_metadata["motion_event_index"] = context.motion_event_index
            motion_snapshot.setdefault("event_index", context.motion_event_index)
        base_metadata["motion_detection"] = motion_snapshot
        processing_error = self._resolve_processing_error_message(
            context.processing_error, allow_fallback=False
        )
        media_available = bool(context.chunk_entries)
        if processing_error is None and not context.chunk_entries:
            processing_error = self._resolve_processing_error_message(
                None, allow_fallback=True
            )
        if processing_error:
            media_available = False
            base_metadata["processing_error"] = processing_error
        base_metadata["media_available"] = media_available
        processing_stub = dict(base_metadata)
        processing_stub["processing"] = True

        async with self._state_lock:
            self._processing_metadata = processing_stub

        task = asyncio.create_task(
            self._run_recording_finalise(
                name=context.name,
                base_metadata=base_metadata,
                chunk_entries=context.chunk_entries,
                invoke_callback=context.notify_stop_callback,
            )
        )
        async with self._state_lock:
            self._active_finalise_tasks.add(task)
            if track:
                self._finalise_task = task
                self._last_finalised_metadata = None
        task.add_done_callback(
            lambda done_task, recording_name=context.name: asyncio.create_task(
                self._finalise_task_cleanup(recording_name, done_task)
            )
        )
        return processing_stub

    async def _finalise_task_cleanup(
        self, recording_name: str, task: asyncio.Task[dict[str, object]]
    ) -> None:
        metadata: dict[str, object] | None
        processing_error = False
        try:
            result = task.result()
        except asyncio.CancelledError:
            metadata = {
                "name": recording_name,
                "processing_error": "cancelled",
            }
            processing_error = True
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Surveillance finalise task failed for %s", recording_name)
            message = str(exc).strip() or "finalise_failed"
            metadata = {"name": recording_name, "processing_error": message}
            processing_error = True
        else:
            if isinstance(result, Mapping):
                metadata = dict(result)
                processing_error = bool(metadata.get("processing_error"))
            else:  # pragma: no cover - defensive guard
                metadata = {"name": recording_name, "processing_error": "finalise_failed"}
                processing_error = True

        async with self._state_lock:
            self._active_finalise_tasks.discard(task)
            if metadata is not None:
                self._last_finalised_metadata = dict(metadata)
            if (
                self._processing_metadata
                and str(self._processing_metadata.get("name")) == recording_name
            ):
                if metadata is None:
                    self._processing_metadata = {
                        "name": recording_name,
                        "processing_error": "finalise_failed",
                    }
                elif processing_error:
                    self._processing_metadata = metadata
                else:
                    self._processing_metadata = None
            if self._finalise_task is task:
                self._finalise_task = None

    def get_motion_status(self) -> dict[str, object]:
        status = self._motion_detector.snapshot()
        status["session_active"] = self.motion_session_active
        status["session_override"] = self._session_motion_override
        status["session_state"] = self._motion_state
        status["event_active"] = self._motion_event_active
        status["session_recording"] = self._motion_state == "recording"
        status["post_event_record_seconds"] = float(self.motion_post_event_seconds)
        return status

    def _offer(self, queue: asyncio.Queue[bytes], payload: bytes) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:  # pragma: no cover
            self._drain_queue(queue)
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover
                pass

    def _drain_queue(self, queue: asyncio.Queue[bytes]) -> None:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover
            return

    def _encode_frame(self, frame) -> bytes:
        from .streaming import encode_frame_to_jpeg

        return encode_frame_to_jpeg(frame, quality=self.jpeg_quality)

    def _render_chunk(self, payload: bytes) -> bytes:
        header = (
            f"--{self.boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "\r\n"
        ).encode("ascii")
        return header + payload + b"\r\n"

    def _calculate_chunk_frame_limit(self) -> int:
        # Recording now uses a single MP4 container, so chunk rotation is disabled.
        return 0

    def _chunk_filename(self, name: str, index: int, extension: str) -> str:
        if not extension.startswith("."):
            extension = f".{extension}"
        if index <= 1:
            return f"{name}{extension}"
        return f"{name}.chunk{index:03d}{extension}"

    def _select_video_codec(self) -> str | None:
        if self._video_encoding_disabled:
            return None
        codec = self._preferred_video_codec
        if codec and codec not in self._codec_failures:
            return codec
        last_error: Exception | None = None
        for candidate in _VIDEO_CODEC_CANDIDATES:
            if candidate in self._codec_failures:
                continue
            try:
                context = av.CodecContext.create(candidate, "w")
            except av.FFmpegError as exc:  # pragma: no cover - codec probing failure
                last_error = exc
                self._mark_codec_failed(candidate)
                continue
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = exc
                self._mark_codec_failed(candidate)
                continue
            if not getattr(context, "is_encoder", True):
                self._mark_codec_failed(candidate)
                continue
            self._preferred_video_codec = candidate
            return candidate
        if last_error is not None:
            logger.warning("No usable surveillance video codec available: %s", last_error)
        return None

    def _mark_codec_failed(self, codec: str) -> None:
        if not codec:
            return
        if codec in self._codec_failures:
            if self._preferred_video_codec == codec:
                self._preferred_video_codec = None
            return
        self._codec_failures.add(codec)
        if self._preferred_video_codec == codec:
            self._preferred_video_codec = None
        self._persist_codec_state()

    def _record_failed_codecs(self, attempted_codecs: Sequence[str] | None) -> None:
        ordered_unique: list[str] = []
        seen: set[str] = set()
        if attempted_codecs:
            for codec in attempted_codecs:
                if not codec or codec in seen:
                    continue
                seen.add(codec)
                ordered_unique.append(codec)
        self._last_failed_codecs = tuple(ordered_unique)

    def _resolve_processing_error_message(
        self, error: str | None, *, allow_fallback: bool
    ) -> str | None:
        candidate = error.strip() if isinstance(error, str) else None
        attempted = self._last_failed_codecs or tuple(sorted(self._codec_failures))
        if candidate:
            if "No video file was created." in candidate:
                return candidate
            return _compose_codec_failure_message(candidate, attempted)
        if not allow_fallback:
            return None
        fallback = self._last_video_initialisation_error
        detail = fallback.strip() if isinstance(fallback, str) else None
        if detail:
            if "No video file was created." in detail:
                return detail
            return _compose_codec_failure_message(detail, attempted)
        if self._video_encoding_disabled or attempted:
            return _compose_codec_failure_message(None, attempted)
        return None

    def _handle_chunk_failure(self, chunk: _ChunkWriter) -> None:
        try:
            chunk.abort()
        finally:
            self._active_chunk = None
            self._active_chunk_bytes = 0
            self._chunk_index = max(0, int(chunk.index) - 1)

    def _get_or_create_chunk(
        self, array: "_np.ndarray"
    ) -> _ChunkWriter | None:
        chunk = self._active_chunk
        if chunk is not None:
            return chunk
        name = self._recording_name
        if not name:
            return None
        next_index = self._chunk_index + 1
        chunk = self._create_chunk_writer(name, next_index, array)
        if chunk is None:
            return None
        self._chunk_index = next_index
        self._active_chunk = chunk
        return chunk

    def _create_chunk_writer(
        self, name: str, index: int, array: "_np.ndarray"
    ) -> _ChunkWriter | None:
        if self._video_encoding_disabled:
            return None
        frame_rate = 1.0 if self.fps <= 0 else float(self.fps)
        frame_rate_fraction = Fraction(str(frame_rate)).limit_denominator(1000)
        time_base = _select_time_base(frame_rate_fraction)
        height, width = array.shape[:2]
        filename = self._chunk_filename(name, index, "mp4")
        relative_file = Path("media") / filename
        path = self.media_directory / filename
        last_error: Exception | None = None
        attempted_codecs: list[str] = []
        if path.exists():
            try:
                path.unlink()
            except OSError:  # pragma: no cover - defensive cleanup
                pass
        while True:
            codec_name = self._select_video_codec()
            if codec_name is None:
                if last_error is not None:
                    logger.error(
                        "Disabling surveillance recording for %s: %s", name, last_error
                    )
                message = _compose_codec_failure_message(last_error, attempted_codecs)
                self._video_encoding_disabled = True
                self._last_video_initialisation_error = message
                self._record_failed_codecs(attempted_codecs)
                self._persist_codec_state()
                return None
            attempted_codecs.append(codec_name)
            try:
                codec_profile = _codec_profile(codec_name)
                filename = self._chunk_filename(name, index, codec_profile["extension"])
                relative_file = Path("media") / filename
                path = self.media_directory / filename
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:  # pragma: no cover - defensive cleanup
                        pass
                options = codec_profile.get("container_options")
                if not isinstance(options, Mapping):
                    options = None
                container = av.open(
                    str(path),
                    mode="w",
                    format=codec_profile["format"],
                    options=options,
                )
            except Exception as exc:  # pragma: no cover - filesystem failure
                last_error = exc
                logger.error("Failed to open chunk container %s: %s", path, exc)
                self._video_encoding_disabled = True
                self._last_video_initialisation_error = _compose_codec_failure_message(
                    exc, attempted_codecs
                )
                self._record_failed_codecs(attempted_codecs)
                self._persist_codec_state()
                return None
            try:
                stream = container.add_stream(codec_name, rate=frame_rate_fraction)
            except Exception as exc:  # pragma: no cover - codec failure
                last_error = exc
                self._mark_codec_failed(codec_name)
                try:
                    container.close()
                except Exception:  # pragma: no cover - defensive close
                    pass
                try:
                    if path.exists():
                        path.unlink()
                except OSError:  # pragma: no cover - defensive cleanup
                    pass
                continue
            target_width = int(width)
            target_height = int(height)
            if codec_name in _CODECS_REQUIRE_EVEN_DIMENSIONS:
                if target_width % 2:
                    target_width -= 1
                if target_height % 2:
                    target_height -= 1
            if target_width <= 0 or target_height <= 0:
                last_error = ValueError("invalid frame dimensions for video encoder")
                self._mark_codec_failed(codec_name)
                try:
                    container.close()
                except Exception:  # pragma: no cover - defensive close
                    pass
                try:
                    if path.exists():
                        path.unlink()
                except OSError:  # pragma: no cover - defensive cleanup
                    pass
                continue
            stream.width = target_width
            stream.height = target_height
            requested_pixel_format = str(codec_profile.get("pixel_format", "yuv420p"))
            pixel_format = _select_stream_pixel_format(stream, requested_pixel_format)
            stream.pix_fmt = pixel_format
            _apply_stream_timing(stream, frame_rate_fraction, time_base)
            try:
                stream.codec_context.pix_fmt = pixel_format
            except Exception:  # pragma: no cover - codec context may not expose pix_fmt
                pass
            self._record_failed_codecs(())
            return _ActiveChunkWriter(
                name=name,
                index=index,
                path=path,
                container=container,
                stream=stream,
                time_base=time_base,
                fps=frame_rate,
                codec=codec_name,
                media_type=codec_profile.get("media_type", "video/mp4"),
                target_width=target_width,
                target_height=target_height,
                relative_file=relative_file.as_posix(),
                pixel_format=pixel_format,
            )

    def _cleanup_failed_recording_media(self, name: str) -> None:
        base = Path(name)
        stems: set[str] = set()
        if base.name:
            stems.add(base.name)
        if base.stem:
            stems.add(base.stem)
        if not stems:
            return
        search_roots = {self.media_directory, self.directory}
        extensions = (".mp4", ".mjpeg", ".avi", ".mov")
        for stem in stems:
            for root in search_roots:
                for extension in extensions:
                    candidate = root / f"{stem}{extension}"
                    try:
                        if candidate.exists() and candidate.is_file():
                            candidate.unlink()
                    except OSError:  # pragma: no cover - best-effort cleanup
                        logger.debug(
                            "Failed to remove incomplete media file %s", candidate
                        )

    def _append_frame_to_chunk_locked(
        self, frame_array: "_np.ndarray", timestamp: float
    ) -> _ChunkWriter | None:
        if _np is None:
            return None
        array = _prepare_frame_for_encoding(frame_array)
        if array is None:
            return None
        attempts = 0
        max_attempts = max(1, len(_VIDEO_CODEC_CANDIDATES))
        while attempts < max_attempts:
            chunk = self._get_or_create_chunk(array)
            if chunk is None:
                return None
            stream = getattr(chunk, "stream", None)
            try:
                chunk.add_frame(array, timestamp)
            except av.FFmpegError as exc:  # pragma: no cover - codec failure path
                logger.warning(
                    "Video codec %s failed while encoding chunk %s for %s: %s",
                    chunk.codec,
                    chunk.index,
                    chunk.name,
                    exc,
                )
                if isinstance(chunk, _ActiveChunkWriter):
                    self._mark_codec_failed(chunk.codec)
                self._handle_chunk_failure(chunk)
                attempts += 1
                continue
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to encode surveillance frame for %s chunk %s",
                    chunk.name,
                    chunk.index,
                )
                if isinstance(chunk, _ActiveChunkWriter):
                    self._mark_codec_failed(chunk.codec)
                self._handle_chunk_failure(chunk)
                return None
            return chunk
        return None
    def _pop_active_chunk_locked(self) -> _ChunkWriter | None:
        chunk = self._active_chunk
        if chunk is None or chunk.frame_count <= 0:
            return None
        self._active_chunk = None
        self._active_chunk_bytes = 0
        return chunk

    async def _persist_chunk(
        self,
        chunk: _ChunkWriter,
        *,
        task_list: list[dict[str, object]] | None = None,
    ) -> dict[str, object] | None:
        try:
            entry = await asyncio.to_thread(chunk.finalise)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to finalise surveillance chunk %s for %s",
                chunk.index,
                chunk.name,
            )
            return None
        entry = await self._prepare_chunk_entry(chunk, entry)
        if task_list is not None:
            task_list.append(entry)
            note_entry = entry
        else:
            async with self._state_lock:
                self._chunk_entries.append(entry)
            note_entry = entry
        if isinstance(chunk, _ActiveChunkWriter) and chunk.codec:
            if chunk.codec not in self._codec_failures:
                if self._preferred_video_codec != chunk.codec:
                    self._preferred_video_codec = chunk.codec
                    self._persist_codec_state()
        return note_entry

    async def _prepare_chunk_entry(
        self, chunk: _ChunkWriter, entry: dict[str, object]
    ) -> dict[str, object]:
        return entry

    def _codec_state_file(self) -> Path | None:
        return self._codec_state_path

    def _load_codec_state(self) -> None:
        path = self._codec_state_file()
        if path is None:
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:  # pragma: no cover - best-effort parsing
            return
        failed = payload.get("failed_codecs")
        if isinstance(failed, list):
            for item in failed:
                if isinstance(item, str):
                    self._codec_failures.add(item)
        preferred = payload.get("preferred_codec")
        if (
            isinstance(preferred, str)
            and preferred
            and preferred not in self._codec_failures
        ):
            self._preferred_video_codec = preferred
        last_error = payload.get("last_error")
        if isinstance(last_error, str) and last_error.strip():
            self._last_video_initialisation_error = last_error.strip()

    def _persist_codec_state(self) -> None:
        path = self._codec_state_file()
        if path is None:
            return
        payload: dict[str, object] = {
            "failed_codecs": sorted(self._codec_failures),
        }
        codec = self._preferred_video_codec
        if codec and codec not in self._codec_failures:
            payload["preferred_codec"] = codec
        error_message = self._last_video_initialisation_error
        if isinstance(error_message, str) and error_message.strip():
            payload["last_error"] = error_message.strip()
        try:
            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            path.write_text(data, encoding="utf-8")
        except OSError:  # pragma: no cover - best-effort persistence
            pass

    def _normalise_chunk_duration(self, value: int | float | None) -> int | None:
        if value in (None, "", 0):
            return None
        try:
            duration = int(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive branch
            return None
        if duration <= 0:
            return None
        if duration > 24 * 60 * 60:
            return 24 * 60 * 60
        return duration

    def _normalise_chunk_byte_limit(self, value: int | float | None) -> int | None:
        if value in (None, "", 0):
            return None
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive branch
            return None
        if numeric < 1_048_576:
            numeric = 1_048_576
        elif numeric > 268_435_456:
            numeric = 268_435_456
        return numeric

    def _normalise_storage_threshold(self, value: float | int | None) -> float:
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive branch
            return self.storage_threshold_percent
        if numeric < 0:
            numeric = 0.0
        if numeric > 90:
            numeric = 90.0
        return numeric

    def _normalise_post_event_duration(self, value: float | int | None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive branch
            return 0.0
        if numeric < 0.0:
            numeric = 0.0
        elif numeric > 60.0:
            numeric = 60.0
        return float(numeric)

    def _compute_storage_status(self) -> dict[str, float | int]:
        usage = shutil.disk_usage(self.directory)
        total = int(getattr(usage, "total", 0))
        free = int(getattr(usage, "free", 0))
        used = int(getattr(usage, "used", total - free))
        free_percent = 100.0 if total <= 0 else max(0.0, min(100.0, (free / total) * 100.0))
        status = {
            "total_bytes": total,
            "free_bytes": free,
            "used_bytes": used,
            "free_percent": round(free_percent, 3),
        }
        self._last_storage_status = status
        return status

    def get_storage_status(self) -> dict[str, float | int]:
        if self._last_storage_status is None:
            return self._compute_storage_status()
        return dict(self._last_storage_status)

    def _schedule_auto_stop(self, reason: str) -> None:
        if self._next_stop_reason is None:
            self._next_stop_reason = reason
        if self._auto_stop_task is None or self._auto_stop_task.done():
            self._auto_stop_task = asyncio.create_task(self._auto_stop())

    async def _auto_stop(self) -> None:
        try:
            await self.stop_recording()
        except RuntimeError:
            pass
        except Exception:  # pragma: no cover - defensive logging
            pass
        finally:
            self._auto_stop_task = None

    def _cancel_auto_stop_task(self) -> None:
        task = self._auto_stop_task
        if task is not None and not task.done():
            task.cancel()
        self._auto_stop_task = None


__all__ = [
    "MotionDetector",
    "RecordingManager",
    "RecordingProcessingError",
    "build_recording_video",
    "export_recording_media",
    "resolve_recording_media_path",
    "iter_recording_frames",
    "load_recording_metadata",
    "load_recording_payload",
    "remove_recording_files",
    "purge_recordings",
]

@dataclass
class _FinaliseContext:
    """Context required to finalise a recording asynchronously."""

    name: str
    started_at: datetime
    total_frames: int
    thumbnail: str | None
    stop_reason: str | None
    chunk_entries: list[dict[str, object]]
    storage_status: dict[str, float | int] | None
    motion_snapshot: dict[str, object]
    motion_event_index: int | None
    notify_stop_callback: bool
    pending_flush: _ChunkWriter | None = None
    processing_error: str | None = None

