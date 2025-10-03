"""Recording helpers for surveillance mode."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable, Mapping

try:  # pragma: no cover - optional dependency on numpy
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency fallback
    _np = None

from .camera import BaseCamera
from .pipeline import FramePipeline


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_recording_name(name: str) -> str:
    safe_name = Path(name).name
    if not safe_name:
        raise FileNotFoundError("Recording not found")
    return safe_name


def load_recording_metadata(directory: Path) -> list[dict[str, object]]:
    """Load recording metadata files from ``directory`` synchronously."""

    items: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.meta.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - best-effort parsing
            continue
        if isinstance(data, dict):
            data.setdefault("name", path.stem.replace(".meta", ""))
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


def load_recording_payload(directory: Path, name: str) -> dict[str, object]:
    """Load a recording payload by ``name`` from ``directory`` synchronously."""

    safe_name = _safe_recording_name(name)
    metadata: dict[str, object] | None = None
    meta_path = directory / f"{safe_name}.meta.json"
    if meta_path.exists() and meta_path.is_file():
        try:
            candidate = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - best-effort parsing
            metadata = None
        else:
            metadata = candidate if isinstance(candidate, dict) else None
    frames: list[dict[str, object]] = []
    chunk_files: list[Path] = []
    if metadata and isinstance(metadata.get("chunks"), list):
        for entry in metadata["chunks"]:  # type: ignore[index]
            if isinstance(entry, Mapping):
                filename = entry.get("file")
                if isinstance(filename, str):
                    chunk_path = directory / filename
                    chunk_files.append(chunk_path)
    if chunk_files:
        for chunk_path in chunk_files:
            if not chunk_path.exists() or not chunk_path.is_file():
                continue
            try:
                chunk_payload = json.loads(chunk_path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover - best-effort parsing
                continue
            if isinstance(chunk_payload, Mapping):
                chunk_frames = chunk_payload.get("frames")
            else:
                chunk_frames = chunk_payload
            if isinstance(chunk_frames, list):
                frames.extend(
                    [frame for frame in chunk_frames if isinstance(frame, Mapping)]
                )
    else:
        data_path = directory / f"{safe_name}.json"
        if not data_path.exists() or not data_path.is_file():
            raise FileNotFoundError("Recording not found")
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            raw_frames = payload.get("frames")
            if isinstance(raw_frames, list):
                frames = [frame for frame in raw_frames if isinstance(frame, Mapping)]
            else:
                frames = []
            if metadata is None:
                metadata = {key: value for key, value in payload.items() if key != "frames"}
        elif isinstance(payload, list):
            frames = [frame for frame in payload if isinstance(frame, Mapping)]
        else:
            frames = []
    payload: dict[str, object]
    if metadata is not None:
        payload = dict(metadata)
    else:
        payload = {"name": safe_name}
    payload.setdefault("name", safe_name)
    payload["frames"] = frames
    return payload


def remove_recording_files(directory: Path, name: str) -> None:
    """Delete recording data and metadata for ``name``."""

    safe_name = _safe_recording_name(name)
    paths = [
        directory / f"{safe_name}.json",
        directory / f"{safe_name}.meta.json",
    ]
    metadata: dict[str, object] | None = None
    meta_path = paths[1]
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive parsing
            metadata = None
    if isinstance(metadata, Mapping):
        chunks = metadata.get("chunks")
        if isinstance(chunks, list):
            for entry in chunks:
                if isinstance(entry, Mapping):
                    filename = entry.get("file")
                    if isinstance(filename, str):
                        paths.append(directory / filename)
    for candidate in paths:
        try:
            candidate.unlink()
        except FileNotFoundError:
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

    def should_record(self, frame, timestamp: float) -> bool:
        """Return ``True`` when the frame should be persisted."""

        if not self.enabled or _np is None:
            self._last_active_monotonic = timestamp
            self._paused = False
            return True

        try:
            array = _np.asarray(frame)
        except Exception:  # pragma: no cover - defensive guard
            return True

        if array.size == 0:
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
            return True

        difference = _np.abs(sample - previous)
        activity = float(difference.mean())
        threshold = self._calculate_threshold()
        if activity >= threshold:
            self._last_active_monotonic = timestamp
            if self._paused:
                self._paused = False
                self._decimation_remaining = 0
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
                return False

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


@dataclass
class RecordingManager:
    """Capture frames for surveillance recordings with MJPEG preview support."""

    camera: BaseCamera
    pipeline: FramePipeline
    directory: Path
    fps: int = 10
    jpeg_quality: int = 80
    boundary: str = "recording"
    max_frames: int | None = None
    chunk_duration_seconds: int | None = None
    storage_threshold_percent: float = 10.0
    motion_detection_enabled: bool = False
    motion_sensitivity: int = 50
    motion_inactivity_timeout_seconds: float = 2.5
    motion_frame_decimation: int = 1
    on_stop: Callable[[dict[str, object]], Awaitable[None]] | None = None
    _subscribers: set[asyncio.Queue[bytes]] = field(init=False, default_factory=set)
    _producer_task: asyncio.Task[None] | None = field(init=False, default=None)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _state_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _frame_interval: float = field(init=False)
    _recording_active: bool = field(init=False, default=False)
    _recording_frames: list[dict[str, object]] = field(init=False, default_factory=list)
    _recording_started_at: datetime | None = field(init=False, default=None)
    _recording_started_monotonic: float | None = field(init=False, default=None)
    _recording_name: str | None = field(init=False, default=None)
    _thumbnail: str | None = field(init=False, default=None)
    _next_stop_reason: str | None = field(init=False, default=None)
    _auto_stop_task: asyncio.Task[None] | None = field(init=False, default=None)
    _last_storage_status: dict[str, float | int] | None = field(init=False, default=None)
    _motion_detector: MotionDetector = field(init=False)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.directory.mkdir(parents=True, exist_ok=True)
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
        self._motion_detector = MotionDetector(
            enabled=self.motion_detection_enabled,
            sensitivity=self.motion_sensitivity,
            inactivity_timeout=self.motion_inactivity_timeout_seconds,
            frame_decimation=self.motion_frame_decimation,
        )

    @property
    def media_type(self) -> str:
        return f"multipart/x-mixed-replace; boundary={self.boundary}"

    @property
    def is_recording(self) -> bool:
        return self._recording_active

    async def stream(self) -> AsyncGenerator[bytes, None]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        async with self._subscriber(queue):
            while True:
                chunk = await queue.get()
                yield self._render_chunk(chunk)

    async def start_recording(self) -> dict[str, object]:
        async with self._state_lock:
            if self._recording_active:
                raise RuntimeError("Recording already in progress")
            started_at = _utcnow()
            monotonic = time.perf_counter()
            name = started_at.strftime("%Y%m%d-%H%M%S")
            self._recording_active = True
            self._recording_frames = []
            self._recording_started_at = started_at
            self._recording_started_monotonic = monotonic
            self._recording_name = name
            self._thumbnail = None
            self._next_stop_reason = None
            self._cancel_auto_stop_task()
            self._motion_detector.reset()

        await self._ensure_producer_running()
        return {"name": name, "started_at": started_at.isoformat()}

    async def stop_recording(self) -> dict[str, object]:
        async with self._state_lock:
            if not self._recording_active:
                raise RuntimeError("No recording in progress")
            name = self._recording_name or _utcnow().strftime("%Y%m%d-%H%M%S")
            started_at = self._recording_started_at or _utcnow()
            frames = list(self._recording_frames)
            thumbnail = self._thumbnail
            self._recording_active = False
            self._recording_frames = []
            self._recording_started_at = None
            self._recording_started_monotonic = None
            self._recording_name = None
            self._thumbnail = None
            stop_reason = self._next_stop_reason
            self._next_stop_reason = None
            self._cancel_auto_stop_task()

        finished_at = _utcnow()
        duration = max(0.0, (finished_at - started_at).total_seconds())
        metadata = {
            "name": name,
            "started_at": started_at.isoformat(),
            "ended_at": finished_at.isoformat(),
            "duration_seconds": duration,
            "fps": self.fps,
            "frame_count": len(frames),
            "thumbnail": thumbnail,
        }
        meta_path = self.directory / f"{name}.meta.json"

        chunks = self._chunk_frames(frames, self.fps)
        chunk_entries: list[dict[str, object]] = []
        total_size = 0

        async def _write_chunk(filename: str, payload: dict[str, object]) -> int:
            def _write() -> int:
                path = self.directory / filename
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path.stat().st_size

            return await asyncio.to_thread(_write)

        if not chunks:
            chunks = [frames]

        if len(chunks) == 1:
            chunk_filename = f"{name}.json"
            payload = {"frames": chunks[0]}
            size = await _write_chunk(chunk_filename, payload)
            chunk_entries.append(
                {"file": chunk_filename, "frame_count": len(chunks[0]), "size_bytes": size}
            )
            total_size = size
        else:
            for index, group in enumerate(chunks, start=1):
                chunk_filename = f"{name}.chunk{index:03d}.json"
                payload = {"frames": group}
                size = await _write_chunk(chunk_filename, payload)
                chunk_entries.append(
                    {"file": chunk_filename, "frame_count": len(group), "size_bytes": size}
                )
                total_size += size

        if stop_reason:
            metadata["stop_reason"] = stop_reason
        if self.chunk_duration_seconds:
            metadata["chunk_duration_seconds"] = self.chunk_duration_seconds
        metadata["chunk_count"] = len(chunks)
        metadata["chunks"] = chunk_entries
        metadata["size_bytes"] = total_size
        if self._last_storage_status:
            metadata["storage_status"] = self._last_storage_status
        metadata["motion_detection"] = self._motion_detector.snapshot()

        def _write_metadata() -> None:
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")

        await asyncio.to_thread(_write_metadata)
        await self._maybe_stop_producer()
        if callable(self.on_stop):
            try:
                await self.on_stop(metadata)
            except Exception:  # pragma: no cover - defensive callback guard
                pass
        return metadata

    async def list_recordings(self) -> list[dict[str, object]]:
        return await asyncio.to_thread(load_recording_metadata, self.directory)

    async def get_recording(self, name: str) -> dict[str, object]:
        return await asyncio.to_thread(load_recording_payload, self.directory, name)

    async def remove_recording(self, name: str) -> None:
        await asyncio.to_thread(remove_recording_files, self.directory, name)

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
    ) -> None:
        async with self._state_lock:
            if fps is not None and fps > 0 and fps != self.fps:
                self.fps = int(fps)
                self._frame_interval = 1.0 / float(self.fps)
            if jpeg_quality is not None:
                value = int(jpeg_quality)
                if value < 1:
                    value = 1
                elif value > 100:
                    value = 100
                self.jpeg_quality = value
            if chunk_duration_seconds is not None:
                self.chunk_duration_seconds = self._normalise_chunk_duration(chunk_duration_seconds)
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
            if detector_updates:
                if "enabled" not in detector_updates:
                    detector_updates["enabled"] = self.motion_detection_enabled
                self._motion_detector.configure(**detector_updates)

    async def aclose(self) -> None:
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
        async with self._state_lock:
            if not self._recording_active:
                return
            should_record = self._motion_detector.should_record(frame, frame_time)
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
                    timestamp = max(0.0, frame_time - start)
                encoded = base64.b64encode(payload).decode("ascii")
                if self._thumbnail is None:
                    self._thumbnail = encoded
                frame_entry: dict[str, object] = {
                    "timestamp": round(timestamp, 4),
                    "jpeg": encoded,
                }
                self._recording_frames.append(frame_entry)
                self._motion_detector.notify_recorded()
                if self.max_frames is not None and len(self._recording_frames) >= self.max_frames:
                    self._recording_active = False
            elif storage_limit_reached and self._next_stop_reason is None:
                self._next_stop_reason = "storage_low"
        if storage_status is None:
            storage_status = self._compute_storage_status()
        if storage_limit_reached:
            self._schedule_auto_stop("storage_low")
        if not should_record:
            return

    def _broadcast(self, payload: bytes) -> None:
        for queue in list(self._subscribers):
            self._offer(queue, payload)

    def get_motion_status(self) -> dict[str, object]:
        return self._motion_detector.snapshot()

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

    def _chunk_frames(self, frames: list[dict[str, object]], fps: int) -> list[list[dict[str, object]]]:
        duration = self.chunk_duration_seconds
        if not frames:
            return []
        if duration is None or duration <= 0 or fps <= 0:
            return [frames]
        chunk_size = max(1, int(round(fps * duration)))
        return [frames[index : index + chunk_size] for index in range(0, len(frames), chunk_size)]

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
    "load_recording_metadata",
    "load_recording_payload",
    "remove_recording_files",
    "purge_recordings",
]

