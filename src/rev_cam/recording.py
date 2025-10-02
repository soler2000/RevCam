"""Recording helpers for surveillance mode."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

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
            items.append(data)
    return items


def load_recording_payload(directory: Path, name: str) -> dict[str, object]:
    """Load a recording payload by ``name`` from ``directory`` synchronously."""

    safe_name = _safe_recording_name(name)
    path = directory / f"{safe_name}.json"
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Recording not found")
    return json.loads(path.read_text(encoding="utf-8"))


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

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._frame_interval = 1.0 / float(self.fps)

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
        payload = dict(metadata)
        payload["frames"] = frames
        data_path = self.directory / f"{name}.json"
        meta_path = self.directory / f"{name}.meta.json"

        def _write_files() -> None:
            data_path.write_text(json.dumps(payload), encoding="utf-8")
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")

        await asyncio.to_thread(_write_files)
        await self._maybe_stop_producer()
        return metadata

    async def list_recordings(self) -> list[dict[str, object]]:
        return await asyncio.to_thread(load_recording_metadata, self.directory)

    async def get_recording(self, name: str) -> dict[str, object]:
        return await asyncio.to_thread(load_recording_payload, self.directory, name)

    async def remove_recording(self, name: str) -> None:
        safe_name = Path(name).name
        paths = [
            self.directory / f"{safe_name}.json",
            self.directory / f"{safe_name}.meta.json",
        ]

        def _remove() -> None:
            for candidate in paths:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    continue

        await asyncio.to_thread(_remove)

    async def apply_settings(
        self,
        *,
        fps: int | None = None,
        jpeg_quality: int | None = None,
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

                await self._record_frame(jpeg, iteration_start)
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

    async def _record_frame(self, payload: bytes, frame_time: float) -> None:
        async with self._state_lock:
            if not self._recording_active:
                return
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
            if self.max_frames is not None and len(self._recording_frames) >= self.max_frames:
                self._recording_active = False

    def _broadcast(self, payload: bytes) -> None:
        for queue in list(self._subscribers):
            self._offer(queue, payload)

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


__all__ = [
    "RecordingManager",
    "load_recording_metadata",
    "load_recording_payload",
]

