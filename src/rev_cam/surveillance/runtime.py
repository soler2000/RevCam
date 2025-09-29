"""Runtime helpers coordinating manual and motion-triggered recording."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

import logging

import numpy as np

from .manager import ClipRecord, SurveillanceManager
from .settings import SurveillanceSettings
from .video import VideoEncoder, ensure_rgb_frame, write_thumbnail

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameSample:
    """A single frame captured for surveillance processing."""

    timestamp: datetime
    frame: np.ndarray


class FrameBuffer:
    """In-memory ring buffer retaining recent frames for pre-roll."""

    def __init__(self, capacity_seconds: float = 30.0) -> None:
        self._capacity = float(max(0.0, capacity_seconds))
        self._frames: deque[FrameSample] = deque()
        self._lock = Lock()

    def configure(self, capacity_seconds: float) -> None:
        with self._lock:
            self._capacity = float(max(0.0, capacity_seconds))
            self._prune(datetime.now(timezone.utc))

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def add(self, frame: np.ndarray, timestamp: datetime) -> FrameSample:
        sample = FrameSample(timestamp=timestamp, frame=frame)
        with self._lock:
            self._frames.append(sample)
            self._prune(timestamp)
        return sample

    def snapshot(self, seconds: float, *, up_to: datetime) -> List[FrameSample]:
        if seconds <= 0:
            return []
        cutoff = up_to - timedelta(seconds=float(seconds))
        with self._lock:
            return [FrameSample(item.timestamp, item.frame) for item in self._frames if item.timestamp >= cutoff]

    def _prune(self, reference: datetime) -> None:
        if self._capacity <= 0:
            self._frames.clear()
            return
        cutoff = reference - timedelta(seconds=self._capacity)
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()


class ActiveRecording:
    """Persist frames to disk while updating clip metadata."""

    def __init__(
        self,
        *,
        clip: ClipRecord,
        manager: SurveillanceManager,
        settings: SurveillanceSettings,
        fps: int,
        encoding: str,
        reason: str,
        fallback_size: tuple[int, int],
        target_end_ts: datetime | None = None,
        motion_score: float | None = None,
    ) -> None:
        self.clip = clip
        self._manager = manager
        self._settings = settings
        self._fps = max(1, int(fps))
        self._encoding = encoding
        self._reason = reason
        self._fallback_size = (int(fallback_size[0]), int(fallback_size[1]))
        self._target_end_ts = target_end_ts
        self._motion_score = motion_score
        self._encoder: VideoEncoder | None = None
        self._first_frame: np.ndarray | None = None
        self._start_ts: datetime | None = None
        self._last_ts: datetime | None = None
        self._closed = False

    # ------------------------------------------------------------------
    def add_samples(self, samples: Iterable[FrameSample]) -> None:
        for sample in samples:
            self.add_sample(sample)

    def add_sample(self, sample: FrameSample) -> None:
        if self._closed:
            return
        frame = sample.frame
        normalised = ensure_rgb_frame(frame, even=True)
        if self._encoder is None:
            height, width = normalised.shape[:2]
            if width == 0 or height == 0:
                width, height = self._fallback_size
                normalised = np.zeros((height, width, 3), dtype=np.uint8)
            self._encoder = VideoEncoder(
                path=Path(self.clip.path),
                fps=self._fps,
                encoding=self._encoding,
                width=width,
                height=height,
            )
        if self._first_frame is None:
            self._first_frame = ensure_rgb_frame(frame, even=False)
            self._start_ts = sample.timestamp
        self._encoder.encode(normalised)
        self._last_ts = sample.timestamp

    def should_stop(self, timestamp: datetime) -> bool:
        return self._target_end_ts is not None and timestamp >= self._target_end_ts

    def finish(self, *, end_ts: datetime | None = None, motion_score: float | None = None) -> ClipRecord:
        if self._closed:
            return self.clip
        if self._encoder is None:
            width, height = self._fallback_size
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            placeholder = FrameSample(timestamp=datetime.now(timezone.utc), frame=blank)
            self.add_sample(placeholder)
        if motion_score is not None:
            self._motion_score = motion_score
        if self._encoder is not None:
            self._encoder.close()
        thumb_path = Path(self.clip.thumb_path) if self.clip.thumb_path else None
        if thumb_path is not None and self._first_frame is not None:
            try:
                write_thumbnail(thumb_path, self._first_frame)
            except Exception:  # pragma: no cover - thumbnail failures should not break recording
                logger.exception("Failed to write surveillance thumbnail")
        start_ts = self._start_ts or datetime.now(timezone.utc)
        final_end_ts = end_ts or self._last_ts or start_ts
        size_bytes = Path(self.clip.path).stat().st_size if Path(self.clip.path).exists() else 0
        updated = self._manager.update_clip_entry(
            self.clip.id,
            start_ts=start_ts,
            end_ts=final_end_ts,
            size_bytes=size_bytes,
            has_audio=self._settings.audio_enabled,
            thumb_path=thumb_path,
            motion_score=self._motion_score,
            settings=self._settings,
        )
        self.clip = updated
        self._closed = True
        return updated

    def cancel(self) -> None:
        if self._closed:
            return
        try:
            self._manager.delete_clips([self.clip.id])
        finally:
            self._closed = True

    @property
    def clip_id(self) -> int:
        return self.clip.id

    @property
    def reason(self) -> str:
        return self._reason


@dataclass(slots=True)
class MotionSample:
    """Summary of motion analysis for a frame."""

    changed_ratio: float
    motion_detected: bool


class MotionRecorder:
    """Detect motion in surveillance frames and manage recordings."""

    def __init__(self, runtime: "SurveillanceRuntime") -> None:
        self._runtime = runtime
        self._settings = runtime.load_settings()
        self._state: str = "idle"
        self._background: np.ndarray | None = None
        self._last_sample_ts: datetime | None = None
        self._motion_started: datetime | None = None
        self._recording_clip_id: int | None = None
        self._last_motion_seen: datetime | None = None
        self._max_motion_ratio: float = 0.0

    def update_settings(self, settings: SurveillanceSettings) -> None:
        self._settings = settings
        self.reset()

    def reset(self) -> None:
        self._state = "idle"
        self._background = None
        self._last_sample_ts = None
        self._motion_started = None
        self._recording_clip_id = None
        self._last_motion_seen = None
        self._max_motion_ratio = 0.0

    def process_frame(self, sample: FrameSample) -> None:
        if not self._settings.record_on_motion:
            return
        timestamp = sample.timestamp
        if not self._should_sample(timestamp):
            return
        analysis = self._analyse_frame(sample.frame)
        if analysis is None:
            return
        if analysis.motion_detected:
            self._max_motion_ratio = max(self._max_motion_ratio, analysis.changed_ratio)
            if self._motion_started is None:
                self._motion_started = timestamp
            self._last_motion_seen = timestamp
            if self._state != "recording" and self._motion_started is not None:
                if self._meets_min_duration(timestamp):
                    self._begin_recording(timestamp)
        else:
            self._motion_started = None

        if self._state == "recording":
            if analysis.motion_detected:
                self._extend_recording(timestamp)
            if self._should_stop(timestamp):
                self._finalise_clip(timestamp)

    # ------------------------------------------------------------------
    def _should_sample(self, timestamp: datetime) -> bool:
        interval = 1.0 / float(max(1, self._settings.detection_fps))
        if self._last_sample_ts is None:
            self._last_sample_ts = timestamp
            return True
        delta = (timestamp - self._last_sample_ts).total_seconds()
        if delta >= interval:
            self._last_sample_ts = timestamp
            return True
        return False

    def _analyse_frame(self, frame: np.ndarray) -> MotionSample | None:
        array = np.asarray(frame)
        if array.ndim == 3:
            array = np.mean(array[..., :3], axis=2)
        elif array.ndim != 2:
            logger.debug("Unsupported frame shape for motion detection: %s", array.shape)
            return None
        array = array.astype(np.float32)
        array = self._downscale(array)
        if array.size == 0:
            return None

        if self._background is None:
            self._background = array
            return MotionSample(changed_ratio=0.0, motion_detected=False)

        diff = np.abs(array - self._background)
        threshold = self._motion_threshold()
        changed = diff >= threshold
        changed_ratio = float(np.count_nonzero(changed)) / float(changed.size) * 100.0
        motion_detected = changed_ratio >= float(self._settings.min_changed_area_percent)
        if not motion_detected and self._state != "recording":
            alpha = 0.15
            self._background = (1.0 - alpha) * self._background + alpha * array
        return MotionSample(changed_ratio=changed_ratio, motion_detected=motion_detected)

    def _motion_threshold(self) -> float:
        sensitivity = float(self._settings.sensitivity)
        sensitivity = min(max(sensitivity, 0.0), 1.0)
        return max(4.0, (1.0 - sensitivity) * 40.0 + 6.0)

    def _downscale(self, array: np.ndarray) -> np.ndarray:
        target_height = 90
        target_width = 160
        height, width = array.shape
        step_y = max(1, height // target_height)
        step_x = max(1, width // target_width)
        result = array[::step_y, ::step_x]
        return result[:target_height, :target_width]

    def _meets_min_duration(self, timestamp: datetime) -> bool:
        if self._motion_started is None:
            return False
        elapsed = (timestamp - self._motion_started).total_seconds()
        return elapsed * 1000.0 >= float(self._settings.min_motion_duration_ms)

    def _begin_recording(self, timestamp: datetime) -> None:
        clip = self._runtime.begin_motion_record(timestamp)
        if clip is None:
            return
        self._recording_clip_id = clip.id
        self._state = "recording"
        self._last_motion_seen = timestamp
        self._max_motion_ratio = 0.0

    def _extend_recording(self, timestamp: datetime) -> None:
        self._last_motion_seen = timestamp

    def _should_stop(self, timestamp: datetime) -> bool:
        if self._recording_clip_id is None:
            return False
        if self._last_motion_seen is not None:
            gap = (timestamp - self._last_motion_seen).total_seconds()
            if gap >= float(self._settings.post_motion_gap_s):
                return True
        if self._motion_started is not None:
            duration = (timestamp - self._motion_started).total_seconds()
            if duration >= float(self._settings.clip_max_length_s):
                return True
        return False

    def _finalise_clip(self, timestamp: datetime) -> None:
        if self._recording_clip_id is None:
            return
        score = None
        if self._max_motion_ratio > 0.0:
            score = min(1.0, self._max_motion_ratio / 100.0)
        try:
            record = self._runtime.finish_motion_record(
                self._recording_clip_id,
                end_ts=timestamp,
                motion_score=score,
            )
            if record is not None:
                logger.info("Motion clip captured: id=%s", record.id)
        finally:
            self._state = "idle"
            self._motion_started = None
            self._recording_clip_id = None
            self._last_motion_seen = None
            self._max_motion_ratio = 0.0


class SurveillanceRuntime:
    """Coordinate surveillance-specific capture helpers."""

    def __init__(self, manager: SurveillanceManager) -> None:
        self._manager = manager
        settings = manager.load_settings()
        self._settings = settings
        buffer_seconds = float(settings.clip_max_length_s + settings.pre_roll_s + 5)
        self._buffer = FrameBuffer(capacity_seconds=buffer_seconds)
        self._recorder = MotionRecorder(self)
        self._lock = Lock()
        self._active = False
        self._recordings: Dict[int, ActiveRecording] = {}

    # ------------------------------------------------------------------
    def load_settings(self) -> SurveillanceSettings:
        return self._settings

    def notify_settings_updated(self, settings: SurveillanceSettings) -> None:
        self._settings = settings
        buffer_seconds = float(settings.clip_max_length_s + settings.pre_roll_s + 5)
        self._buffer.configure(buffer_seconds)
        self._recorder.update_settings(settings)

    def set_active(self, active: bool) -> None:
        with self._lock:
            if self._active == active:
                return
            self._active = active
        if not active:
            self._recorder.reset()
            self._flush_recordings(reason="mode switch")
            self._buffer.clear()

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    # ------------------------------------------------------------------
    def ingest_frame(self, frame: object, timestamp: datetime | None = None) -> object:
        if not self.is_active():
            return frame
        timestamp = timestamp or datetime.now(timezone.utc)
        array = ensure_rgb_frame(frame, even=False)
        array.setflags(write=False)
        sample = self._buffer.add(array, timestamp)
        recordings = self._active_recordings_snapshot()
        for recording in recordings:
            recording.add_sample(sample)
        self._recorder.process_frame(sample)
        self._finalise_completed(timestamp)
        return frame

    def start_manual_record(self, duration_s: Optional[float] = None) -> ClipRecord:
        if not self.is_active():
            raise ValueError("Surveillance mode must be active before recording")
        settings = self._settings
        duration = float(duration_s) if duration_s is not None else float(min(30, settings.clip_max_length_s))
        if duration <= 0:
            raise ValueError("Manual clip duration must be positive")
        if duration > settings.clip_max_length_s:
            duration = float(settings.clip_max_length_s)
        now = datetime.now(timezone.utc)
        pre_roll_samples = self._buffer.snapshot(settings.pre_roll_s, up_to=now)
        start_ts = pre_roll_samples[0].timestamp if pre_roll_samples else now
        end_ts = start_ts + timedelta(seconds=duration)
        clip = self._manager.create_clip_entry(
            start_ts=start_ts,
            end_ts=end_ts,
            has_audio=settings.audio_enabled,
            motion_score=None,
            settings=settings,
            reason="manual",
        )
        width, height = self._parse_resolution(settings.resolution)
        recording = ActiveRecording(
            clip=clip,
            manager=self._manager,
            settings=settings,
            fps=settings.framerate,
            encoding=settings.encoding,
            reason="manual",
            fallback_size=(width, height),
            target_end_ts=end_ts,
        )
        recording.add_samples(pre_roll_samples)
        with self._lock:
            self._recordings[clip.id] = recording
        logger.info("Manual recording started: id=%s", clip.id)
        return clip

    def create_test_clip(self) -> ClipRecord:
        clip = self._manager.create_test_clip()
        return clip

    # ------------------------------------------------------------------
    def begin_motion_record(self, trigger_ts: datetime) -> ClipRecord | None:
        if not self.is_active():
            return None
        settings = self._settings
        samples = self._buffer.snapshot(settings.pre_roll_s, up_to=trigger_ts)
        start_ts = samples[0].timestamp if samples else trigger_ts
        end_ts = start_ts + timedelta(seconds=settings.clip_max_length_s)
        clip = self._manager.create_clip_entry(
            start_ts=start_ts,
            end_ts=end_ts,
            has_audio=settings.audio_enabled,
            motion_score=None,
            settings=settings,
            reason="motion",
        )
        width, height = self._parse_resolution(settings.resolution)
        recording = ActiveRecording(
            clip=clip,
            manager=self._manager,
            settings=settings,
            fps=settings.framerate,
            encoding=settings.encoding,
            reason="motion",
            fallback_size=(width, height),
            target_end_ts=end_ts,
        )
        recording.add_samples(samples)
        with self._lock:
            self._recordings[clip.id] = recording
        return clip

    def finish_motion_record(
        self,
        clip_id: int,
        *,
        end_ts: datetime,
        motion_score: float | None,
    ) -> ClipRecord | None:
        return self._complete_recording(clip_id, end_ts=end_ts, motion_score=motion_score)

    # ------------------------------------------------------------------
    def _active_recordings_snapshot(self) -> List[ActiveRecording]:
        with self._lock:
            return list(self._recordings.values())

    def _finalise_completed(self, timestamp: datetime) -> None:
        completed: List[int] = []
        with self._lock:
            for clip_id, recording in self._recordings.items():
                if recording.reason == "manual" and recording.should_stop(timestamp):
                    completed.append(clip_id)
        for clip_id in completed:
            record = self._complete_recording(clip_id, end_ts=timestamp, motion_score=None)
            if record is not None:
                logger.info("Manual clip captured: id=%s", record.id)

    def _complete_recording(
        self,
        clip_id: int,
        *,
        end_ts: datetime,
        motion_score: float | None,
    ) -> ClipRecord | None:
        with self._lock:
            recording = self._recordings.pop(clip_id, None)
        if recording is None:
            return None
        try:
            return recording.finish(end_ts=end_ts, motion_score=motion_score)
        except Exception:
            logger.exception("Failed to finalise surveillance clip %s", clip_id)
            recording.cancel()
            return None

    def _flush_recordings(self, *, reason: str) -> None:
        with self._lock:
            recordings = list(self._recordings.values())
            self._recordings.clear()
        for recording in recordings:
            try:
                recording.finish(end_ts=datetime.now(timezone.utc))
            except Exception:
                logger.exception("Failed to finalise clip %s during %s", recording.clip_id, reason)
                recording.cancel()

    @staticmethod
    def _parse_resolution(resolution: str) -> tuple[int, int]:
        if "x" in resolution:
            try:
                width, height = [int(part) for part in resolution.split("x", 1)]
                return max(2, width), max(2, height)
            except ValueError:
                pass
        return (1280, 720)


__all__ = ["FrameSample", "MotionRecorder", "SurveillanceRuntime"]
