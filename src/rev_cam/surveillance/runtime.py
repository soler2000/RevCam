"""Runtime helpers coordinating manual and motion-triggered recording."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Optional

import logging

try:  # pragma: no cover - optional dependency guard
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency guard
    np = None  # type: ignore[assignment]

from .manager import ClipRecord, SurveillanceManager
from .settings import SurveillanceSettings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MotionSample:
    """Lightweight container summarising motion analysis for a frame."""

    changed_ratio: float
    motion_detected: bool


class MotionRecorder:
    """Detect motion from incoming frames and emit placeholder clips."""

    def __init__(self, manager: SurveillanceManager) -> None:
        self._manager = manager
        self._settings = manager.load_settings()
        self._state: str = "idle"
        self._background: Optional["np.ndarray"] = None
        self._previous_frame: Optional["np.ndarray"] = None
        self._last_sample_ts: Optional[datetime] = None
        self._motion_started: Optional[datetime] = None
        self._recording_started: Optional[datetime] = None
        self._last_motion_seen: Optional[datetime] = None
        self._max_motion_ratio: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_settings(self, settings: SurveillanceSettings) -> None:
        self._settings = settings
        self.reset()

    def reset(self) -> None:
        self._state = "idle"
        self._background = None
        self._previous_frame = None
        self._last_sample_ts = None
        self._motion_started = None
        self._recording_started = None
        self._last_motion_seen = None
        self._max_motion_ratio = 0.0

    def process_frame(
        self,
        frame: object,
        *,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ClipRecord]:
        if np is None:
            return None

        if not self._settings.record_on_motion:
            return None

        timestamp = timestamp or datetime.now(timezone.utc)
        if not self._should_sample(timestamp):
            return None

        sample = self._analyse_frame(frame)
        if sample is None:
            return None

        if sample.motion_detected:
            self._max_motion_ratio = max(self._max_motion_ratio, sample.changed_ratio)
            if self._motion_started is None:
                self._motion_started = timestamp
            self._last_motion_seen = timestamp
            if self._state != "recording" and self._motion_started is not None:
                if self._meets_min_duration(timestamp):
                    self._begin_recording(timestamp)
        else:
            self._motion_started = None

        if self._state == "recording":
            if sample.motion_detected:
                self._extend_recording(timestamp)
            if self._should_stop(timestamp):
                return self._finalise_clip(timestamp)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
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

    def _analyse_frame(self, frame: object) -> Optional[MotionSample]:
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
            self._previous_frame = array
            return MotionSample(changed_ratio=0.0, motion_detected=False)

        background = self._background
        diff = np.abs(array - background)
        threshold = self._motion_threshold()
        changed = diff >= threshold
        changed_ratio = float(np.count_nonzero(changed)) / float(changed.size) * 100.0
        motion_detected = changed_ratio >= float(self._settings.min_changed_area_percent)
        if not motion_detected and self._state != "recording":
            alpha = 0.15
            self._background = (1.0 - alpha) * background + alpha * array
        self._previous_frame = array
        return MotionSample(changed_ratio=changed_ratio, motion_detected=motion_detected)

    def _motion_threshold(self) -> float:
        sensitivity = float(self._settings.sensitivity)
        sensitivity = min(max(sensitivity, 0.0), 1.0)
        # Map sensitivity so that higher sensitivity lowers the threshold.
        return max(4.0, (1.0 - sensitivity) * 40.0 + 6.0)

    def _downscale(self, array: "np.ndarray") -> "np.ndarray":
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
        pre_roll = max(0.0, float(self._settings.pre_roll_s))
        start_ts = timestamp - timedelta(seconds=pre_roll)
        self._recording_started = start_ts
        self._last_motion_seen = timestamp
        self._state = "recording"
        self._max_motion_ratio = 0.0

    def _extend_recording(self, timestamp: datetime) -> None:
        self._last_motion_seen = timestamp

    def _should_stop(self, timestamp: datetime) -> bool:
        if self._recording_started is None:
            return False
        if self._last_motion_seen is not None:
            gap = (timestamp - self._last_motion_seen).total_seconds()
            if gap >= float(self._settings.post_motion_gap_s):
                return True
        duration = (timestamp - self._recording_started).total_seconds()
        if duration >= float(self._settings.clip_max_length_s):
            return True
        return False

    def _finalise_clip(self, timestamp: datetime) -> Optional[ClipRecord]:
        if self._recording_started is None:
            return None
        start_ts = self._recording_started
        end_ts = timestamp
        if end_ts <= start_ts:
            end_ts = start_ts + timedelta(seconds=1)
        score = None
        if self._max_motion_ratio > 0.0:
            score = min(1.0, self._max_motion_ratio / 100.0)
        try:
            record = self._manager.create_motion_clip(
                start_ts=start_ts,
                end_ts=end_ts,
                motion_score=score,
            )
        finally:
            self.reset()
        return record


class SurveillanceRuntime:
    """Coordinate surveillance-specific capture helpers."""

    def __init__(self, manager: SurveillanceManager) -> None:
        self._manager = manager
        self._recorder = MotionRecorder(manager)
        self._lock = Lock()
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def set_active(self, active: bool) -> None:
        with self._lock:
            if self._active == active:
                return
            self._active = active
            if not active:
                self._recorder.reset()

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def notify_settings_updated(self, settings: SurveillanceSettings) -> None:
        self._recorder.update_settings(settings)

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------
    def ingest_frame(self, frame: object) -> object:
        active = self.is_active()
        if not active:
            return frame
        timestamp = datetime.now(timezone.utc)
        record = self._recorder.process_frame(frame, timestamp=timestamp)
        if record is not None:
            logger.info("Motion clip captured: id=%s", record.id)
        return frame

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------
    def start_manual_record(self, duration_s: Optional[float] = None) -> ClipRecord:
        with self._lock:
            clip = self._manager.create_manual_clip(duration_s)
        logger.info("Manual clip captured: id=%s", clip.id)
        return clip

    def create_test_clip(self) -> ClipRecord:
        clip = self._manager.create_test_clip()
        logger.info("Test clip generated: id=%s", clip.id)
        return clip


__all__ = ["MotionRecorder", "SurveillanceRuntime"]
