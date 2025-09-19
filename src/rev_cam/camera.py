"""Camera source abstractions."""
from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

# User visible identifiers for camera backends. The order controls how they are
# presented in the settings UI.
CAMERA_SOURCES: dict[str, str] = {
    "auto": "Automatic (PiCamera2 with synthetic fallback)",
    "picamera": "PiCamera2",  # Native Pi camera module
    "opencv": "OpenCV (USB webcam)",
    "synthetic": "Synthetic test pattern",
}

# Default camera selection when no explicit configuration is provided.
DEFAULT_CAMERA_CHOICE = "auto"

_CAMERA_ALIASES = {
    "picamera2": "picamera",
    "picamera2camera": "picamera",
}


class CameraError(RuntimeError):
    """Raised when the camera cannot be initialised."""


class BaseCamera(ABC):
    """Abstract camera capable of producing RGB frames."""

    @abstractmethod
    async def get_frame(self) -> np.ndarray:  # pragma: no cover - interface only
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - optional override
        return None


def _summarise_exception(exc: BaseException) -> str:
    """Collect the unique error messages from an exception chain."""

    details: list[str] = []
    seen: set[str] = set()
    to_consider: Iterable[BaseException | None] = (
        exc,
        getattr(exc, "__cause__", None),
        getattr(exc, "__context__", None),
    )
    for candidate in to_consider:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text not in seen:
            details.append(text)
            seen.add(text)
    return " | ".join(details)


class Picamera2Camera(BaseCamera):
    """Camera implementation using the Picamera2 stack."""

    def __init__(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # pragma: no cover - hardware dependent
            detail = _summarise_exception(exc)
            message = (
                "picamera2 is not available. Install the 'python3-picamera2' package "
                "and ensure the application can access system packages"
            )
            if detail:
                message = f"{message} ({detail})"
            raise CameraError(message) from exc

        self._camera = None
        started = False
        try:
            camera = Picamera2()
            self._camera = camera
            config = camera.create_video_configuration(main={"format": "RGB888"})
            camera.configure(config)
            camera.start()
            started = True
        except Exception as exc:  # pragma: no cover - hardware dependent
            camera = self._camera
            if camera is not None:
                try:
                    if started:
                        camera.stop()
                except Exception:
                    pass
                try:
                    camera.close()
                except Exception:
                    pass
            detail = _summarise_exception(exc)
            message = "Failed to initialise Picamera2 camera"
            hints: list[str] = []
            if detail:
                lower_detail = detail.lower()
                if "device or resource busy" in lower_detail:
                    hints.append(
                        "Another process is using the camera. Close libcamera-* applications or stop conflicting services (e.g. `sudo systemctl stop libcamera-apps`)."
                    )
                message = f"{message}: {detail}"
            if hints:
                message = f"{message}. {' '.join(hints)}"
            raise CameraError(message) from exc

    async def get_frame(self) -> np.ndarray:  # pragma: no cover - hardware dependent
        return await asyncio.to_thread(self._camera.capture_array)

    async def close(self) -> None:  # pragma: no cover - hardware dependent
        await asyncio.to_thread(self._camera.stop)
        await asyncio.to_thread(self._camera.close)


class OpenCVCamera(BaseCamera):
    """Fallback implementation using OpenCV VideoCapture."""

    def __init__(self, index: int = 0) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise CameraError("OpenCV is not installed") from exc

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise CameraError(f"Failed to open camera index {index}")

    async def get_frame(self) -> np.ndarray:
        ret, frame = await asyncio.to_thread(self._capture.read)
        if not ret:
            raise CameraError("Failed to read frame from OpenCV camera")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    async def close(self) -> None:
        await asyncio.to_thread(self._capture.release)


class SyntheticCamera(BaseCamera):
    """Generates synthetic frames for development and testing."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width = width
        self._height = height
        self._start = time.perf_counter()

    async def get_frame(self) -> np.ndarray:
        elapsed = time.perf_counter() - self._start
        horizontal = np.linspace(0, 255, self._width, dtype=np.uint8)
        vertical = np.linspace(0, 255, self._height, dtype=np.uint8).reshape(-1, 1)
        red = np.tile(horizontal, (self._height, 1))
        green = np.roll(red, int(elapsed * 10), axis=1)
        blue = np.tile(vertical, (1, self._width))
        frame = np.stack([red, green, blue], axis=2)
        return frame.astype(np.uint8)


def _normalise_choice(choice: str | None) -> str:
    if choice is None:
        choice = os.getenv("REVCAM_CAMERA", DEFAULT_CAMERA_CHOICE)
    normalised = choice.strip().lower()
    return _CAMERA_ALIASES.get(normalised, normalised)


def create_camera(choice: str | None = None) -> BaseCamera:
    """Create the camera specified by *choice* or the environment.

    When ``choice`` is ``"auto"`` the function attempts to construct a
    :class:`Picamera2Camera` and falls back to :class:`SyntheticCamera` if the
    dependency stack is unavailable. Explicit selections raise
    :class:`CameraError` on failure so callers can surface a helpful message to
    users.
    """

    resolved_choice = _normalise_choice(choice)
    if resolved_choice == "synthetic":
        return SyntheticCamera()
    if resolved_choice == "opencv":
        return OpenCVCamera()
    if resolved_choice == "picamera":
        return Picamera2Camera()
    if resolved_choice == "auto":
        try:
            return Picamera2Camera()
        except CameraError:
            return SyntheticCamera()
    raise CameraError(f"Unknown camera choice: {choice}")


def identify_camera(camera: BaseCamera) -> str:
    """Return the canonical identifier for a camera instance."""

    if isinstance(camera, Picamera2Camera):
        return "picamera"
    if isinstance(camera, OpenCVCamera):
        return "opencv"
    if isinstance(camera, SyntheticCamera):
        return "synthetic"
    return "unknown"


__all__ = [
    "CAMERA_SOURCES",
    "DEFAULT_CAMERA_CHOICE",
    "BaseCamera",
    "CameraError",
    "identify_camera",
    "Picamera2Camera",
    "OpenCVCamera",
    "SyntheticCamera",
    "create_camera",
]
