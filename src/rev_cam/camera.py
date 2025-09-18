"""Camera source abstractions."""
from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod

import numpy as np


class CameraError(RuntimeError):
    """Raised when the camera cannot be initialised."""


class BaseCamera(ABC):
    """Abstract camera capable of producing RGB frames."""

    @abstractmethod
    async def get_frame(self) -> np.ndarray:  # pragma: no cover - interface only
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - optional override
        return None


class Picamera2Camera(BaseCamera):
    """Camera implementation using the Picamera2 stack."""

    def __init__(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # pragma: no cover - hardware dependent
            raise CameraError("picamera2 is not available") from exc

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
            raise CameraError("Failed to initialise Picamera2 camera") from exc

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


def _normalise_choice(raw_choice: str | None) -> str:
    """Return the canonical camera choice string."""

    if raw_choice is None:
        return "auto"
    choice = raw_choice.strip().lower()
    return choice or "auto"


def create_camera() -> BaseCamera:
    """Create the camera specified by the environment."""

    raw_choice = os.getenv("REVCAM_CAMERA")
    choice = _normalise_choice(raw_choice)

    if choice == "synthetic":
        return SyntheticCamera()
    if choice == "opencv":
        return OpenCVCamera()
    if choice in {"picamera", "picamera2"}:
        return Picamera2Camera()
    if choice == "auto":
        try:
            return Picamera2Camera()
        except CameraError as exc:
            # Picamera2 import errors mean we are not running on Pi hardware or
            # the dependency is missing. Fall back to OpenCV if available before
            # using the synthetic test pattern so that USB cameras work out of
            # the box on development machines.
            if not isinstance(exc.__cause__, ImportError):
                raise
        try:
            return OpenCVCamera()
        except CameraError:
            return SyntheticCamera()
    raise CameraError(f"Unknown camera backend: {raw_choice!r}")


__all__ = [
    "BaseCamera",
    "CameraError",
    "Picamera2Camera",
    "OpenCVCamera",
    "SyntheticCamera",
    "create_camera",
]
