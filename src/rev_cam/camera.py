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


def _create_opencv_camera(config: str | None = None) -> BaseCamera:
    """Return an OpenCV camera instance for the supplied config string."""

    if not config or config == "opencv":
        return OpenCVCamera()

    _, _, suffix = config.partition(":")
    if not suffix:
        raise CameraError(f"Invalid OpenCV camera configuration: {config!r}")
    try:
        index = int(suffix)
    except ValueError as exc:
        raise CameraError(f"Invalid OpenCV camera index: {suffix!r}") from exc
    return OpenCVCamera(index=index)


def create_camera() -> BaseCamera:
    """Create the camera specified by the environment or via auto detection."""

    choice_raw = os.getenv("REVCAM_CAMERA")
    choice = choice_raw.lower().strip() if choice_raw else None

    if choice and choice not in {"auto", ""}:
        if choice == "synthetic":
            return SyntheticCamera()
        if choice.startswith("opencv"):
            return _create_opencv_camera(choice)
        if choice == "picamera":
            return Picamera2Camera()
        raise CameraError(f"Unknown camera type: {choice_raw}")

    for factory in (Picamera2Camera, OpenCVCamera):
        try:
            return factory()
        except CameraError:
            continue

    # Fall back to synthetic frames when running on development machines.
    return SyntheticCamera()


__all__ = [
    "BaseCamera",
    "CameraError",
    "Picamera2Camera",
    "OpenCVCamera",
    "SyntheticCamera",
    "create_camera",
]
