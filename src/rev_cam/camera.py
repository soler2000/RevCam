"""Camera source abstractions."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

NUMPY_ABI_HINT = (
    "Detected a NumPy ABI mismatch. Reinstall the Raspberry Pi OS packages so "
    "NumPy and Picamera2 share compatible binaries (for example `sudo apt install "
    "--reinstall python3-numpy python3-picamera2 python3-simplejpeg`—note the "
    "`python3-` prefix on SimpleJPEG; install it via `pip install --prefer-binary "
    "simplejpeg` if APT cannot find the package). If NumPy was upgraded inside a "
    "virtual environment, recreate it with `python3 -m venv --system-site-packages "
    ".venv`."
)

LEGACY_SDN_HINT = (
    "The camera tuning file still uses the legacy SDN layout. Move the `sdn` "
    "block into the `rpi.denoise` section of the tuning file (or update to the "
    "latest Raspberry Pi OS camera stack) so the warning disappears."
)

_NUMPY_ABI_TOKENS = (
    "abi",
    "dtype size changed",
    "pyarray",
    "multiarray",
    "umath",
)

_LEGACY_SDN_TOKENS = (
    "legacy sdn tuning",
    "legacy sdn",
    "warn rpisdn",
)

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


def summarise_exception(exc: BaseException) -> str:
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


def detect_numpy_abi_mismatch(detail: str | None) -> bool:
    """Return ``True`` when *detail* looks like a NumPy ABI mismatch."""

    if not detail:
        return False

    lower_detail = detail.lower()
    if "numpy" not in lower_detail and not any(
        token in lower_detail for token in ("pyarray", "multiarray", "umath")
    ):
        return False

    return any(token in lower_detail for token in _NUMPY_ABI_TOKENS)


def detect_legacy_sdn_warning(detail: str | None) -> bool:
    """Return ``True`` when *detail* reports legacy SDN tuning."""

    if not detail:
        return False

    lower_detail = detail.lower()
    if "sdn" not in lower_detail:
        return False

    if "rpi.denoise" in lower_detail:
        return True

    return any(token in lower_detail for token in _LEGACY_SDN_TOKENS)


class _NullSync:
    """Context manager used by :class:`_NullAllocator`."""

    def __enter__(self) -> "_NullSync":  # pragma: no cover - defensive shim
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:  # pragma: no cover - defensive shim
        return False


class _NullAllocator:
    """Fallback allocator used when Picamera2 initialisation fails early."""

    def sync(self, *args: object, **kwargs: object) -> _NullSync:  # pragma: no cover - defensive shim
        return _NullSync()

    def acquire(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - defensive shim
        return None

    def release(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - defensive shim
        return None

    def __getattr__(self, name: str):  # pragma: no cover - defensive shim
        def _noop(*args: object, **kwargs: object) -> None:
            logger.debug("_NullAllocator ignoring %s call", name)
            return None

        return _noop


_NULL_ALLOCATOR = _NullAllocator()


def _ensure_picamera_allocator(camera: object) -> None:
    """Ensure *camera* exposes an allocator with a ``sync`` method."""

    if camera is None:  # pragma: no cover - defensive guard
        return

    def allocator_ready() -> bool:
        candidate = getattr(camera, "allocator", None)
        return hasattr(candidate, "sync")

    if allocator_ready():
        return

    def set_instance_attr() -> None:
        setattr(camera, "allocator", _NULL_ALLOCATOR)

    def set_object_attr() -> None:
        object.__setattr__(camera, "allocator", _NULL_ALLOCATOR)

    def set_class_attr() -> None:
        setattr(camera.__class__, "allocator", _NULL_ALLOCATOR)

    setters = (
        ("instance setattr", set_instance_attr),
        ("object setattr", set_object_attr),
        ("class setattr", set_class_attr),
    )

    for description, setter in setters:
        try:
            setter()
        except Exception:  # pragma: no cover - defensive logging only
            logger.debug(
                "Failed to install Picamera2 allocator via %s", description, exc_info=True
            )
            continue
        if allocator_ready():
            logger.debug(
                "Installed Picamera2 allocator shim via %s on object %r",
                description,
                camera,
            )
            return

    if not allocator_ready():  # pragma: no cover - defensive logging only
        logger.warning("Unable to install Picamera2 allocator shim on object %r", camera)


def _is_service_active(name: str) -> bool | None:
    """Return whether *name* is an active systemd service."""

    try:
        result = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # pragma: no cover - system without systemd
        logger.debug("systemctl not available when checking %s", name)
        return None
    except Exception:  # pragma: no cover - defensive logging only
        logger.debug("Failed to query systemctl status for %s", name, exc_info=True)
        return None

    if result.returncode == 0:
        return True

    output = (result.stdout or result.stderr or "").strip().lower()
    if result.returncode == 3 or output == "inactive":
        return False

    return None


def _list_camera_processes() -> list[str]:
    """Return a list of processes that look like they may own the camera."""

    keywords = ("libcamera", "picamera", "rpicam", "v4l2", "mmal")
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,cmd"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # pragma: no cover - defensive logging only
        logger.debug("Failed to enumerate processes for camera diagnostics", exc_info=True)
        return []

    matches: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        lower_command = command.lower()
        if any(keyword in lower_command for keyword in keywords):
            matches.append(f"{pid} ({command.strip()})")

    return matches


def _collect_camera_conflicts() -> list[str]:
    """Return human-readable hints about known camera conflicts."""

    hints: list[str] = []
    service_active = _is_service_active("libcamera-apps")
    if service_active:
        hints.append(
            "System service 'libcamera-apps' is running. Stop it with `sudo systemctl stop libcamera-apps` to free the camera."
        )
    processes = _list_camera_processes()
    if processes:
        formatted = ", ".join(processes)
        hints.append(
            f"Processes currently using the camera: {formatted}. Stop these processes to free the device."
        )
        lower_processes = [process.lower() for process in processes]
        if any("kworker" in text and "mmal" in text for text in lower_processes):
            hints.append(
                "Kernel threads named kworker/R-mmal-vchiq indicate the legacy camera interface is still enabled. Disable the legacy camera (e.g. via `sudo raspi-config` -> Interface Options -> Legacy Camera, or remove `start_x=1` from `/boot/config.txt`) and reboot to free the device."
            )
    return hints


def diagnose_camera_conflicts() -> list[str]:
    """Public helper used by diagnostics tooling to surface camera conflicts."""

    return _collect_camera_conflicts()


class Picamera2Camera(BaseCamera):
    """Camera implementation using the Picamera2 stack."""

    def __init__(
        self,
        resolution: tuple[int, int] | None = None,
        *,
        fps: int | None = None,
    ) -> None:
        try:
            from picamera2 import Picamera2
        except Exception as exc:  # pragma: no cover - hardware dependent
            detail = summarise_exception(exc)
            message = (
                "picamera2 is not available. Install the 'python3-picamera2' package "
                "and ensure the application can access system packages (for example "
                "recreate the virtual environment with `python3 -m venv --system-"
                "site-packages .venv` or run `./scripts/install.sh --pi`)."
            )
            hints: list[str] = []
            if detail:
                if detect_numpy_abi_mismatch(detail) and NUMPY_ABI_HINT not in hints:
                    hints.append(NUMPY_ABI_HINT)
                message = f"{message} ({detail})"
            if hints:
                message = f"{message} {' '.join(hints)}"
            logger.exception("Failed to import Picamera2: %s", detail or exc)
            raise CameraError(message) from exc

        _ensure_picamera_allocator(Picamera2)

        self._camera = None
        self._frame_duration_limits: tuple[int, int] | None = None
        target_fps = self._normalise_fps(fps)
        if target_fps is not None:
            frame_period = max(1, int(round(1_000_000 / float(target_fps))))
            self._frame_duration_limits = (frame_period, frame_period)
        camera_instance = None
        started = False
        try:
            camera_instance = Picamera2()
            self._camera = camera_instance
            _ensure_picamera_allocator(camera_instance)
            main_config: dict[str, object] = {"format": "RGB888"}
            if resolution is not None:
                width, height = resolution
                main_config["size"] = (int(width), int(height))
            config = camera_instance.create_video_configuration(
                main=main_config,
                buffer_count=1,
            )
            self._apply_frame_duration_hint(config)
            camera_instance.configure(config)
            camera_instance.start()
            started = True
            self._initialise_controls()
        except Exception as exc:  # pragma: no cover - hardware dependent
            camera = camera_instance or self._camera
            if camera is not None:
                try:
                    _ensure_picamera_allocator(camera)
                except Exception:
                    pass
                try:
                    if started:
                        camera.stop()
                except Exception:
                    pass
                try:
                    _ensure_picamera_allocator(camera)
                except Exception:
                    pass
                try:
                    camera.close()
                except Exception:
                    pass
            else:
                try:
                    _ensure_picamera_allocator(Picamera2)
                except Exception:
                    pass
            detail = summarise_exception(exc)
            message = "Failed to initialise Picamera2 camera"
            hints: list[str] = []
            if detail:
                if detect_numpy_abi_mismatch(detail) and NUMPY_ABI_HINT not in hints:
                    hints.append(NUMPY_ABI_HINT)
                lower_detail = detail.lower()
                if "device or resource busy" in lower_detail:
                    hints.append(
                        "Another process is using the camera. Close libcamera-* applications or stop conflicting services (e.g. `sudo systemctl stop libcamera-apps`)."
                    )
                    hints.extend(_collect_camera_conflicts())
                if detect_legacy_sdn_warning(detail) and (
                    LEGACY_SDN_HINT not in hints
                ):
                    hints.append(LEGACY_SDN_HINT)
                message = f"{message}: {detail}"
            if hints:
                message = f"{message}. {' '.join(hints)}"
            logger.exception(
                "Picamera2 initialisation failed with detail '%s' and hints %s",
                detail or exc,
                hints or None,
            )
            raise CameraError(message) from exc

    async def get_frame(self) -> np.ndarray:  # pragma: no cover - hardware dependent
        frame = await asyncio.to_thread(self._camera.capture_array)
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[2] >= 3:
            converted = np.ascontiguousarray(array[..., :3][:, :, ::-1])
            if converted.dtype != np.uint8:
                converted = np.clip(converted, 0, 255).astype(np.uint8, copy=False)
            return converted
        return array

    async def close(self) -> None:  # pragma: no cover - hardware dependent
        await asyncio.to_thread(self._camera.stop)
        await asyncio.to_thread(self._camera.close)

    def _normalise_fps(self, fps: int | None) -> int | None:
        if fps is None:
            env_value = os.getenv("REVCAM_CAMERA_FPS")
            if env_value:
                try:
                    fps = int(round(float(env_value)))
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid REVCAM_CAMERA_FPS value %r; ignoring",
                        env_value,
                    )
                    fps = None
        if fps is None or fps <= 0:
            return None
        return int(min(max(fps, 1), 120))

    def _apply_frame_duration_hint(self, config: object) -> None:
        if self._frame_duration_limits is None:
            return

        limits = self._frame_duration_limits
        controls: dict[str, object] | None = None
        if isinstance(config, dict):
            controls = dict(config.get("controls") or {})
        else:
            existing = getattr(config, "controls", None)
            if isinstance(existing, dict):
                controls = dict(existing)
        if controls is None:
            controls = {}
        controls.setdefault("FrameDurationLimits", limits)
        if isinstance(config, dict):
            config["controls"] = controls
        else:
            try:
                setattr(config, "controls", controls)
            except Exception:
                logger.debug(
                    "Unable to attach frame duration hint to Picamera2 config", exc_info=True
                )

    def _initialise_controls(self) -> None:
        camera = self._camera
        if camera is None:
            return

        if self._frame_duration_limits is not None and hasattr(camera, "set_controls"):
            try:
                camera.set_controls({"FrameDurationLimits": self._frame_duration_limits})
            except Exception:
                logger.debug("Unable to apply frame duration limits", exc_info=True)

        metadata: dict[str, object] | None = None
        if hasattr(camera, "capture_metadata"):
            try:
                time.sleep(0.2)
                captured = camera.capture_metadata()
            except Exception:
                logger.debug("Picamera2 metadata capture failed", exc_info=True)
            else:
                if isinstance(captured, dict):
                    metadata = captured

        controls: dict[str, object] = {"AeEnable": False}
        if metadata is not None:
            exposure = metadata.get("ExposureTime")
            gain = metadata.get("AnalogueGain")
            if isinstance(exposure, (int, float)) and exposure > 0:
                controls["ExposureTime"] = int(exposure)
            if isinstance(gain, (int, float)) and gain > 0:
                controls["AnalogueGain"] = float(gain)
        controls.setdefault("AwbEnable", False)

        if hasattr(camera, "set_controls"):
            try:
                camera.set_controls(controls)
            except Exception:
                logger.debug("Unable to lock Picamera2 automatic controls", exc_info=True)


class OpenCVCamera(BaseCamera):
    """Fallback implementation using OpenCV VideoCapture."""

    def __init__(
        self,
        index: int = 0,
        resolution: tuple[int, int] | None = None,
        *,
        fps: int | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise CameraError("OpenCV is not installed") from exc

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise CameraError(f"Failed to open camera index {index}")
        if resolution is not None:
            width, height = resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None and fps > 0:
            self._capture.set(cv2.CAP_PROP_FPS, float(fps))

    async def get_frame(self) -> np.ndarray:
        ret, frame = await asyncio.to_thread(self._capture.read)
        if not ret:
            raise CameraError("Failed to read frame from OpenCV camera")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    async def close(self) -> None:
        await asyncio.to_thread(self._capture.release)


class SyntheticCamera(BaseCamera):
    """Generates synthetic frames for development and testing."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        *,
        resolution: tuple[int, int] | None = None,
        fps: int | None = None,
    ) -> None:
        if resolution is not None:
            width, height = resolution
        self._width = int(width)
        self._height = int(height)
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


def create_camera(
    choice: str | None = None,
    *,
    resolution: tuple[int, int] | None = None,
    fps: int | None = None,
) -> BaseCamera:
    """Create the camera specified by *choice* or the environment.

    When ``choice`` is ``"auto"`` the function attempts to construct a
    :class:`Picamera2Camera` and falls back to :class:`SyntheticCamera` if the
    dependency stack is unavailable. Explicit selections raise
    :class:`CameraError` on failure so callers can surface a helpful message to
    users.
    """

    resolved_choice = _normalise_choice(choice)
    if resolved_choice == "synthetic":
        return SyntheticCamera(resolution=resolution)
    if resolved_choice == "opencv":
        return OpenCVCamera(resolution=resolution, fps=fps)
    if resolved_choice == "picamera":
        return Picamera2Camera(resolution=resolution, fps=fps)
    if resolved_choice == "auto":
        try:
            return Picamera2Camera(resolution=resolution, fps=fps)
        except CameraError as exc:
            logger.error("Picamera2 unavailable during auto selection: %s", exc)
            return SyntheticCamera(resolution=resolution)
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
    "diagnose_camera_conflicts",
    "Picamera2Camera",
    "OpenCVCamera",
    "SyntheticCamera",
    "create_camera",
]
