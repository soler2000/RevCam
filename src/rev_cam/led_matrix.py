"""Helpers for driving the RevCam LED ring.

The production hardware exposes a 16 pixel WS2812-compatible ring on GPIO18.
The :class:`LedRing` helper encapsulates the small amount of animation logic
used by the application to surface boot/ready/error states.  The class is
designed so the FastAPI application can drive the ring asynchronously without
having to worry about the underlying hardware details.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
import math
from typing import Callable, Iterable, Iterator, Protocol, Sequence


NEOPIXEL_PERMISSION_MESSAGE = (
    "NeoPixel driver requires root privileges (sudo) to access /dev/mem."
)

Color = tuple[int, int, int]


@dataclass(slots=True, frozen=True)
class LedRingStatus:
    """Snapshot of the LED ring configuration exposed to the API layer."""

    patterns: tuple[str, ...]
    pattern: str
    active_pattern: str
    error: bool
    available: bool
    message: str | None = None
    illumination_color: Color = field(default_factory=lambda: (255, 255, 255))
    illumination_intensity: float = 1.0


@dataclass(slots=True, frozen=True)
class PatternStep:
    """Single animation step for the LED ring.

    Parameters
    ----------
    colors:
        Sequence of RGB values (0-255) representing the desired state for each
        pixel in the ring.
    duration:
        Amount of time, in seconds, the frame should remain active before the
        next step is applied.  A ``duration`` of ``0`` indicates the next step
        should be applied immediately.
    """

    colors: tuple[Color, ...]
    duration: float = 0.1


class _RingDriver(Protocol):
    """Protocol describing the minimal interface required from a driver."""

    pixel_count: int

    def apply(self, colors: Sequence[Color]) -> None:  # pragma: no cover - protocol
        """Apply a new colour frame to the hardware."""

    def close(self) -> None:  # pragma: no cover - protocol
        """Release any hardware resources."""


def _is_permission_error(exc: BaseException) -> bool:
    """Return ``True`` when the exception looks like a privilege failure."""

    message = str(exc).lower()
    return (
        isinstance(exc, PermissionError)
        or "permission" in message
        or "sudo" in message
        or "/dev/mem" in message
    )


class _NeoPixelDriver:
    """Driver that talks to a WS2812/NeoPixel ring via ``adafruit-circuitpython``."""

    def __init__(
        self,
        *,
        pixel_count: int,
        brightness: float,
        logger: logging.Logger,
    ) -> None:
        self._logger = logger
        self.pixel_count = pixel_count

        try:  # pragma: no cover - hardware specific import
            import board  # type: ignore
            import neopixel  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised on systems without hardware
            if _is_permission_error(exc):
                raise RuntimeError(NEOPIXEL_PERMISSION_MESSAGE) from None
            raise RuntimeError("NeoPixel libraries unavailable") from None

        try:
            pin = getattr(board, "D18")
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Board module does not expose GPIO18 pin") from exc

        try:
            self._pixels = neopixel.NeoPixel(
                pin,
                pixel_count,
                brightness=brightness,
                auto_write=False,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            if _is_permission_error(exc):
                raise RuntimeError(NEOPIXEL_PERMISSION_MESSAGE) from None
            raise RuntimeError(f"Failed to initialise NeoPixel ring: {exc}") from exc

        try:
            # Ensure the hardware starts dark.
            self.apply(((0, 0, 0),) * pixel_count)
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            if _is_permission_error(exc):
                raise RuntimeError(NEOPIXEL_PERMISSION_MESSAGE) from None
            raise RuntimeError(f"Failed to initialise NeoPixel ring: {exc}") from exc

    def apply(self, colors: Sequence[Color]) -> None:  # pragma: no cover - hardware path
        try:
            for index in range(self.pixel_count):
                self._pixels[index] = colors[index]
            self._pixels.show()
        except Exception as exc:  # pragma: no cover - defensive logging
            message = str(exc)
            lower_message = message.lower()
            if _is_permission_error(exc):
                self._logger.warning(
                    "NeoPixel driver unavailable due to insufficient privileges: %s",
                    message,
                )
                raise RuntimeError(NEOPIXEL_PERMISSION_MESSAGE) from None
            self._logger.exception("Failed to update LED ring state")
            raise

    def close(self) -> None:  # pragma: no cover - hardware path
        try:
            self.apply(((0, 0, 0),) * self.pixel_count)
        finally:
            try:
                self._pixels.deinit()
            except AttributeError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                self._logger.debug("Failed to de-initialise NeoPixel ring", exc_info=True)


class LedRing:
    """Asynchronous animation helper for the RevCam LED ring."""

    DEFAULT_BRIGHTNESS = 0.2
    DEFAULT_PIXEL_COUNT = 16
    DEFAULT_STEP_DURATION = 0.1

    def __init__(
        self,
        *,
        pixel_count: int = DEFAULT_PIXEL_COUNT,
        brightness: float = DEFAULT_BRIGHTNESS,
        driver: _RingDriver | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._closed = False
        self._base_pattern = "off"
        self._error_active = False
        self._active_pattern = "off"
        self._pattern_task: asyncio.Task[None] | None = None
        self._driver_message: str | None = None
        self._illumination_color: Color = (255, 255, 255)
        self._illumination_intensity: float = 1.0

        if driver is not None:
            self._driver: _RingDriver | None = driver
            driver_pixels = getattr(driver, "pixel_count", pixel_count)
            if isinstance(driver_pixels, int) and driver_pixels > 0:
                self._pixel_count = driver_pixels
            else:  # pragma: no cover - defensive fallback
                self._pixel_count = pixel_count
            self._driver_message = None
        else:
            self._pixel_count = pixel_count
            try:
                self._driver = _NeoPixelDriver(
                    pixel_count=pixel_count,
                    brightness=max(0.0, min(1.0, brightness)),
                    logger=self._logger,
                )
                self._driver_message = None
            except RuntimeError as exc:
                reason = str(exc)
                self._driver_message = reason or "LED ring unavailable"
                self._logger.info("LED ring disabled: %s", reason)
                self._driver = None

        self._patterns: dict[str, Callable[[], Iterator[PatternStep]]] = {
            "off": self._pattern_off,
            "boot": self._pattern_boot,
            "ready": self._pattern_ready,
            "error": self._pattern_error,
            "surveillance": self._pattern_surveillance,
            "recording": self._pattern_recording,
            "hotspot": self._pattern_hotspot,
            "illumination": self._pattern_illumination,
        }

    async def set_pattern(self, name: str) -> None:
        """Activate the requested base pattern.

        If an error condition is active the pattern is stored but will not take
        effect until :meth:`set_error` clears the fault state.
        """

        async with self._lock:
            if self._closed:
                return
            if name not in self._patterns:
                raise ValueError(f"Unknown LED ring pattern {name!r}")
            self._base_pattern = name
            await self._apply_state_locked()

    async def set_error(self, active: bool) -> None:
        """Enable or disable the error animation overlay."""

        async with self._lock:
            if self._closed:
                return
            active = bool(active)
            if active == self._error_active:
                return
            self._error_active = active
            await self._apply_state_locked()

    async def set_illumination(
        self,
        *,
        color: Color | None = None,
        intensity: float | None = None,
    ) -> None:
        """Update the illumination colour or intensity."""

        async with self._lock:
            if self._closed:
                return

            changed = False

            if color is not None:
                if not isinstance(color, Sequence) or len(color) != 3:
                    raise ValueError("Illumination colour must include red, green, and blue values")
                parsed: list[int] = []
                for component in color:
                    if not isinstance(component, (int, float)):
                        raise ValueError("Illumination colour components must be numbers")
                    value = int(round(component))
                    if not 0 <= value <= 255:
                        raise ValueError("Illumination colour components must be between 0 and 255")
                    parsed.append(value)
                rgb = (parsed[0], parsed[1], parsed[2])
                if rgb != self._illumination_color:
                    self._illumination_color = rgb
                    changed = True

            if intensity is not None:
                try:
                    intensity_value = float(intensity)
                except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
                    raise ValueError("Illumination intensity must be a number") from exc
                if not math.isfinite(intensity_value):
                    raise ValueError("Illumination intensity must be finite")
                if not 0.0 <= intensity_value <= 1.0:
                    raise ValueError("Illumination intensity must be between 0 and 1")
                if abs(intensity_value - self._illumination_intensity) > 1e-6:
                    self._illumination_intensity = intensity_value
                    changed = True

            if changed and self._base_pattern == "illumination" and not self._error_active:
                await self._apply_state_locked()

    async def get_status(self) -> LedRingStatus:
        """Return the currently configured state of the LED ring."""

        async with self._lock:
            patterns = tuple(self._patterns.keys())
            pattern = self._base_pattern
            active = self._active_pattern
            error_active = self._error_active
            available = self._driver is not None and not self._closed
            message = self._driver_message if not available else None
            illumination_color = self._illumination_color
            illumination_intensity = self._illumination_intensity
        return LedRingStatus(
            patterns=patterns,
            pattern=pattern,
            active_pattern=active,
            error=error_active,
            available=available,
            message=message,
            illumination_color=illumination_color,
            illumination_intensity=illumination_intensity,
        )

    async def aclose(self) -> None:
        """Cancel any animations and release the underlying driver."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            await self._cancel_pattern_locked()
            driver = self._driver
            self._driver = None
            self._driver_message = None
        if driver is not None:
            try:
                driver.apply(((0, 0, 0),) * getattr(driver, "pixel_count", self._pixel_count))
            except Exception:  # pragma: no cover - defensive logging
                self._logger.debug("Failed to blank LED ring during shutdown", exc_info=True)
            try:
                driver.close()
            except Exception:  # pragma: no cover - defensive logging
                self._logger.debug("Failed to close LED ring driver", exc_info=True)

    async def _apply_state_locked(self) -> None:
        target = "error" if self._error_active else self._base_pattern
        if target not in self._patterns:  # pragma: no cover - defensive guard
            self._logger.warning("Unknown LED ring pattern requested: %s", target)
            return
        if target == self._active_pattern and self._pattern_task is not None:
            return
        await self._start_pattern_locked(target)

    async def _start_pattern_locked(self, name: str) -> None:
        pattern_factory = self._patterns[name]
        await self._cancel_pattern_locked()
        ready = asyncio.Event()
        task = asyncio.create_task(self._run_pattern(name, pattern_factory(), ready))
        self._pattern_task = task
        await ready.wait()

    async def _cancel_pattern_locked(self) -> None:
        task = self._pattern_task
        if task is None:
            return
        self._pattern_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive logging
            self._logger.debug("LED ring pattern task raised during cancellation", exc_info=True)

    async def _run_pattern(
        self,
        name: str,
        steps: Iterable[PatternStep],
        ready: asyncio.Event,
    ) -> None:
        self._active_pattern = name
        first_step = True
        try:
            for step in steps:
                self._apply_colors(step.colors)
                if first_step:
                    ready.set()
                    first_step = False
                duration = step.duration if step.duration is not None else self.DEFAULT_STEP_DURATION
                if duration > 0:
                    await asyncio.sleep(duration)
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise
        except Exception:  # pragma: no cover - defensive logging
            self._logger.exception("LED ring pattern %s failed", name)
        finally:
            if first_step:
                ready.set()
            if self._pattern_task is asyncio.current_task():
                self._pattern_task = None
            if name != "off" and not self._closed and self._active_pattern == name:
                # When a looping generator exhausts unexpectedly we revert to "off".
                self._active_pattern = "off"

    def _apply_colors(self, colors: Sequence[Color]) -> None:
        if self._driver is None:
            return
        if len(colors) != self._pixel_count:
            if len(colors) < self._pixel_count:
                padded = list(colors) + [(0, 0, 0)] * (self._pixel_count - len(colors))
                colors = tuple(padded)
            else:
                colors = tuple(colors[: self._pixel_count])
        try:
            self._driver.apply(colors)
        except Exception as exc:  # pragma: no cover - defensive logging
            reason = str(exc) or "LED ring driver failure"
            self._driver_message = reason
            self._logger.info("LED ring disabled: %s", reason)
            try:
                self._driver.close()
            except Exception:  # pragma: no cover - defensive logging
                self._logger.debug("Failed to close LED ring driver after error", exc_info=True)
            self._driver = None

    def _pattern_off(self) -> Iterator[PatternStep]:
        yield PatternStep(colors=((0, 0, 0),) * self._pixel_count, duration=0)

    def _pattern_boot(self) -> Iterator[PatternStep]:
        index = 0
        highlight: Color = (0, 32, 96)
        off: Color = (0, 0, 0)
        while True:
            frame = [off] * self._pixel_count
            frame[index] = highlight
            index = (index + 1) % self._pixel_count
            yield PatternStep(colors=tuple(frame), duration=0.08)

    def _pattern_ready(self) -> Iterator[PatternStep]:
        brightness = 16
        direction = 1
        while True:
            value = max(0, min(255, brightness))
            color: Color = (0, value, 0)
            yield PatternStep(colors=(color,) * self._pixel_count, duration=0.12)
            brightness += direction * 16
            if brightness >= 160 or brightness <= 32:
                direction *= -1

    def _pattern_error(self) -> Iterator[PatternStep]:
        intense: Color = (96, 0, 0)
        dim: Color = (8, 0, 0)
        while True:
            yield PatternStep(colors=(intense,) * self._pixel_count, duration=0.3)
            yield PatternStep(colors=(dim,) * self._pixel_count, duration=0.12)

    def _pattern_surveillance(self) -> Iterator[PatternStep]:
        brightness = 32
        direction = 1
        while True:
            value = max(0, min(200, brightness))
            color: Color = (value, value // 2, 0)
            yield PatternStep(colors=(color,) * self._pixel_count, duration=0.12)
            brightness += direction * 12
            if brightness >= 160 or brightness <= 32:
                direction *= -1

    def _pattern_recording(self) -> Iterator[PatternStep]:
        index = 0
        tail = max(1, self._pixel_count // 4)
        base: Color = (16, 0, 0)
        while True:
            frame: list[Color] = [base] * self._pixel_count
            for offset in range(tail):
                position = (index - offset) % self._pixel_count
                scale = (tail - offset) / tail
                intensity = max(32, min(255, int(round(200 * scale))))
                frame[position] = (intensity, 0, 0)
            index = (index + 1) % self._pixel_count
            yield PatternStep(colors=tuple(frame), duration=0.06)

    def _pattern_hotspot(self) -> Iterator[PatternStep]:
        step = 0
        highlight: Color = (0, 0, 128)
        ambient: Color = (0, 0, 24)
        while True:
            frame: list[Color] = []
            for index in range(self._pixel_count):
                if (index + step) % 4 == 0:
                    frame.append(highlight)
                else:
                    frame.append(ambient)
            step = (step + 1) % 4
            yield PatternStep(colors=tuple(frame), duration=0.1)

    def _pattern_illumination(self) -> Iterator[PatternStep]:
        while True:
            color = self._scaled_illumination_color()
            yield PatternStep(colors=(color,) * self._pixel_count, duration=0.12)

    def _scaled_illumination_color(self) -> Color:
        base = self._illumination_color
        intensity = self._illumination_intensity
        return (
            max(0, min(255, int(round(base[0] * intensity)))),
            max(0, min(255, int(round(base[1] * intensity)))),
            max(0, min(255, int(round(base[2] * intensity)))),
        )


__all__ = ["LedRing", "LedRingStatus", "NEOPIXEL_PERMISSION_MESSAGE"]
