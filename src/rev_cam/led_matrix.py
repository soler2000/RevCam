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
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Protocol, Sequence

Color = tuple[int, int, int]


@dataclass(slots=True, frozen=True)
class LedRingStatus:
    """Snapshot of the LED ring configuration exposed to the API layer."""

    patterns: tuple[str, ...]
    pattern: str
    active_pattern: str
    error: bool
    available: bool


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
        except Exception:  # pragma: no cover - exercised on systems without hardware
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
            raise RuntimeError(f"Failed to initialise NeoPixel ring: {exc}") from exc

        # Ensure the hardware starts dark.
        self.apply(((0, 0, 0),) * pixel_count)

    def apply(self, colors: Sequence[Color]) -> None:  # pragma: no cover - hardware path
        try:
            for index in range(self.pixel_count):
                self._pixels[index] = colors[index]
            self._pixels.show()
        except Exception:  # pragma: no cover - defensive logging
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

        if driver is not None:
            self._driver: _RingDriver | None = driver
            driver_pixels = getattr(driver, "pixel_count", pixel_count)
            if isinstance(driver_pixels, int) and driver_pixels > 0:
                self._pixel_count = driver_pixels
            else:  # pragma: no cover - defensive fallback
                self._pixel_count = pixel_count
        else:
            self._pixel_count = pixel_count
            try:
                self._driver = _NeoPixelDriver(
                    pixel_count=pixel_count,
                    brightness=max(0.0, min(1.0, brightness)),
                    logger=self._logger,
                )
            except RuntimeError as exc:
                self._logger.info("LED ring disabled: %s", exc)
                self._driver = None

        self._patterns: dict[str, Callable[[], Iterator[PatternStep]]] = {
            "off": self._pattern_off,
            "boot": self._pattern_boot,
            "ready": self._pattern_ready,
            "error": self._pattern_error,
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

    async def get_status(self) -> LedRingStatus:
        """Return the currently configured state of the LED ring."""

        async with self._lock:
            patterns = tuple(self._patterns.keys())
            pattern = self._base_pattern
            active = self._active_pattern
            error_active = self._error_active
            available = self._driver is not None and not self._closed
        return LedRingStatus(
            patterns=patterns,
            pattern=pattern,
            active_pattern=active,
            error=error_active,
            available=available,
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
        except Exception:  # pragma: no cover - defensive logging
            self._logger.debug("LED ring driver apply failed; disabling ring", exc_info=True)
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


__all__ = ["LedRing", "LedRingStatus"]
