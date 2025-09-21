import asyncio

import pytest

from rev_cam.led_matrix import LedRing, NEOPIXEL_PERMISSION_MESSAGE


class _RecordingDriver:
    def __init__(self, pixel_count: int = 6) -> None:
        self.pixel_count = pixel_count
        self.frames: list[tuple[tuple[int, int, int], ...]] = []
        self.closed = False

    def apply(self, colors):
        self.frames.append(tuple(colors))

    def close(self) -> None:
        self.closed = True


class _PermissionDriver:
    def __init__(self, pixel_count: int = 4) -> None:
        self.pixel_count = pixel_count
        self.closed = False

    def apply(self, colors):
        raise RuntimeError(
            "NeoPixel driver requires root privileges (sudo) to access /dev/mem."
        )

    def close(self) -> None:
        self.closed = True


def test_led_ring_patterns_cycle() -> None:
    async def runner() -> None:
        driver = _RecordingDriver(pixel_count=4)
        ring = LedRing(pixel_count=4, driver=driver)

        await ring.set_pattern("boot")
        await asyncio.sleep(0.09)
        assert driver.frames, "Boot pattern should emit frames"
        first = driver.frames[0]
        assert sum(1 for color in first if color != (0, 0, 0)) == 1

        await ring.set_pattern("ready")
        ready_frame = driver.frames[-1]
        assert len(set(ready_frame)) == 1

        await ring.set_error(True)
        await asyncio.sleep(0.02)
        assert any(color[0] > color[1] for color in driver.frames[-1])

        await ring.set_error(False)
        await asyncio.sleep(0.02)
        assert driver.frames[-1] == ready_frame

        await ring.aclose()
        assert driver.closed

    asyncio.run(runner())


def test_led_ring_invalid_pattern() -> None:
    async def runner() -> None:
        ring = LedRing(pixel_count=3, driver=_RecordingDriver(pixel_count=3))
        with pytest.raises(ValueError):
            await ring.set_pattern("unknown")
        await ring.aclose()

    asyncio.run(runner())


def test_led_ring_reports_driver_permission_failure() -> None:
    async def runner() -> None:
        driver = _PermissionDriver(pixel_count=4)
        ring = LedRing(pixel_count=4, driver=driver)

        await ring.set_pattern("boot")
        status = await ring.get_status()
        assert status.available is False
        assert status.message == NEOPIXEL_PERMISSION_MESSAGE
        assert driver.closed is True

        await ring.aclose()

    asyncio.run(runner())
