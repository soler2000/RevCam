import asyncio

import numpy as np
import pytest

from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline
from rev_cam.video import VideoSource


class DummyCamera:
    def __init__(self) -> None:
        self.closed = False

    async def get_frame(self) -> np.ndarray:
        return np.ones((2, 2, 3), dtype=np.uint8)

    async def close(self) -> None:
        self.closed = True


def test_video_source_emits_frames() -> None:
    async def runner() -> None:
        pipeline = FramePipeline(lambda: Orientation())
        source = VideoSource(pipeline, fps=5)

        await source.start()
        camera = DummyCamera()
        await source.set_camera(camera)

        frame = await asyncio.wait_for(source.get_frame(), timeout=1)
        assert frame.shape == (2, 2, 3)
        assert np.array_equal(frame, np.ones((2, 2, 3), dtype=np.uint8))

        await source.stop()
        assert camera.closed is True

    asyncio.run(runner())
