import asyncio
from fractions import Fraction
from unittest.mock import AsyncMock

import numpy as np

from rev_cam.camera import BaseCamera
from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline
from rev_cam.webrtc import PipelineVideoTrack


class CountingCamera(BaseCamera):
    def __init__(self) -> None:
        self.calls = 0

    async def get_frame(self) -> np.ndarray:
        value = self.calls % 256
        self.calls += 1
        await asyncio.sleep(0.001)
        return np.full((2, 2, 3), value, dtype=np.uint8)


def run_async(coro):
    return asyncio.run(coro)


def test_pipeline_video_track_drops_intermediate_frames():
    async def _test() -> None:
        camera = CountingCamera()
        pipeline = FramePipeline(lambda: Orientation())
        track = PipelineVideoTrack(camera, pipeline, fps=30)
        track.next_timestamp = AsyncMock(
            side_effect=[(0, Fraction(1, 30)), (1, Fraction(1, 30))]
        )
        try:
            first_frame = await track.recv()
            calls_after_first = camera.calls
            await asyncio.sleep(0.05)
            second_frame = await track.recv()
        finally:
            track.stop()
            await asyncio.sleep(0)
        final_calls = camera.calls
        first_value = int(first_frame.to_ndarray(format="rgb24")[0, 0, 0])
        second_value = int(second_frame.to_ndarray(format="rgb24")[0, 0, 0])
        frame_diff = (second_value - first_value) % 256
        assert final_calls - calls_after_first >= 5
        assert frame_diff >= 5

    run_async(_test())
