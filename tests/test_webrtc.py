import asyncio
from fractions import Fraction
from unittest.mock import AsyncMock

from aiortc import RTCPeerConnection, RTCSessionDescription
import numpy as np

from rev_cam.camera import BaseCamera
from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline
from rev_cam.webrtc import MediaMTXError, PipelineVideoTrack, WebRTCManager


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


class StubMediaMTXClient:
    def __init__(self) -> None:
        self.publish_calls = 0
        self.play_offers: list[str] = []
        self._pcs: list[RTCPeerConnection] = []
        self.closed = False

    async def publish(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        self._pcs.append(pc)
        self.publish_calls += 1
        assert pc.localDescription is not None
        return pc.localDescription

    async def play(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        self.play_offers.append(offer.sdp)
        return RTCSessionDescription(sdp="dummy-answer", type="answer")

    async def close(self) -> None:
        for pc in self._pcs:
            await pc.close()
        self.closed = True


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


class ReusedBufferCamera(BaseCamera):
    def __init__(self) -> None:
        self._buffer = np.zeros((2, 2, 3), dtype=np.uint8)
        self._values = iter((0, 50))
        self._last_value = 50

    async def get_frame(self) -> np.ndarray:
        try:
            value = next(self._values)
        except StopIteration:
            value = self._last_value
        self._buffer.fill(value)
        self._last_value = value
        await asyncio.sleep(0.001)
        return self._buffer


def test_pipeline_video_track_copies_camera_frames() -> None:
    async def _test() -> None:
        camera = ReusedBufferCamera()
        pipeline = FramePipeline(lambda: Orientation())
        track = PipelineVideoTrack(camera, pipeline, fps=30)
        track.next_timestamp = AsyncMock(
            side_effect=[(0, Fraction(1, 30)), (1, Fraction(1, 30))]
        )
        try:
            first_frame = await track.recv()
            await asyncio.sleep(0.02)
            second_frame = await track.recv()
            second_pixels = second_frame.to_ndarray(format="rgb24")
            first_pixels = first_frame.to_ndarray(format="rgb24")
        finally:
            track.stop()
            await asyncio.sleep(0)

        assert np.all(first_pixels == 0)
        assert np.all(second_pixels == 50)

    run_async(_test())


def test_webrtc_manager_proxies_offers_via_mediamtx():
    async def _test() -> None:
        client = StubMediaMTXClient()
        camera = CountingCamera()
        pipeline = FramePipeline(lambda: Orientation())
        manager = WebRTCManager(camera=camera, pipeline=pipeline, client=client)
        offer = RTCSessionDescription(sdp="offer-a", type="offer")
        answer = await manager.handle_offer(offer)
        assert answer.sdp == "dummy-answer"
        assert client.publish_calls == 1
        assert client.play_offers == ["offer-a"]
        await manager.shutdown()
        assert client.closed is True

    run_async(_test())


def test_webrtc_manager_reuses_publisher_for_multiple_offers():
    async def _test() -> None:
        client = StubMediaMTXClient()
        camera = CountingCamera()
        pipeline = FramePipeline(lambda: Orientation())
        manager = WebRTCManager(camera=camera, pipeline=pipeline, client=client)
        offer1 = RTCSessionDescription(sdp="offer-1", type="offer")
        offer2 = RTCSessionDescription(sdp="offer-2", type="offer")
        await manager.handle_offer(offer1)
        await manager.handle_offer(offer2)
        assert client.publish_calls == 1
        assert client.play_offers == ["offer-1", "offer-2"]
        await manager.shutdown()

    run_async(_test())


class _FailingPublishClient(StubMediaMTXClient):
    async def publish(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        raise MediaMTXError("publisher offline")


class _FailingPlayClient(StubMediaMTXClient):
    async def play(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        raise MediaMTXError("viewer endpoint offline")


async def _create_viewer_offer() -> tuple[RTCPeerConnection, RTCSessionDescription]:
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    assert pc.localDescription is not None
    return pc, pc.localDescription


async def _consume_answer(pc: RTCPeerConnection, answer: RTCSessionDescription) -> None:
    await pc.setRemoteDescription(answer)
    await asyncio.sleep(0.01)


def test_webrtc_manager_falls_back_when_mediamtx_unavailable() -> None:
    async def _test() -> None:
        pipeline = FramePipeline(lambda: Orientation())

        publish_camera = CountingCamera()
        publish_client = _FailingPublishClient()
        manager = WebRTCManager(camera=publish_camera, pipeline=pipeline, client=publish_client)
        viewer, offer = await _create_viewer_offer()
        answer = await manager.handle_offer(offer)
        assert answer.type == "answer"
        await _consume_answer(viewer, answer)
        await asyncio.sleep(0.05)
        assert publish_camera.calls > 0
        assert publish_client.closed is True
        await viewer.close()
        await manager.shutdown()

        play_camera = CountingCamera()
        play_client = _FailingPlayClient()
        manager = WebRTCManager(camera=play_camera, pipeline=pipeline, client=play_client)
        viewer, offer = await _create_viewer_offer()
        answer = await manager.handle_offer(offer)
        assert answer.type == "answer"
        await _consume_answer(viewer, answer)
        await asyncio.sleep(0.05)
        assert play_camera.calls > 0
        assert play_client.publish_calls == 1
        assert play_client.closed is True
        await viewer.close()
        await manager.shutdown()

    run_async(_test())


def test_webrtc_manager_restarts_publisher_on_camera_change():
    async def _test() -> None:
        client = StubMediaMTXClient()
        camera = CountingCamera()
        pipeline = FramePipeline(lambda: Orientation())
        manager = WebRTCManager(camera=camera, pipeline=pipeline, client=client)
        await manager.handle_offer(RTCSessionDescription(sdp="offer-1", type="offer"))
        assert client.publish_calls == 1

        new_camera = CountingCamera()
        await manager.set_camera(new_camera)
        await manager.handle_offer(RTCSessionDescription(sdp="offer-2", type="offer"))
        assert client.publish_calls == 2
        assert client.play_offers == ["offer-1", "offer-2"]
        await manager.shutdown()

    run_async(_test())
