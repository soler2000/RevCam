from fractions import Fraction

import numpy as np

from rev_cam import recording


class _DummyStream:
    def __init__(self, time_base: Fraction) -> None:
        self.time_base = time_base
        self.encoded_pts: list[int | None] = []
        self.width = 0
        self.height = 0
        self.pix_fmt = "yuv420p"

    def encode(self, frame=None):
        if frame is not None:
            self.encoded_pts.append(frame.pts)
        return []


class _DummyContainer:
    def __init__(self) -> None:
        self.muxed = []
        self.closed = False

    def mux(self, packet) -> None:
        self.muxed.append(packet)

    def close(self) -> None:
        self.closed = True


def test_chunk_writer_pts_are_relative(tmp_path):
    path = tmp_path / "chunk.mp4"
    stream = _DummyStream(Fraction(1, 1_000_000))
    container = _DummyContainer()
    writer = recording._ActiveChunkWriter(
        name="test",
        index=1,
        path=path,
        container=container,
        stream=stream,
        time_base=Fraction(1, 1_000_000),
        target_width=4,
        target_height=4,
        fps=10.0,
        media_type="video/mp4",
        codec="mpeg4",
        relative_file="media/chunk.mp4",
        pixel_format="yuv420p",
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    writer.add_frame(frame, timestamp=100.0)
    writer.add_frame(frame, timestamp=100.1)
    writer.add_frame(frame, timestamp=100.2)

    assert stream.encoded_pts[0] == 0
    assert stream.encoded_pts[1] > stream.encoded_pts[0]
    assert stream.encoded_pts[2] > stream.encoded_pts[1]

    entry = writer.finalise()
    assert entry["frame_count"] == 3
    assert container.closed
