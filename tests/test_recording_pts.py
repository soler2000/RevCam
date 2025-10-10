from fractions import Fraction

import numpy as np
import pytest

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


class _DummyCodecContext:
    def __init__(self, formats) -> None:
        self.pix_fmts = formats
        self.time_base = None
        self.pix_fmt = None


class _DummyStreamWithFormats:
    def __init__(self, formats) -> None:
        self.codec_context = _DummyCodecContext(formats)
        self.pix_fmt = None


class _DummyTimingCodecContext:
    def __init__(self) -> None:
        self.time_base = None
        self.framerate = None
        self.ticks_per_frame = 2
        self.pix_fmt = None


class _DummyStreamWithTiming:
    def __init__(self) -> None:
        self.codec_context = _DummyTimingCodecContext()
        self.pix_fmt = None
        self.time_base = None
        self.rate = None
        self.average_rate = None


def test_select_stream_pixel_format_prefers_requested_when_available():
    stream = _DummyStreamWithFormats(["yuv420p", "nv12"])
    result = recording._select_stream_pixel_format(stream, "yuv420p")
    assert result == "yuv420p"


def test_select_stream_pixel_format_falls_back_to_supported_format():
    stream = _DummyStreamWithFormats([b"NV12"])
    result = recording._select_stream_pixel_format(stream, "yuv420p")
    assert result == "nv12"


def test_prepare_frame_for_encoding_normalises_shape_and_dtype():
    source = np.linspace(0, 1024, 36, dtype=np.float32).reshape(6, 6)
    prepared = recording._prepare_frame_for_encoding(source)
    assert prepared is not None
    assert prepared.dtype == np.uint8
    assert prepared.shape == (6, 6, 3)
    assert prepared.flags["C_CONTIGUOUS"]


def test_prepare_frame_for_encoding_trims_extra_channels():
    source = np.ones((4, 4, 5), dtype=np.uint16) * 1024
    prepared = recording._prepare_frame_for_encoding(source)
    assert prepared is not None
    assert prepared.shape == (4, 4, 3)
    assert np.all(prepared == 255)


def test_select_time_base_matches_frame_duration():
    result = recording._select_time_base(Fraction(30, 1))
    assert result == Fraction(1, 30_000)


def test_single_frame_duration_respects_time_base(tmp_path):
    path = tmp_path / "chunk.mp4"
    path.write_bytes(b"\x00")
    time_base = recording._select_time_base(Fraction(30, 1))
    stream = _DummyStream(time_base)
    container = _DummyContainer()
    writer = recording._ActiveChunkWriter(
        name="test",
        index=1,
        path=path,
        container=container,
        stream=stream,
        time_base=time_base,
        target_width=4,
        target_height=4,
        fps=30.0,
        media_type="video/mp4",
        codec="libx264",
        relative_file="media/chunk.mp4",
        pixel_format="yuv420p",
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    writer.add_frame(frame, timestamp=0.0)
    entry = writer.finalise()

    assert entry["frame_count"] == 1
    assert entry["duration_seconds"] == pytest.approx(1 / 30, abs=1e-3)


def test_apply_stream_timing_sets_codec_context_properties():
    stream = _DummyStreamWithTiming()
    frame_rate = Fraction(30, 1)
    time_base = recording._select_time_base(frame_rate)

    recording._apply_stream_timing(stream, frame_rate, time_base)

    assert stream.time_base == time_base
    assert stream.rate == frame_rate
    assert stream.average_rate == frame_rate
    assert stream.codec_context.time_base == time_base
    assert stream.codec_context.framerate == frame_rate
    assert stream.codec_context.ticks_per_frame == 1
