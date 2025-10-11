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


class _DummyPacket:
    def __init__(self, time_base: Fraction | None = None, size: int = 0) -> None:
        self.time_base = time_base
        self.size = size


class _DummyContainer:
    def __init__(self) -> None:
        self.muxed = []
        self.closed = False

    def mux(self, packet) -> None:
        self.muxed.append(packet)

    def close(self) -> None:
        self.closed = True


class _DynamicCodecContext:
    def __init__(self, time_base: Fraction) -> None:
        self.time_base = time_base


class _DynamicTimeBaseStream(_DummyStream):
    def __init__(
        self, initial: Fraction, updated: Fraction, *, switch_after: int = 1
    ) -> None:
        super().__init__(initial)
        self.codec_context = _DynamicCodecContext(initial)
        self._updated = updated
        self._switch_after = max(1, int(switch_after))

    def encode(self, frame=None):
        if frame is not None:
            self.encoded_pts.append(frame.pts)
            if len(self.encoded_pts) == self._switch_after:
                self.time_base = self._updated
                self.codec_context.time_base = self._updated
        return []


class _PacketTimeBaseStream(_DummyStream):
    def __init__(self, initial: Fraction, packet: Fraction) -> None:
        super().__init__(initial)
        self._packet_time_base = packet
        self._packets_emitted = 0

    def encode(self, frame=None):
        if frame is not None:
            self.encoded_pts.append(frame.pts)
            time_base = (
                self._packet_time_base if self._packets_emitted == 0 else self.time_base
            )
            self._packets_emitted += 1
            return [_DummyPacket(time_base=time_base)]
        return []


def test_chunk_writer_pts_are_relative(tmp_path):
    path = tmp_path / "chunk.avi"
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
        media_type="video/x-msvideo",
        codec="mpeg4",
        relative_file="media/chunk.avi",
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


def test_chunk_writer_pts_resist_timestamp_jitter(tmp_path):
    path = tmp_path / "chunk.avi"
    time_base = recording._select_time_base(Fraction(10, 1))
    stream = _DummyStream(time_base)
    container = _DummyContainer()
    writer = recording._ActiveChunkWriter(
        name="test",
        index=2,
        path=path,
        container=container,
        stream=stream,
        time_base=time_base,
        target_width=4,
        target_height=4,
        fps=10.0,
        media_type="video/x-msvideo",
        codec="libx264",
        relative_file="media/chunk.avi",
        pixel_format="yuv420p",
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    timestamps = [0.0, 0.08, 0.22, 0.31, 0.49]
    for ts in timestamps:
        writer.add_frame(frame, timestamp=ts)

    pts_values = stream.encoded_pts
    assert len(pts_values) == len(timestamps)
    deltas = [b - a for a, b in zip(pts_values, pts_values[1:])]
    assert deltas
    assert all(delta == deltas[0] for delta in deltas)

    frame_rate_fraction = Fraction(str(writer.fps)).limit_denominator(1_000_000)
    expected_increment_fraction = (Fraction(1, 1) / frame_rate_fraction) / time_base
    assert expected_increment_fraction.denominator == 1
    assert deltas[0] == expected_increment_fraction.numerator


def test_chunk_writer_uses_stream_time_base_for_tick_math(tmp_path):
    path = tmp_path / "chunk.avi"
    stream_time_base = Fraction(1, 1000)
    stream = _DummyStream(stream_time_base)
    container = _DummyContainer()
    writer = recording._ActiveChunkWriter(
        name="test",
        index=3,
        path=path,
        container=container,
        stream=stream,
        time_base=Fraction(1, 24),
        target_width=4,
        target_height=4,
        fps=7.0,
        media_type="video/x-msvideo",
        codec="libx264",
        relative_file="media/chunk.avi",
        pixel_format="yuv420p",
    )

    assert writer.time_base == stream_time_base

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    timestamps = [0.0, 0.19, 0.33, 0.51, 0.69, 0.82]
    for ts in timestamps:
        writer.add_frame(frame, timestamp=ts)

    pts_values = stream.encoded_pts
    assert len(pts_values) == len(timestamps)
    deltas = [b - a for a, b in zip(pts_values, pts_values[1:])]
    assert deltas
    assert max(deltas) - min(deltas) <= 1

    entry = writer.finalise()
    expected_duration = len(pts_values) / writer.fps
    assert entry["duration_seconds"] == pytest.approx(expected_duration, rel=1e-6, abs=1e-3)
    assert container.closed


def test_chunk_writer_retimes_when_stream_updates_time_base(tmp_path):
    path = tmp_path / "chunk.avi"
    initial_time_base = Fraction(1, 1000)
    updated_time_base = Fraction(1, 6000)
    stream = _DynamicTimeBaseStream(initial_time_base, updated_time_base)
    container = _DummyContainer()
    writer = recording._ActiveChunkWriter(
        name="test",
        index=4,
        path=path,
        container=container,
        stream=stream,
        time_base=initial_time_base,
        target_width=4,
        target_height=4,
        fps=10.0,
        media_type="video/x-msvideo",
        codec="libx264",
        relative_file="media/chunk.avi",
        pixel_format="yuv420p",
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    timestamps = [i * 0.1 for i in range(6)]
    for ts in timestamps:
        writer.add_frame(frame, timestamp=ts)

    assert writer.time_base == updated_time_base

    pts_values = stream.encoded_pts
    assert len(pts_values) == len(timestamps)
    deltas = [b - a for a, b in zip(pts_values, pts_values[1:])]
    assert deltas

    frame_rate_fraction = Fraction(str(writer.fps)).limit_denominator(1_000_000)
    expected_increment_fraction = (
        (Fraction(1, 1) / frame_rate_fraction) / updated_time_base
    )
    assert expected_increment_fraction.denominator == 1
    assert all(delta == expected_increment_fraction.numerator for delta in deltas)

    entry = writer.finalise()
    expected_duration = len(pts_values) / writer.fps
    assert entry["duration_seconds"] == pytest.approx(expected_duration, rel=1e-6, abs=1e-3)
    assert container.closed


def test_chunk_writer_tracks_packet_announced_time_base(tmp_path):
    path = tmp_path / "chunk.avi"
    initial_time_base = Fraction(1, 20)
    packet_time_base = Fraction(1, 10240)
    stream = _PacketTimeBaseStream(initial_time_base, packet_time_base)
    container = _DummyContainer()
    writer = recording._ActiveChunkWriter(
        name="test",
        index=5,
        path=path,
        container=container,
        stream=stream,
        time_base=initial_time_base,
        target_width=4,
        target_height=4,
        fps=20.0,
        media_type="video/x-msvideo",
        codec="libx264",
        relative_file="media/chunk.avi",
        pixel_format="yuv420p",
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    timestamps = [i * 0.05 for i in range(4)]
    for ts in timestamps:
        writer.add_frame(frame, timestamp=ts)

    assert writer.time_base == packet_time_base

    pts_values = stream.encoded_pts
    assert len(pts_values) == len(timestamps)
    expected_increment = ((Fraction(1, 1) / Fraction(20, 1)) / packet_time_base).limit_denominator()
    assert expected_increment.denominator == 1
    deltas = [b - a for a, b in zip(pts_values, pts_values[1:])]
    assert deltas
    assert all(delta == expected_increment.numerator for delta in deltas)

    entry = writer.finalise()
    expected_duration = len(pts_values) / writer.fps
    assert entry["duration_seconds"] == pytest.approx(expected_duration, rel=1e-6, abs=1e-3)
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
    assert result == Fraction(1, 30)


def test_single_frame_duration_respects_time_base(tmp_path):
    path = tmp_path / "chunk.avi"
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
        media_type="video/x-msvideo",
        codec="libx264",
        relative_file="media/chunk.avi",
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
