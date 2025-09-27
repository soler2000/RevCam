"""Primitive helpers for drawing text overlays onto frames."""

from __future__ import annotations

from typing import Iterable

try:  # pragma: no cover - optional dependency
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None


_FONT_BASE_WIDTH = 7
_FONT_BASE_HEIGHT = 9


FONT_SMOOTH_7x9: dict[str, tuple[int, ...]] = {
    "0": (
        0b0011100,
        0b0100010,
        0b1000001,
        0b1001001,
        0b1001001,
        0b1001001,
        0b1000001,
        0b0100010,
        0b0011100,
    ),
    "1": (
        0b0001000,
        0b0011000,
        0b0101000,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0111110,
        0b0000000,
    ),
    "2": (
        0b0011100,
        0b0100010,
        0b0000010,
        0b0000100,
        0b0001000,
        0b0010000,
        0b0100000,
        0b0111110,
        0b0000000,
    ),
    "3": (
        0b0011100,
        0b0100010,
        0b0000010,
        0b0001100,
        0b0000010,
        0b0000010,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    "4": (
        0b0000110,
        0b0001010,
        0b0010010,
        0b0100010,
        0b0111110,
        0b0000010,
        0b0000010,
        0b0000010,
        0b0000000,
    ),
    "5": (
        0b0111110,
        0b0100000,
        0b0100000,
        0b0111100,
        0b0000010,
        0b0000010,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    "6": (
        0b0011100,
        0b0100000,
        0b1000000,
        0b1111100,
        0b1000010,
        0b1000010,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    "7": (
        0b0111110,
        0b0000010,
        0b0000100,
        0b0000100,
        0b0001000,
        0b0001000,
        0b0010000,
        0b0010000,
        0b0000000,
    ),
    "8": (
        0b0011100,
        0b0100010,
        0b0100010,
        0b0011100,
        0b0100010,
        0b0100010,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    "9": (
        0b0011100,
        0b0100010,
        0b0100010,
        0b0011110,
        0b0000010,
        0b0000010,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    ".": (
        0b0000000,
        0b0000000,
        0b0000000,
        0b0000000,
        0b0000000,
        0b0000000,
        0b0001100,
        0b0001100,
        0b0000000,
    ),
    " ": (0b0000000,) * _FONT_BASE_HEIGHT,
    "-": (
        0b0000000,
        0b0000000,
        0b0000000,
        0b0011100,
        0b0000000,
        0b0000000,
        0b0000000,
        0b0000000,
        0b0000000,
    ),
    "%": (
        0b1000010,
        0b1000100,
        0b0001000,
        0b0010000,
        0b0100000,
        0b1000000,
        0b0000100,
        0b0000100,
        0b0000000,
    ),
    "/": (
        0b0000001,
        0b0000010,
        0b0000100,
        0b0001000,
        0b0010000,
        0b0100000,
        0b1000000,
        0b0000000,
        0b0000000,
    ),
    "A": (
        0b0011100,
        0b0100010,
        0b1000001,
        0b1000001,
        0b1111111,
        0b1000001,
        0b1000001,
        0b1000001,
        0b0000000,
    ),
    "B": (
        0b1111100,
        0b1000010,
        0b1000010,
        0b1111100,
        0b1000010,
        0b1000010,
        0b1000010,
        0b1111100,
        0b0000000,
    ),
    "C": (
        0b0011110,
        0b0100001,
        0b1000000,
        0b1000000,
        0b1000000,
        0b1000000,
        0b0100001,
        0b0011110,
        0b0000000,
    ),
    "D": (
        0b1111100,
        0b1000010,
        0b1000001,
        0b1000001,
        0b1000001,
        0b1000001,
        0b1000010,
        0b1111100,
        0b0000000,
    ),
    "E": (
        0b1111110,
        0b1000000,
        0b1000000,
        0b1111100,
        0b1000000,
        0b1000000,
        0b1000000,
        0b1111110,
        0b0000000,
    ),
    "F": (
        0b1111110,
        0b1000000,
        0b1000000,
        0b1111100,
        0b1000000,
        0b1000000,
        0b1000000,
        0b1000000,
        0b0000000,
    ),
    "G": (
        0b0011110,
        0b0100000,
        0b1000000,
        0b1001110,
        0b1000010,
        0b1000010,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    "H": (
        0b1000001,
        0b1000001,
        0b1000001,
        0b1111111,
        0b1000001,
        0b1000001,
        0b1000001,
        0b1000001,
        0b0000000,
    ),
    "I": (
        0b0111110,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0111110,
        0b0000000,
    ),
    "L": (
        0b1000000,
        0b1000000,
        0b1000000,
        0b1000000,
        0b1000000,
        0b1000000,
        0b1000000,
        0b1111110,
        0b0000000,
    ),
    "M": (
        0b1000001,
        0b1100011,
        0b1010101,
        0b1010101,
        0b1001001,
        0b1000001,
        0b1000001,
        0b1000001,
        0b0000000,
    ),
    "N": (
        0b1000001,
        0b1100001,
        0b1010001,
        0b1001001,
        0b1000101,
        0b1000011,
        0b1000001,
        0b1000001,
        0b0000000,
    ),
    "O": (
        0b0011100,
        0b0100010,
        0b1000001,
        0b1000001,
        0b1000001,
        0b1000001,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    "R": (
        0b1111100,
        0b1000010,
        0b1000010,
        0b1111100,
        0b1010000,
        0b1001000,
        0b1000100,
        0b1000010,
        0b0000000,
    ),
    "S": (
        0b0011110,
        0b0100000,
        0b0100000,
        0b0011100,
        0b0000010,
        0b0000010,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    "T": (
        0b1111110,
        0b0011000,
        0b0011000,
        0b0011000,
        0b0011000,
        0b0011000,
        0b0011000,
        0b0011000,
        0b0000000,
    ),
    "U": (
        0b1000001,
        0b1000001,
        0b1000001,
        0b1000001,
        0b1000001,
        0b1000001,
        0b0100010,
        0b0011100,
        0b0000000,
    ),
    "V": (
        0b1000001,
        0b1000001,
        0b1000001,
        0b0100010,
        0b0100010,
        0b0010100,
        0b0010100,
        0b0001000,
        0b0000000,
    ),
    "W": (
        0b1000001,
        0b1000001,
        0b1000001,
        0b1001001,
        0b1010101,
        0b1010101,
        0b1100011,
        0b1000001,
        0b0000000,
    ),
    "Y": (
        0b1000001,
        0b0100010,
        0b0010100,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0001000,
        0b0000000,
    ),
    "Z": (
        0b1111110,
        0b0000010,
        0b0000100,
        0b0001000,
        0b0010000,
        0b0100000,
        0b1000000,
        0b1111110,
        0b0000000,
    ),
}


def measure_text(text: str, glyph_width: int, spacing: int) -> int:
    """Return the width in pixels required to render *text*."""

    width = 0
    for index, char in enumerate(text.upper()):
        if index:
            width += spacing
        glyph = FONT_SMOOTH_7x9.get(char)
        width += glyph_width if glyph else glyph_width
    return width


def apply_background(frame, x: int, y: int, width: int, height: int, colour: tuple[int, int, int]) -> None:
    """Tint the frame region with a translucent background."""

    if _np is None:
        return
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame.shape[1], x + width)
    y1 = min(frame.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1].astype(_np.float32)
    tint = _np.array(colour, dtype=_np.float32)
    blended = (region * 0.3 + tint * 0.2).clip(0, 255).astype(frame.dtype)
    frame[y0:y1, x0:x1] = blended
    frame[y0:y0 + 2, x0:x1] = colour
    frame[y1 - 2:y1, x0:x1] = colour
    frame[y0:y1, x0:x0 + 2] = colour
    frame[y0:y1, x1 - 2:x1] = colour


def draw_text(frame, text: str, x: int, y: int, scale: int, colour: tuple[int, int, int]) -> None:
    """Render *text* onto *frame* using the smoother 7x9 bitmap font."""

    glyph_w = _FONT_BASE_WIDTH * scale
    spacing = 1 * scale
    cursor_x = x
    for char in text.upper():
        glyph = FONT_SMOOTH_7x9.get(char)
        if glyph is None:
            cursor_x += glyph_w + spacing
            continue
        _draw_glyph(frame, glyph, cursor_x, y, scale, colour)
        cursor_x += glyph_w + spacing


def _draw_glyph(frame, glyph: Iterable[int], x: int, y: int, scale: int, colour: tuple[int, int, int]) -> None:
    for row_index, row in enumerate(glyph):
        for col_index in range(_FONT_BASE_WIDTH):
            if (row >> (_FONT_BASE_WIDTH - 1 - col_index)) & 1:
                _fill_rect(
                    frame,
                    x + col_index * scale,
                    y + row_index * scale,
                    scale,
                    scale,
                    colour,
                )


def _fill_rect(frame, x: int, y: int, width: int, height: int, colour: tuple[int, int, int]) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame.shape[1], x + width)
    y1 = min(frame.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] = colour


FONT_5x7 = FONT_SMOOTH_7x9  # Backwards compatibility for existing imports


FONT_BASE_WIDTH = _FONT_BASE_WIDTH
FONT_BASE_HEIGHT = _FONT_BASE_HEIGHT


__all__ = [
    "FONT_SMOOTH_7x9",
    "FONT_5x7",
    "FONT_BASE_WIDTH",
    "FONT_BASE_HEIGHT",
    "apply_background",
    "draw_text",
    "measure_text",
]

