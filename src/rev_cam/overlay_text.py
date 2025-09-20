"""Primitive helpers for drawing text overlays onto frames."""

from __future__ import annotations

from typing import Iterable

try:  # pragma: no cover - optional dependency
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None


FONT_5x7: dict[str, tuple[int, ...]] = {
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111),
    "3": (0b11110, 0b00001, 0b00001, 0b00110, 0b00001, 0b00001, 0b11110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    ".": (0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110),
    " ": (0b00000,) * 7,
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "D": (0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "I": (0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "M": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
    "N": (0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "U": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
    "Z": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111),
    "-": (0b00000, 0b00000, 0b00000, 0b01110, 0b00000, 0b00000, 0b00000),
}


def measure_text(text: str, glyph_width: int, spacing: int) -> int:
    """Return the width in pixels required to render *text*."""

    width = 0
    for index, char in enumerate(text.upper()):
        if index:
            width += spacing
        glyph = FONT_5x7.get(char)
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
    """Render *text* onto *frame* using the 5x7 bitmap font."""

    glyph_w = 5 * scale
    spacing = 1 * scale
    cursor_x = x
    for char in text.upper():
        glyph = FONT_5x7.get(char)
        if glyph is None:
            cursor_x += glyph_w + spacing
            continue
        _draw_glyph(frame, glyph, cursor_x, y, scale, colour)
        cursor_x += glyph_w + spacing


def _draw_glyph(frame, glyph: Iterable[int], x: int, y: int, scale: int, colour: tuple[int, int, int]) -> None:
    for row_index, row in enumerate(glyph):
        for col_index in range(5):
            if (row >> (4 - col_index)) & 1:
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


__all__ = ["FONT_5x7", "apply_background", "draw_text", "measure_text"]

