"""Tests covering graceful handling of the optional aiortc dependency."""

from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest

from rev_cam.camera import BaseCamera
from rev_cam.config import Orientation
from rev_cam.pipeline import FramePipeline


class DummyCamera(BaseCamera):
    async def get_frame(self) -> np.ndarray:  # pragma: no cover - trivial implementation
        return np.zeros((1, 1, 3), dtype=np.uint8)


def _simulate_missing_aiortc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the import machinery so that aiortc appears to be unavailable."""

    real_import = builtins.__import__

    def fake_import(name: str, globals: dict | None = None, locals: dict | None = None, fromlist=(), level: int = 0):
        if name.startswith("aiortc"):
            raise ModuleNotFoundError("No module named 'aiortc'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for module in list(sys.modules):
        if module.startswith("aiortc"):
            monkeypatch.delitem(sys.modules, module, raising=False)


def test_webrtc_import_succeeds_without_aiortc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the WebRTC helpers should not crash when aiortc is missing."""

    _simulate_missing_aiortc(monkeypatch)
    monkeypatch.delitem(sys.modules, "rev_cam.webrtc", raising=False)

    module = importlib.import_module("rev_cam.webrtc")

    pipeline = FramePipeline(lambda: Orientation())

    with pytest.raises(RuntimeError, match="aiortc is required"):
        module.WebRTCManager(camera=DummyCamera(), pipeline=pipeline)

