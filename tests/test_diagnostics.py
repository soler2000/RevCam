from __future__ import annotations

import json

import types

import pytest

pytest.importorskip("numpy")

import rev_cam.diagnostics as diagnostics


def test_collect_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose_camera_conflicts", lambda: ["service"])
    monkeypatch.setattr(
        diagnostics,
        "diagnose_picamera_stack",
        lambda: {"status": "ok", "details": []},
    )
    monkeypatch.setattr(
        diagnostics,
        "diagnose_webrtc_stack",
        lambda: {"status": "ok", "details": []},
    )
    monkeypatch.setattr(
        diagnostics,
        "collect_system_metrics",
        lambda: {
            "cpu": {
                "usage_percent": 12.5,
                "count": 4,
                "load": {"1m": 0.5},
                "per_core": [
                    {"index": 0, "usage_percent": 10.0},
                    {"index": 1, "usage_percent": 15.0},
                ],
            },
            "memory": {"used_percent": 42.0},
        },
    )

    payload = diagnostics.collect_diagnostics()

    assert payload["version"] == diagnostics.APP_VERSION
    assert payload["camera_conflicts"] == ["service"]
    assert payload["picamera"] == {"status": "ok", "details": []}
    assert payload["webrtc"] == {"status": "ok", "details": []}
    assert payload["system"] == {
        "cpu": {
            "usage_percent": 12.5,
            "count": 4,
            "load": {"1m": 0.5},
            "per_core": [
                {"index": 0, "usage_percent": 10.0},
                {"index": 1, "usage_percent": 15.0},
            ],
        },
        "memory": {"used_percent": 42.0},
    }


def test_run_outputs_conflicts(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose_camera_conflicts", lambda: ["service running"])
    monkeypatch.setattr(
        diagnostics,
        "diagnose_picamera_stack",
        lambda: {"status": "error", "details": ["picamera2 module not found."], "hints": ["install"]},
    )
    monkeypatch.setattr(
        diagnostics,
        "diagnose_webrtc_stack",
        lambda: {"status": "error", "details": ["aiortc module not found."], "hints": ["install"]},
    )
    monkeypatch.setattr(
        diagnostics,
        "collect_system_metrics",
        lambda: {
            "cpu": {
                "usage_percent": 65.0,
                "count": 4,
                "load": {"1m": 2.6},
                "per_core": [
                    {"index": 0, "usage_percent": 72.5},
                    {"index": 1, "usage_percent": 58.0},
                ],
            },
            "memory": {
                "used_percent": 71.0,
                "used_bytes": 2147483648,
                "total_bytes": 4294967296,
            },
        },
    )

    exit_code = diagnostics.run([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "service running" in out
    assert "Picamera2 Python stack issues detected:" in out
    assert "picamera2 module not found." in out
    assert "WebRTC stack issues detected:" in out
    assert "aiortc module not found." in out
    assert diagnostics.APP_VERSION in out
    assert "System resource usage:" in out
    assert "CPU:" in out
    assert "Per-core usage:" in out
    assert "Memory:" in out


def test_run_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose_camera_conflicts", lambda: ["process"])
    monkeypatch.setattr(
        diagnostics,
        "diagnose_picamera_stack",
        lambda: {"status": "ok", "details": [], "numpy_version": "1.26.1"},
    )
    monkeypatch.setattr(
        diagnostics,
        "diagnose_webrtc_stack",
        lambda: {"status": "ok", "details": []},
    )
    monkeypatch.setattr(
        diagnostics,
        "collect_system_metrics",
        lambda: {
            "cpu": {
                "usage_percent": 33.3,
                "count": 4,
                "load": {"1m": 1.2},
                "per_core": [
                    {"index": 0, "usage_percent": 25.0},
                    {"index": 1, "usage_percent": 41.5},
                ],
            },
            "memory": {
                "used_percent": 55.5,
                "used_bytes": 123456789,
                "total_bytes": 987654321,
            },
        },
    )

    buffer: list[str] = []

    class DummyIO:
        def write(self, data: str) -> int:
            buffer.append(data)
            return len(data)

    monkeypatch.setattr(diagnostics.sys, "stdout", DummyIO())

    exit_code = diagnostics.run(["--json"])

    assert exit_code == 0
    payload = json.loads("".join(buffer))
    assert payload["camera_conflicts"] == ["process"]
    assert payload["version"] == diagnostics.APP_VERSION
    assert payload["picamera"] == {"status": "ok", "details": [], "numpy_version": "1.26.1"}
    assert payload["webrtc"] == {"status": "ok", "details": []}
    assert payload["system"] == {
        "cpu": {
            "usage_percent": 33.3,
            "count": 4,
            "load": {"1m": 1.2},
            "per_core": [
                {"index": 0, "usage_percent": 25.0},
                {"index": 1, "usage_percent": 41.5},
            ],
        },
        "memory": {
            "used_percent": 55.5,
            "used_bytes": 123456789,
            "total_bytes": 987654321,
        },
    }


def test_diagnose_picamera_stack_numpy_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str):
        if name == "numpy":
            return types.SimpleNamespace(__version__="1.26.2")
        if name == "picamera2":
            raise ValueError("numpy.dtype size changed, may indicate binary incompatibility")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(diagnostics.importlib, "import_module", fake_import)

    result = diagnostics.diagnose_picamera_stack()

    assert result["status"] == "error"
    assert any("picamera2 import failed" in detail for detail in result["details"])
    assert diagnostics.NUMPY_ABI_HINT in result["hints"]


def test_diagnose_picamera_stack_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str):
        if name == "numpy":
            return types.SimpleNamespace(__version__="1.26.2")
        if name == "picamera2":
            return object()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(diagnostics.importlib, "import_module", fake_import)

    result = diagnostics.diagnose_picamera_stack()

    assert result == {"status": "ok", "details": [], "numpy_version": "1.26.2"}


def test_diagnose_webrtc_stack_missing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(diagnostics.importlib, "import_module", fake_import)

    result = diagnostics.diagnose_webrtc_stack()

    assert result["status"] == "error"
    assert any("PyAV module not found." in detail for detail in result["details"])
    assert diagnostics.WEBRTC_PIP_HINT in result["hints"]


def test_diagnose_webrtc_stack_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str):
        if name == "av":
            return types.SimpleNamespace(VideoFrame=object())
        if name == "aiortc":
            return types.SimpleNamespace()
        if name == "aiortc.rtcrtpsender":
            return object()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(diagnostics.importlib, "import_module", fake_import)

    result = diagnostics.diagnose_webrtc_stack()

    assert result == {"status": "ok", "details": []}
