from __future__ import annotations

import json

import types

import pytest

pytest.importorskip("numpy")

import rev_cam.diagnostics as diagnostics


def test_run_outputs_conflicts(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose_camera_conflicts", lambda: ["service running"])
    monkeypatch.setattr(
        diagnostics,
        "diagnose_picamera_stack",
        lambda: {"status": "error", "details": ["picamera2 module not found."], "hints": ["install"]},
    )

    exit_code = diagnostics.run([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "service running" in out
    assert "Picamera2 Python stack issues detected:" in out
    assert "picamera2 module not found." in out
    assert diagnostics.APP_VERSION in out


def test_run_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose_camera_conflicts", lambda: ["process"])
    monkeypatch.setattr(
        diagnostics,
        "diagnose_picamera_stack",
        lambda: {"status": "ok", "details": [], "numpy_version": "1.26.1"},
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
