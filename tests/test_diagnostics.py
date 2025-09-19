from __future__ import annotations

import json

import pytest

import rev_cam.diagnostics as diagnostics


def test_run_outputs_conflicts(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose_camera_conflicts", lambda: ["service running"])

    exit_code = diagnostics.run([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "service running" in out
    assert diagnostics.APP_VERSION in out


def test_run_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose_camera_conflicts", lambda: ["process"])

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
