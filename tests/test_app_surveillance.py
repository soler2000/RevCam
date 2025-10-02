"""Tests for the surveillance settings API."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam import app as app_module
from rev_cam.config import ConfigManager, SURVEILLANCE_STANDARD_PRESETS


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    class _StubBatteryMonitor:
        def __init__(self, *args, **kwargs) -> None:
            self.capacity_mah = kwargs.get("capacity_mah", 1000)

        def read(self):  # pragma: no cover - indirect use
            return SimpleNamespace(to_dict=lambda: {})

    class _StubSupervisor:
        def start(self) -> None:  # pragma: no cover - indirect use
            return None

        async def aclose(self) -> None:  # pragma: no cover - indirect use
            return None

    class _StubDistanceMonitor:
        def __init__(self, *args, **kwargs) -> None:
            self.i2c_bus = kwargs.get("i2c_bus")

    def _noop_overlay(*args, **kwargs):
        return lambda frame: frame

    monkeypatch.setattr(app_module, "BatteryMonitor", lambda *args, **kwargs: _StubBatteryMonitor(*args, **kwargs))
    monkeypatch.setattr(app_module, "BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr(app_module, "DistanceMonitor", lambda *args, **kwargs: _StubDistanceMonitor(*args, **kwargs))
    monkeypatch.setattr(app_module, "create_battery_overlay", lambda *args, **kwargs: _noop_overlay())
    monkeypatch.setattr(app_module, "create_wifi_overlay", lambda *args, **kwargs: _noop_overlay())
    monkeypatch.setattr(app_module, "create_distance_overlay", lambda *args, **kwargs: _noop_overlay())
    monkeypatch.setattr(app_module, "create_reversing_aids_overlay", lambda *args, **kwargs: _noop_overlay())

    config_path = tmp_path / "config.json"
    app = app_module.create_app(config_path)
    with TestClient(app) as test_client:
        test_client.config_path = config_path
        yield test_client


def test_get_surveillance_settings(client: TestClient) -> None:
    response = client.get("/api/surveillance/settings")
    assert response.status_code == 200
    payload = response.json()
    assert "settings" in payload
    assert payload["settings"]["profile"] == "standard"
    assert payload["settings"]["preset"] in SURVEILLANCE_STANDARD_PRESETS
    assert "presets" in payload
    assert any(item["name"] == payload["settings"]["preset"] for item in payload["presets"])


def test_update_surveillance_settings_standard(client: TestClient) -> None:
    response = client.post(
        "/api/surveillance/settings",
        json={"profile": "standard", "preset": "detail"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["settings"]["profile"] == "standard"
    assert payload["settings"]["preset"] == "detail"
    expected = SURVEILLANCE_STANDARD_PRESETS["detail"]
    assert payload["settings"]["fps"] == expected[0]
    assert payload["settings"]["jpeg_quality"] == expected[1]

    manager = ConfigManager(client.config_path)
    stored = manager.get_surveillance_settings()
    assert stored.preset == "detail"
    assert stored.resolved_fps == expected[0]


def test_update_surveillance_settings_expert_validation(client: TestClient) -> None:
    response = client.post(
        "/api/surveillance/settings",
        json={"profile": "expert", "expert_fps": "fast"},
    )
    assert response.status_code == 400
    detail = response.json()
    assert detail["detail"]
