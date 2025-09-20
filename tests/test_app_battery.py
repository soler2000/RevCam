"""Integration tests for the battery status API."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam import app as app_module
from rev_cam.battery import BatteryReading
from rev_cam.config import ConfigManager


class _StubSupervisor:
    def __init__(self, *args, **kwargs) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    async def aclose(self) -> None:
        self.started = False


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monitor_holder: dict[str, object] = {}

    class _StubMonitor:
        def __init__(self, capacity_mah: int = 1000, **kwargs) -> None:
            self.capacity_mah = capacity_mah
            monitor_holder["instance"] = self

        def read(self) -> BatteryReading:
            return BatteryReading(
                available=True,
                percentage=64.3,
                voltage=3.92,
                current_ma=-120.0,
                charging=False,
                capacity_mah=self.capacity_mah,
                error=None,
            )

    monkeypatch.setattr(app_module, "BatteryMonitor", lambda *args, **kwargs: _StubMonitor(*args, **kwargs))
    monkeypatch.setattr(app_module, "BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr(app_module, "create_battery_overlay", lambda *args, **kwargs: (lambda frame: frame))
    app = app_module.create_app(tmp_path / "config.json")
    with TestClient(app) as test_client:
        monitor_instance = monitor_holder.get("instance")
        assert monitor_instance is not None
        test_client.monitor = monitor_instance  # type: ignore[attr-defined]
        yield test_client


def test_battery_endpoint_returns_reading(client: TestClient) -> None:
    response = client.get("/api/battery")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["percentage"] == pytest.approx(64.3)
    assert payload["voltage"] == pytest.approx(3.92)
    assert payload["current_ma"] == pytest.approx(-120.0)
    assert payload["charging"] is False
    assert payload["capacity_mah"] == 1000
    assert payload["error"] is None


def test_battery_monitor_uses_configured_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.json"
    manager = ConfigManager(config_file)
    manager.set_battery_capacity(2150)

    captured: dict[str, int] = {}

    class _StubMonitor:
        def __init__(self, capacity_mah: int = 1000, **kwargs) -> None:
            self.capacity_mah = capacity_mah
            captured["capacity"] = capacity_mah

        def read(self) -> BatteryReading:
            return BatteryReading(
                available=True,
                percentage=50.0,
                voltage=3.8,
                current_ma=0.0,
                charging=False,
                capacity_mah=self.capacity_mah,
                error=None,
            )

    monkeypatch.setattr(app_module, "BatteryMonitor", lambda *args, **kwargs: _StubMonitor(*args, **kwargs))
    monkeypatch.setattr(app_module, "BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr(app_module, "create_battery_overlay", lambda *args, **kwargs: (lambda frame: frame))

    app_module.create_app(config_file)

    assert captured["capacity"] == 2150


def test_battery_limits_endpoint_returns_config(client: TestClient) -> None:
    response = client.get("/api/battery/limits")
    assert response.status_code == 200
    payload = response.json()
    assert payload["warning_percent"] == pytest.approx(20.0)
    assert payload["shutdown_percent"] == pytest.approx(5.0)
    assert payload["capacity_mah"] == 1000


def test_battery_limits_endpoint_accepts_updates(client: TestClient) -> None:
    response = client.post(
        "/api/battery/limits",
        json={"warning_percent": 35.0, "shutdown_percent": 10.0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["warning_percent"] == pytest.approx(35.0)
    assert payload["shutdown_percent"] == pytest.approx(10.0)
    assert payload["capacity_mah"] == 1000

    confirm = client.get("/api/battery/limits")
    assert confirm.status_code == 200
    assert confirm.json() == payload


def test_battery_capacity_endpoint_updates_monitor(client: TestClient) -> None:
    monitor = getattr(client, "monitor", None)
    assert monitor is not None

    response = client.post("/api/battery/capacity", json={"capacity_mah": 1800})
    assert response.status_code == 200
    payload = response.json()
    assert payload["capacity_mah"] == 1800
    assert getattr(monitor, "capacity_mah") == 1800

    confirm = client.get("/api/battery/limits")
    assert confirm.status_code == 200
    assert confirm.json()["capacity_mah"] == 1800

    status = client.get("/api/battery")
    assert status.status_code == 200
    assert status.json()["capacity_mah"] == 1800


def test_battery_endpoint_surfaces_unavailable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _OfflineMonitor:
        def read(self) -> BatteryReading:
            return BatteryReading(
                available=False,
                percentage=None,
                voltage=None,
                current_ma=None,
                charging=None,
                capacity_mah=1000,
                error="sensor offline",
            )

    monkeypatch.setattr(app_module, "BatteryMonitor", lambda *args, **kwargs: _OfflineMonitor())
    monkeypatch.setattr(app_module, "BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr(app_module, "create_battery_overlay", lambda *args, **kwargs: (lambda frame: frame))
    app = app_module.create_app(tmp_path / "config.json")
    with TestClient(app) as test_client:
        response = test_client.get("/api/battery")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["percentage"] is None
    assert payload["voltage"] is None
    assert payload["current_ma"] is None
    assert payload["charging"] is None
    assert payload["capacity_mah"] == 1000
    assert payload["error"] == "sensor offline"
