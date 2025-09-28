"""Integration tests for the distance sensor API."""

from __future__ import annotations

import types
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam import app as app_module
from rev_cam.distance import DistanceCalibration, DistanceReading
from rev_cam.config import DEFAULT_DISTANCE_MOUNTING


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    class _StubMonitor:
        def __init__(
            self,
            *args,
            calibration: DistanceCalibration | None = None,
            **kwargs,
        ) -> None:
            self.raw_value = 1.25
            if calibration is None:
                self._calibration = DistanceCalibration()
            else:
                self._calibration = DistanceCalibration(
                    calibration.offset_m,
                    calibration.scale,
                )
            self._timestamp = 0.0

        def read(self) -> DistanceReading:
            self._timestamp += 0.1
            calibrated = self.raw_value * self._calibration.scale + self._calibration.offset_m
            return DistanceReading(
                available=True,
                distance_m=calibrated,
                raw_distance_m=self.raw_value,
                timestamp=self._timestamp,
                error=None,
            )

        def set_calibration(
            self,
            calibration: DistanceCalibration | None = None,
            *,
            offset_m: float | None = None,
            scale: float | None = None,
        ) -> DistanceCalibration:
            if calibration is not None:
                self._calibration = DistanceCalibration(calibration.offset_m, calibration.scale)
            else:
                current = self._calibration
                self._calibration = DistanceCalibration(
                    offset_m=current.offset_m if offset_m is None else offset_m,
                    scale=current.scale if scale is None else scale,
                )
            return self._calibration

        def get_calibration(self) -> DistanceCalibration:
            return self._calibration

    class _StubBatteryMonitor:
        def read(self):  # pragma: no cover - used indirectly
            return types.SimpleNamespace(to_dict=lambda: {})

    class _StubSupervisor:
        def start(self) -> None:  # pragma: no cover - used indirectly
            return None

        async def aclose(self) -> None:  # pragma: no cover - used indirectly
            return None

    monitor_holder: dict[str, _StubMonitor] = {}

    def _make_monitor(*args, **kwargs) -> _StubMonitor:
        monitor = _StubMonitor(*args, **kwargs)
        monitor_holder["instance"] = monitor
        return monitor

    monkeypatch.setattr(app_module, "DistanceMonitor", _make_monitor)
    monkeypatch.setattr(app_module, "BatteryMonitor", lambda *args, **kwargs: _StubBatteryMonitor())
    monkeypatch.setattr(app_module, "BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr(app_module, "create_battery_overlay", lambda *args, **kwargs: (lambda frame: frame))
    config_file = tmp_path / "config.json"
    app = app_module.create_app(config_file)
    with TestClient(app) as test_client:
        test_client.monitor = monitor_holder.get("instance")
        test_client.config_path = config_file
        yield test_client


def test_distance_endpoint_returns_reading(client: TestClient) -> None:
    response = client.get("/api/distance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["distance_m"] == pytest.approx(1.25)
    assert payload["raw_distance_m"] == pytest.approx(1.25)
    assert payload["zone"] == "warning"
    assert payload["display_distance_m"] == pytest.approx(1.25)
    assert payload["use_projected_distance"] is False
    assert payload["zones"]["danger"] > 0
    assert payload["calibration"]["offset_m"] == pytest.approx(0.0)
    assert payload["calibration"]["scale"] == pytest.approx(1.0)
    assert payload["geometry"]["mount_height_m"] == pytest.approx(
        DEFAULT_DISTANCE_MOUNTING.mount_height_m
    )
    assert payload["geometry"]["mount_angle_deg"] == pytest.approx(
        DEFAULT_DISTANCE_MOUNTING.mount_angle_deg
    )
    expected_projection = (
        DEFAULT_DISTANCE_MOUNTING.mount_height_m
        * math.tan(math.radians(DEFAULT_DISTANCE_MOUNTING.mount_angle_deg))
    )
    assert payload["projected_distance_m"] == pytest.approx(expected_projection)


def test_distance_zone_update_returns_updated_values(client: TestClient) -> None:
    response = client.post(
        "/api/distance/zones",
        json={"caution": 4.0, "warning": 2.5, "danger": 1.0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["zones"]["caution"] == pytest.approx(4.0)
    assert payload["zones"]["warning"] == pytest.approx(2.5)
    assert payload["zones"]["danger"] == pytest.approx(1.0)
    assert payload["display_distance_m"] == pytest.approx(1.25)
    assert payload["zone"] == "warning"
    assert payload["calibration"]["scale"] == pytest.approx(1.0)


def test_distance_zone_update_validates_order(client: TestClient) -> None:
    response = client.post(
        "/api/distance/zones",
        json={"caution": 1.0, "warning": 2.0, "danger": 0.5},
    )
    assert response.status_code == 400
    payload = response.json()
    assert "distance" in payload.get("detail", "").lower()


def test_distance_calibration_update_persists(client: TestClient) -> None:
    monitor = getattr(client, "monitor", None)
    assert monitor is not None
    response = client.post(
        "/api/distance/calibration",
        json={"offset_m": 0.4, "scale": 1.2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["calibration"]["offset_m"] == pytest.approx(0.4)
    assert payload["calibration"]["scale"] == pytest.approx(1.2)
    expected_distance = monitor.raw_value * 1.2 + 0.4
    assert payload["distance_m"] == pytest.approx(expected_distance)
    assert payload["display_distance_m"] == pytest.approx(expected_distance)
    current_calibration = monitor.get_calibration()
    assert current_calibration.offset_m == pytest.approx(0.4)
    assert current_calibration.scale == pytest.approx(1.2)
    calibration_response = client.get("/api/distance/calibration")
    assert calibration_response.status_code == 200
    calibration_payload = calibration_response.json()
    assert calibration_payload["calibration"]["offset_m"] == pytest.approx(0.4)
    assert calibration_payload["calibration"]["scale"] == pytest.approx(1.2)
    config_data = json.loads(Path(client.config_path).read_text())
    calibration = config_data["distance"]["calibration"]
    assert calibration["offset_m"] == pytest.approx(0.4)
    assert calibration["scale"] == pytest.approx(1.2)


def test_distance_calibration_rejects_invalid_input(client: TestClient) -> None:
    response = client.post(
        "/api/distance/calibration",
        json={"offset_m": 6.0, "scale": 1.0},
    )
    assert response.status_code == 400


def test_distance_calibration_zero_endpoint(client: TestClient) -> None:
    monitor = getattr(client, "monitor", None)
    assert monitor is not None
    monitor.raw_value = 0.35
    response = client.post("/api/distance/calibration/zero")
    assert response.status_code == 200
    payload = response.json()
    assert payload["calibration"]["offset_m"] == pytest.approx(-0.35, rel=1e-6)
    assert payload["calibration"]["scale"] == pytest.approx(1.0)
    assert payload["distance_m"] == pytest.approx(0.0, abs=1e-6)
    assert payload["display_distance_m"] == pytest.approx(0.0, abs=1e-6)
    config_data = json.loads(Path(client.config_path).read_text())
    calibration = config_data["distance"]["calibration"]
    assert calibration["offset_m"] == pytest.approx(-0.35, rel=1e-6)
    assert calibration["scale"] == pytest.approx(1.0)


def test_distance_geometry_endpoint_returns_defaults(client: TestClient) -> None:
    response = client.get("/api/distance/geometry")
    assert response.status_code == 200
    payload = response.json()
    assert payload["geometry"]["mount_height_m"] == pytest.approx(
        DEFAULT_DISTANCE_MOUNTING.mount_height_m
    )
    assert payload["geometry"]["mount_angle_deg"] == pytest.approx(
        DEFAULT_DISTANCE_MOUNTING.mount_angle_deg
    )
    expected_projection = (
        DEFAULT_DISTANCE_MOUNTING.mount_height_m
        * math.tan(math.radians(DEFAULT_DISTANCE_MOUNTING.mount_angle_deg))
    )
    assert payload["projected_distance_m"] == pytest.approx(expected_projection)
    assert payload["use_projected_distance"] is False


def test_distance_display_mode_toggle(client: TestClient) -> None:
    response = client.post(
        "/api/distance/display",
        json={"use_projected_distance": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["use_projected_distance"] is True
    expected_projection = (
        DEFAULT_DISTANCE_MOUNTING.mount_height_m
        * math.tan(math.radians(DEFAULT_DISTANCE_MOUNTING.mount_angle_deg))
    )
    assert payload["projected_distance_m"] == pytest.approx(expected_projection)
    assert payload["display_distance_m"] == pytest.approx(expected_projection)
    assert payload["zone"] == "warning"
    config_data = json.loads(Path(client.config_path).read_text())
    assert config_data["distance"]["use_projected_distance"] is True
    follow_up = client.get("/api/distance")
    assert follow_up.status_code == 200
    follow_payload = follow_up.json()
    assert follow_payload["use_projected_distance"] is True
    assert follow_payload["display_distance_m"] == pytest.approx(expected_projection)


def test_distance_geometry_update_persists(client: TestClient) -> None:
    response = client.post(
        "/api/distance/geometry",
        json={"mount_height_m": 1.9, "mount_angle_deg": 32.0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["geometry"]["mount_height_m"] == pytest.approx(1.9)
    assert payload["geometry"]["mount_angle_deg"] == pytest.approx(32.0)
    expected_projection = 1.9 * math.tan(math.radians(32.0))
    assert payload["projected_distance_m"] == pytest.approx(expected_projection)
    config_data = json.loads(Path(client.config_path).read_text())
    geometry = config_data["distance"]["geometry"]
    assert geometry["mount_height_m"] == pytest.approx(1.9)
    assert geometry["mount_angle_deg"] == pytest.approx(32.0)


def test_distance_diagram_image_served(client: TestClient) -> None:
    response = client.get("/images/distance-diagram.svg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in response.text
