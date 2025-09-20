"""Integration tests for the distance sensor API."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam import app as app_module
from rev_cam.distance import DistanceReading


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    class _StubMonitor:
        def read(self) -> DistanceReading:
            return DistanceReading(
                available=True,
                distance_m=1.25,
                raw_distance_m=1.25,
                timestamp=0.0,
                error=None,
            )

    monkeypatch.setattr(app_module, "DistanceMonitor", lambda *args, **kwargs: _StubMonitor())
    app = app_module.create_app(tmp_path / "config.json")
    with TestClient(app) as test_client:
        yield test_client


def test_distance_endpoint_returns_reading(client: TestClient) -> None:
    response = client.get("/api/distance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["distance_m"] == pytest.approx(1.25)
    assert payload["raw_distance_m"] == pytest.approx(1.25)
    assert payload["zone"] == "warning"
    assert payload["zones"]["danger"] > 0


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
    assert payload["zone"] == "danger"


def test_distance_zone_update_validates_order(client: TestClient) -> None:
    response = client.post(
        "/api/distance/zones",
        json={"caution": 1.0, "warning": 2.0, "danger": 0.5},
    )
    assert response.status_code == 400
    payload = response.json()
    assert "distance" in payload.get("detail", "").lower()
