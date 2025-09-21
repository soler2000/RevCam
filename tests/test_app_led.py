import pytest

pytest.importorskip("httpx")

from pathlib import Path

from fastapi.testclient import TestClient

from rev_cam.app import create_app

from tests.test_app_camera import _RecorderStreamer, _apply_common_stubs


@pytest.fixture
def led_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _apply_common_stubs(monkeypatch)
    _RecorderStreamer.instances.clear()
    monkeypatch.setattr("rev_cam.app.MJPEGStreamer", _RecorderStreamer)
    config_path = tmp_path / "config.json"
    app = create_app(config_path)
    with TestClient(app) as client:
        yield client


def test_led_status_endpoint_reports_state(led_client: TestClient) -> None:
    response = led_client.get("/api/led")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("patterns"), list)
    patterns = set(payload["patterns"])
    for expected in {"off", "boot", "ready", "error"}:
        assert expected in patterns
    assert payload["pattern"] in patterns
    assert payload["active_pattern"] in patterns
    assert payload["error"] is False
    assert isinstance(payload["available"], bool)


def test_led_update_pattern_and_error(led_client: TestClient) -> None:
    response = led_client.post("/api/led", json={"pattern": "off"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["pattern"] == "off"
    assert payload["error"] is False

    toggled = led_client.post("/api/led", json={"error": True})
    assert toggled.status_code == 200
    updated = toggled.json()
    assert updated["error"] is True
    assert updated["pattern"] in updated["patterns"]


def test_led_update_requires_changes(led_client: TestClient) -> None:
    response = led_client.post("/api/led", json={})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "pattern" in detail.lower() or "led" in detail.lower()


def test_led_update_rejects_unknown_pattern(led_client: TestClient) -> None:
    response = led_client.post("/api/led", json={"pattern": "unknown"})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "unknown" in detail.lower()
