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
    for expected in {"off", "boot", "ready", "error", "illumination"}:
        assert expected in patterns
    assert payload["pattern"] in patterns
    assert payload["active_pattern"] in patterns
    assert payload["error"] is False
    assert isinstance(payload["available"], bool)
    assert "message" in payload
    illumination = payload.get("illumination", {})
    assert isinstance(illumination, dict)
    assert isinstance(illumination.get("color"), str)
    assert isinstance(illumination.get("intensity"), (int, float))


def test_led_update_pattern_and_error(led_client: TestClient) -> None:
    response = led_client.post("/api/led", json={"pattern": "off"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["pattern"] == "off"
    assert payload["error"] is False
    assert "message" in payload
    assert "illumination" in payload

    toggled = led_client.post("/api/led", json={"error": True})
    assert toggled.status_code == 200
    updated = toggled.json()
    assert updated["error"] is True
    assert updated["pattern"] in updated["patterns"]
    assert "message" in updated
    assert "illumination" in updated


def test_led_update_requires_changes(led_client: TestClient) -> None:
    response = led_client.post("/api/led", json={})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    lowered = detail.lower()
    assert "pattern" in lowered or "led" in lowered or "illumination" in lowered


def test_led_update_rejects_unknown_pattern(led_client: TestClient) -> None:
    response = led_client.post("/api/led", json={"pattern": "unknown"})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "unknown" in detail.lower()


def test_led_update_illumination_settings(led_client: TestClient) -> None:
    response = led_client.post(
        "/api/led",
        json={"illumination_color": "#336699", "illumination_intensity": 42},
    )
    assert response.status_code == 200
    payload = response.json()
    illumination = payload.get("illumination")
    assert isinstance(illumination, dict)
    assert illumination.get("color", "").lower() == "#336699"
    assert illumination.get("intensity") == 42


def test_led_update_rejects_invalid_illumination_colour(led_client: TestClient) -> None:
    response = led_client.post("/api/led", json={"illumination_color": "bad"})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "colour" in detail.lower() or "color" in detail.lower()


def test_led_update_rejects_invalid_illumination_intensity(led_client: TestClient) -> None:
    response = led_client.post("/api/led", json={"illumination_intensity": 400})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "intensity" in detail.lower()
