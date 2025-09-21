import json
import types
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rev_cam import app as app_module
from rev_cam.distance import DistanceCalibration, DistanceReading


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    class _StubMonitor:
        def __init__(
            self,
            *args,
            calibration: DistanceCalibration | None = None,
            **kwargs,
        ) -> None:
            self._timestamp = 0.0
            self._raw_value = 1.2
            if calibration is None:
                self._calibration = DistanceCalibration()
            else:
                self._calibration = DistanceCalibration(calibration.offset_m, calibration.scale)

        def read(self) -> DistanceReading:
            self._timestamp += 0.1
            return DistanceReading(
                available=True,
                distance_m=self._raw_value,
                raw_distance_m=self._raw_value,
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
        def read(self) -> object:
            return types.SimpleNamespace(to_dict=lambda: {})

    class _StubSupervisor:
        def start(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(app_module, "DistanceMonitor", lambda *args, **kwargs: _StubMonitor(*args, **kwargs))
    monkeypatch.setattr(app_module, "BatteryMonitor", lambda *args, **kwargs: _StubBatteryMonitor())
    monkeypatch.setattr(app_module, "BatterySupervisor", lambda *args, **kwargs: _StubSupervisor())
    monkeypatch.setattr(app_module, "create_battery_overlay", lambda *args, **kwargs: (lambda frame: frame))
    monkeypatch.setattr(app_module, "create_distance_overlay", lambda *args, **kwargs: (lambda frame: frame))
    monkeypatch.setattr(app_module, "create_reversing_aids_overlay", lambda *args, **kwargs: (lambda frame: frame))

    config_file = tmp_path / "config.json"
    app = app_module.create_app(config_file)
    with TestClient(app) as test_client:
        test_client.config_path = config_file
        yield test_client


def test_reversing_aids_defaults(client: TestClient) -> None:
    response = client.get("/api/reversing-aids")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert isinstance(payload.get("left"), list)
    assert isinstance(payload.get("right"), list)
    assert len(payload["left"]) == 3
    assert len(payload["right"]) == 3


def test_reversing_aids_update(client: TestClient) -> None:
    payload = {
        "enabled": False,
        "left": [
            {"start": {"x": 0.58, "y": 0.15}, "end": {"x": 0.38, "y": 0.32}},
            {"start": {"x": 0.48, "y": 0.45}, "end": {"x": 0.28, "y": 0.63}},
            {"start": {"x": 0.4, "y": 0.7}, "end": {"x": 0.2, "y": 0.88}},
        ],
        "right": [
            {"start": {"x": 0.42, "y": 0.15}, "end": {"x": 0.62, "y": 0.32}},
            {"start": {"x": 0.52, "y": 0.45}, "end": {"x": 0.72, "y": 0.63}},
            {"start": {"x": 0.6, "y": 0.7}, "end": {"x": 0.8, "y": 0.88}},
        ],
    }
    response = client.post("/api/reversing-aids", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["left"][0]["start"]["x"] == pytest.approx(payload["left"][0]["start"]["x"])

    config_file = Path(client.config_path)
    data = json.loads(config_file.read_text())
    stored = data["reversing_aids"]
    assert stored["enabled"] is False
    assert stored["left"][0]["start"]["x"] == pytest.approx(payload["left"][0]["start"]["x"])


def test_reversing_aids_validation_error(client: TestClient) -> None:
    invalid_payload = {
        "enabled": True,
        "left": [
            {"start": {"x": -0.1, "y": 0.2}, "end": {"x": 0.3, "y": 0.35}},
            {"start": {"x": 0.4, "y": 0.5}, "end": {"x": 0.25, "y": 0.65}},
            {"start": {"x": 0.35, "y": 0.7}, "end": {"x": 0.2, "y": 0.85}},
        ],
        "right": [
            {"start": {"x": 0.5, "y": 0.2}, "end": {"x": 0.7, "y": 0.35}},
            {"start": {"x": 0.6, "y": 0.5}, "end": {"x": 0.8, "y": 0.65}},
            {"start": {"x": 0.7, "y": 0.7}, "end": {"x": 0.9, "y": 0.85}},
        ],
    }
    response = client.post("/api/reversing-aids", json=invalid_payload)
    assert response.status_code == 400
    detail = response.json().get("detail")
    assert isinstance(detail, str)
    assert "reversing" in detail.lower()

