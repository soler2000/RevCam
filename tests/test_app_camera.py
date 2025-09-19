"""Tests covering camera switching behaviour in the FastAPI app."""

from __future__ import annotations

import asyncio
import numpy as np
import pytest
from fastapi import HTTPException

from rev_cam.app import create_app
from rev_cam.camera import BaseCamera, CameraError


def test_switch_camera_shuts_down_streams_before_closing_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching cameras should close WebRTC peers before the old device."""

    events: list[tuple[str, str]] = []
    created: list[BaseCamera] = []

    class DummyCamera(BaseCamera):
        def __init__(self, name: str) -> None:
            self.name = name

        async def get_frame(self) -> np.ndarray:  # pragma: no cover - required by abstract base
            return np.zeros((1, 1, 3), dtype=np.uint8)

        async def close(self) -> None:
            events.append(("close", self.name))

    def fake_create_camera(choice: str | None = None) -> DummyCamera:
        label = (choice or "default").lower()
        camera = DummyCamera(f"{label}-{len(created)}")
        created.append(camera)
        return camera

    manager_instances: list["DummyWebRTCManager"] = []

    class DummyWebRTCManager:
        def __init__(self, camera: DummyCamera, pipeline: object) -> None:  # pragma: no cover - simple stub
            self.camera = camera
            self.pipeline = pipeline
            manager_instances.append(self)
            events.append(("manager_init", camera.name))

        async def shutdown(self) -> None:  # pragma: no cover - simple stub
            events.append(("shutdown", self.camera.name))

    monkeypatch.setattr("rev_cam.app.create_camera", fake_create_camera)
    monkeypatch.setattr("rev_cam.app.WebRTCManager", DummyWebRTCManager)

    app = create_app()
    route = next(r for r in app.router.routes if getattr(r, "path", None) == "/api/camera")
    payload_model = route.dependant.body_params[0].type_

    async def scenario() -> None:
        await app.router.startup()
        try:
            payload = payload_model(backend="synthetic")
            result = await route.endpoint(payload)
            assert result == {"backend": "synthetic", "camera": "DummyCamera"}

            snapshot = list(events)
            assert ("manager_init", "default-0") in snapshot
            assert ("manager_init", "synthetic-1") in snapshot
            assert ("shutdown", "default-0") in snapshot
            assert ("close", "default-0") in snapshot
            assert snapshot.index(("shutdown", "default-0")) < snapshot.index(("close", "default-0"))
            assert len(manager_instances) >= 2
            assert manager_instances[-1].camera.name == "synthetic-1"
        finally:
            await app.router.shutdown()

    asyncio.run(scenario())


def test_switch_camera_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid backend should not disturb the active camera or peers."""

    events: list[str] = []

    class DummyCamera(BaseCamera):
        async def get_frame(self) -> np.ndarray:  # pragma: no cover - required by abstract base
            return np.zeros((1, 1, 3), dtype=np.uint8)

        async def close(self) -> None:
            events.append("close")

    def fake_create_camera(choice: str | None = None) -> DummyCamera:
        if choice and choice != "default":
            raise CameraError(f"Unknown camera backend: {choice}")
        return DummyCamera()

    class DummyWebRTCManager:
        def __init__(self, camera: BaseCamera, pipeline: object) -> None:  # pragma: no cover - simple stub
            self.camera = camera

        async def shutdown(self) -> None:  # pragma: no cover - simple stub
            events.append("shutdown")

    monkeypatch.setattr("rev_cam.app.create_camera", fake_create_camera)
    monkeypatch.setattr("rev_cam.app.WebRTCManager", DummyWebRTCManager)

    app = create_app()
    route = next(r for r in app.router.routes if getattr(r, "path", None) == "/api/camera")
    payload_model = route.dependant.body_params[0].type_

    async def scenario() -> None:
        await app.router.startup()
        try:
            payload = payload_model(backend="invalid")
            with pytest.raises(HTTPException) as excinfo:
                await route.endpoint(payload)
            assert excinfo.value.status_code == 400
            assert excinfo.value.detail.startswith("Unknown camera backend")
            assert "shutdown" not in events
            assert "close" not in events
        finally:
            await app.router.shutdown()

    asyncio.run(scenario())

