"""Tests for application startup behaviour."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from fastapi import HTTPException

from rev_cam import app as app_module
from rev_cam import camera as camera_module
from rev_cam.camera import CameraError, CameraSelection


@pytest.fixture(autouse=True)
def reset_camera_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure camera selection cache does not leak between tests."""

    monkeypatch.setattr(camera_module, "_LAST_SELECTION", None)


def test_app_startup_survives_camera_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The API should start even when the camera fails to initialise."""

    error = CameraError("camera failed to initialise")

    def fake_select(choice: str | None = None) -> CameraSelection:
        requested = (choice or "auto").strip().lower() or "auto"
        return CameraSelection(
            requested=requested,
            active_backend=None,
            camera=None,
            fallbacks=[],
            error=error,
        )

    monkeypatch.setattr(app_module, "select_camera", fake_select)

    application = app_module.create_app(tmp_path / "config.json")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        for handler in application.router.on_startup:
            loop.run_until_complete(handler())

        assert application.state.camera is None
        assert isinstance(application.state.camera_error, CameraError)
        status = application.state.camera_status
        assert status is not None
        assert status["error"] == "camera failed to initialise"
        assert status["requested"] == "auto"

        offer_route = next(
            route
            for route in application.router.routes
            if getattr(route, "path", None) == "/api/offer"
        )

        with pytest.raises(HTTPException) as excinfo:
            loop.run_until_complete(offer_route.endpoint(app_module.OfferPayload(sdp="", type="offer")))

        assert excinfo.value.status_code == 503
        assert "camera failed to initialise" in excinfo.value.detail

        camera_route = next(
            route
            for route in application.router.routes
            if getattr(route, "path", None) == "/api/camera" and "GET" in getattr(route, "methods", set())
        )
        camera_payload = loop.run_until_complete(camera_route.endpoint())
        assert camera_payload["error"] == "camera failed to initialise"
        assert camera_payload["requested"] == "auto"
        assert camera_payload["active_backend"] is None
        assert "modes" in camera_payload
        assert camera_payload["mode"] == "auto"

        for handler in application.router.on_shutdown:
            loop.run_until_complete(handler())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_camera_status_endpoint_reports_fallbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Expose the fallback chain when the synthetic feed is in use."""

    def broken_picamera() -> None:
        raise CameraError("picamera2 is not available")

    def broken_opencv(*_: object, **__: object) -> None:
        raise CameraError("opencv offline")

    monkeypatch.setattr(camera_module, "Picamera2Camera", broken_picamera)
    monkeypatch.setattr(camera_module, "OpenCVCamera", broken_opencv)
    monkeypatch.delenv("REVCAM_CAMERA", raising=False)

    application = app_module.create_app(tmp_path / "config.json")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        for handler in application.router.on_startup:
            loop.run_until_complete(handler())

        status = application.state.camera_status
        assert status is not None
        assert status["requested"] == "auto"
        assert status["active_backend"] == "synthetic"
        assert status["error"] is None
        assert status["fallbacks"] == [
            "Picamera2: picamera2 is not available",
            "OpenCV: opencv offline",
        ]

        camera_route = next(
            route
            for route in application.router.routes
            if getattr(route, "path", None) == "/api/camera" and "GET" in getattr(route, "methods", set())
        )
        payload = loop.run_until_complete(camera_route.endpoint())
        expected = {
            **status,
            "modes": app_module.get_available_camera_modes(),
            "mode": "auto",
        }
        assert payload == expected

        for handler in application.router.on_shutdown:
            loop.run_until_complete(handler())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_camera_status_endpoint_reports_picamera_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Expose Picamera2 errors when no fallback camera is available."""

    def broken_picamera() -> None:
        raise CameraError("picamera driver crashed")

    def broken_opencv(*_: object, **__: object) -> None:
        raise CameraError("opencv offline")

    monkeypatch.setattr(camera_module, "Picamera2Camera", broken_picamera)
    monkeypatch.setattr(camera_module, "OpenCVCamera", broken_opencv)
    monkeypatch.delenv("REVCAM_CAMERA", raising=False)

    application = app_module.create_app(tmp_path / "config.json")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        for handler in application.router.on_startup:
            loop.run_until_complete(handler())

        status = application.state.camera_status
        assert status is not None
        assert status["requested"] == "auto"
        assert status["active_backend"] is None
        assert status["error"] == "picamera driver crashed"
        assert status["fallbacks"] == [
            "Picamera2: picamera driver crashed",
            "OpenCV: opencv offline",
        ]

        camera_route = next(
            route
            for route in application.router.routes
            if getattr(route, "path", None) == "/api/camera" and "GET" in getattr(route, "methods", set())
        )
        payload = loop.run_until_complete(camera_route.endpoint())
        assert payload["error"] == "picamera driver crashed"
        assert payload["fallbacks"] == [
            "Picamera2: picamera driver crashed",
            "OpenCV: opencv offline",
        ]
        assert payload["mode"] == "auto"

        for handler in application.router.on_shutdown:
            loop.run_until_complete(handler())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_camera_update_endpoint_switches_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """POST /api/camera should reconfigure the backend when requested."""

    class DummyCamera:
        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        async def get_frame(self) -> np.ndarray:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        async def close(self) -> None:
            self.closed = True

    def fake_select(choice: str | None = None) -> CameraSelection:
        requested = (choice or "auto").strip().lower() or "auto"
        if requested == "synthetic":
            camera = DummyCamera("synthetic")
            return camera_module._remember_selection(
                CameraSelection(
                    requested=requested,
                    active_backend="synthetic",
                    camera=camera,
                    fallbacks=[],
                    error=None,
                )
            )
        if requested == "opencv":
            camera = DummyCamera("opencv")
            return camera_module._remember_selection(
                CameraSelection(
                    requested=requested,
                    active_backend="opencv",
                    camera=camera,
                    fallbacks=[],
                    error=None,
                )
            )
        return camera_module._remember_selection(
            CameraSelection(
                requested=requested,
                active_backend=None,
                camera=None,
                fallbacks=[],
                error=CameraError(f"Unsupported mode: {requested}"),
            )
        )

    monkeypatch.setattr(app_module, "select_camera", fake_select)

    application = app_module.create_app(tmp_path / "config.json")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        for handler in application.router.on_startup:
            loop.run_until_complete(handler())

        camera_update_route = next(
            route
            for route in application.router.routes
            if getattr(route, "path", None) == "/api/camera" and "POST" in getattr(route, "methods", set())
        )

        # Switch to OpenCV mode successfully.
        payload = loop.run_until_complete(
            camera_update_route.endpoint(
                app_module.CameraUpdatePayload(mode="opencv")
            )
        )
        assert payload["active_backend"] == "opencv"
        assert payload["mode"] == "opencv"

        # Requesting an unknown mode should raise a validation error.
        with pytest.raises(HTTPException) as excinfo:
            loop.run_until_complete(
                camera_update_route.endpoint(
                    app_module.CameraUpdatePayload(mode="unknown")
                )
            )
        assert excinfo.value.status_code == 400

        # Requesting a failing mode should surface as a 503.
        with pytest.raises(HTTPException) as excinfo:
            loop.run_until_complete(
                camera_update_route.endpoint(
                    app_module.CameraUpdatePayload(mode="auto")
                )
            )
        assert excinfo.value.status_code == 503

        for handler in application.router.on_shutdown:
            loop.run_until_complete(handler())
    finally:
        asyncio.set_event_loop(None)
        loop.close()

