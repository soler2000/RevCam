"""Tests for application startup behaviour."""

from __future__ import annotations

import asyncio

import pytest

from fastapi import HTTPException

from rev_cam import app as app_module
from rev_cam import camera as camera_module
from rev_cam.camera import CameraError


@pytest.fixture(autouse=True)
def reset_camera_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure camera selection cache does not leak between tests."""

    monkeypatch.setattr(camera_module, "_LAST_SELECTION", None)


def test_app_startup_survives_camera_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API should start even when the camera fails to initialise."""

    def raise_camera_error() -> None:
        raise CameraError("camera failed to initialise")

    monkeypatch.setattr(app_module, "create_camera", raise_camera_error)

    application = app_module.create_app()

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

        offer_route = next(route for route in application.router.routes if getattr(route, "path", None) == "/api/offer")

        with pytest.raises(HTTPException) as excinfo:
            loop.run_until_complete(offer_route.endpoint(app_module.OfferPayload(sdp="", type="offer")))

        assert excinfo.value.status_code == 503
        assert "camera failed to initialise" in excinfo.value.detail

        camera_route = next(route for route in application.router.routes if getattr(route, "path", None) == "/api/camera")
        camera_payload = loop.run_until_complete(camera_route.endpoint())
        assert camera_payload["error"] == "camera failed to initialise"
        assert camera_payload["requested"] == "auto"
        assert camera_payload["active_backend"] is None

        for handler in application.router.on_shutdown:
            loop.run_until_complete(handler())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_camera_status_endpoint_reports_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the fallback chain when the synthetic feed is in use."""

    def broken_picamera() -> None:
        raise CameraError("picamera offline")

    def broken_opencv(*_: object, **__: object) -> None:
        raise CameraError("opencv offline")

    monkeypatch.setattr(camera_module, "Picamera2Camera", broken_picamera)
    monkeypatch.setattr(camera_module, "OpenCVCamera", broken_opencv)
    monkeypatch.delenv("REVCAM_CAMERA", raising=False)

    application = app_module.create_app()

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
            "Picamera2: picamera offline",
            "OpenCV: opencv offline",
        ]

        camera_route = next(route for route in application.router.routes if getattr(route, "path", None) == "/api/camera")
        payload = loop.run_until_complete(camera_route.endpoint())
        assert payload == status

        for handler in application.router.on_shutdown:
            loop.run_until_complete(handler())
    finally:
        asyncio.set_event_loop(None)
        loop.close()

