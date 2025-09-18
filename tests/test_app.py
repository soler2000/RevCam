"""Tests for application startup behaviour."""

from __future__ import annotations

import asyncio

import pytest

from fastapi import HTTPException

from rev_cam import app as app_module
from rev_cam.camera import CameraError


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

        offer_route = next(route for route in application.router.routes if getattr(route, "path", None) == "/api/offer")

        with pytest.raises(HTTPException) as excinfo:
            loop.run_until_complete(offer_route.endpoint(app_module.OfferPayload(sdp="", type="offer")))

        assert excinfo.value.status_code == 503
        assert "camera failed to initialise" in excinfo.value.detail

        for handler in application.router.on_shutdown:
            loop.run_until_complete(handler())
    finally:
        asyncio.set_event_loop(None)
        loop.close()

