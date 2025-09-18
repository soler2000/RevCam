"""FastAPI application wiring together the RevCam services."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from aiortc import RTCSessionDescription

from .camera import (
    BaseCamera,
    CameraError,
    CameraSelection,
    get_available_camera_modes,
    get_camera_status,
    select_camera,
)
from .config import ConfigManager
from .pipeline import FramePipeline
from .video import VideoSource
from .webrtc import WebRTCManager

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_static(name: str) -> str:
    path = STATIC_DIR / name
    if not path.exists():  # pragma: no cover - sanity check
        raise FileNotFoundError(f"Static asset {name!r} missing")
    return path.read_text(encoding="utf-8")


class OfferPayload(BaseModel):
    sdp: str
    type: Literal["offer"]


class OrientationPayload(BaseModel):
    rotation: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False


class CameraUpdatePayload(BaseModel):
    mode: str


def create_app(config_path: Path | str = Path("data/config.json")) -> FastAPI:
    app = FastAPI(title="RevCam", version="0.1.0")

    logger = logging.getLogger(__name__)

    config_manager = ConfigManager(Path(config_path))
    pipeline = FramePipeline(lambda: config_manager.get_orientation())
    video_source = VideoSource(pipeline)
    camera: BaseCamera | None = None
    webrtc_manager: WebRTCManager | None = None
    camera_error: CameraError | None = None
    camera_mode = config_manager.get_camera_mode()
    camera_lock = asyncio.Lock()

    app.state.camera = None
    app.state.webrtc_manager = None
    app.state.camera_error = None
    app.state.camera_status = None
    app.state.video_source = video_source
    app.state.available_camera_modes = get_available_camera_modes()

    async def _shutdown_webrtc() -> None:
        nonlocal webrtc_manager
        if webrtc_manager is not None:
            await webrtc_manager.shutdown()
            webrtc_manager = None
        app.state.webrtc_manager = webrtc_manager

    async def _start_webrtc() -> None:
        nonlocal webrtc_manager
        await _shutdown_webrtc()
        webrtc_manager = WebRTCManager(video_source=video_source)
        app.state.webrtc_manager = webrtc_manager

    def _update_state(selection) -> None:
        status = get_camera_status()
        if camera_error is not None:
            status = {**status, "error": str(camera_error)}
        app.state.camera = camera
        app.state.camera_error = camera_error
        app.state.camera_status = status
        app.state.camera_mode = camera_mode

    async def _apply_camera_mode(mode: str) -> CameraSelection:
        nonlocal camera, camera_error, camera_mode

        selection = select_camera(mode)
        if selection.error is not None and selection.camera is None and camera is not None:
            await video_source.set_camera(None)
            camera = None
            await _shutdown_webrtc()
            selection = select_camera(mode)

        if selection.camera is not None:
            await video_source.set_camera(selection.camera)
            camera = selection.camera
            camera_error = None
            await _start_webrtc()
        else:
            camera = None
            camera_error = selection.error
            await video_source.set_camera(None)
            await _shutdown_webrtc()

        camera_mode = mode
        _update_state(selection)
        return selection

    @app.on_event("startup")
    async def startup() -> None:  # pragma: no cover - framework hook
        nonlocal camera_mode
        await video_source.start()
        async with camera_lock:
            selection = await _apply_camera_mode(camera_mode)
        if selection.error is not None:
            logger.error("Failed to initialise camera: %s", selection.error)

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - framework hook
        await _shutdown_webrtc()
        await video_source.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _load_static("index.html")

    @app.get("/settings", response_class=HTMLResponse)
    async def settings() -> str:
        return _load_static("settings.html")

    @app.get("/api/orientation")
    async def get_orientation() -> dict[str, int | bool]:
        orientation = config_manager.get_orientation()
        return {
            "rotation": orientation.rotation,
            "flip_horizontal": orientation.flip_horizontal,
            "flip_vertical": orientation.flip_vertical,
        }

    @app.post("/api/orientation")
    async def update_orientation(payload: OrientationPayload) -> dict[str, int | bool]:
        try:
            orientation = config_manager.set_orientation(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "rotation": orientation.rotation,
            "flip_horizontal": orientation.flip_horizontal,
            "flip_vertical": orientation.flip_vertical,
        }

    @app.post("/api/offer")
    async def webrtc_offer(payload: OfferPayload) -> dict[str, str]:
        if camera_error is not None:
            raise HTTPException(status_code=503, detail=f"Camera unavailable: {camera_error}")
        if webrtc_manager is None:
            raise HTTPException(status_code=503, detail="WebRTC service unavailable")
        offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
        answer = await webrtc_manager.handle_offer(offer)
        return {"sdp": answer.sdp, "type": answer.type}

    @app.get("/api/camera")
    async def camera_status_endpoint() -> dict[str, str | list[str] | None]:
        status = get_camera_status()
        if camera_error is not None:
            status = {**status, "error": str(camera_error)}
        return {
            **status,
            "modes": get_available_camera_modes(),
            "mode": camera_mode,
        }

    @app.post("/api/camera")
    async def update_camera(payload: CameraUpdatePayload) -> dict[str, str | list[str] | None]:
        requested = (payload.mode or "").strip().lower() or "auto"
        if requested not in get_available_camera_modes() and requested != "picamera":
            raise HTTPException(status_code=400, detail=f"Unknown camera mode: {payload.mode!r}")
        canonical = "picamera2" if requested == "picamera" else requested

        async with camera_lock:
            persisted = config_manager.set_camera_mode(canonical)
            selection = await _apply_camera_mode(persisted)
            status = get_camera_status()
            if camera_error is not None:
                status = {**status, "error": str(camera_error)}
            status = {
                **status,
                "modes": get_available_camera_modes(),
                "mode": persisted,
            }

        if selection.error is not None:
            raise HTTPException(status_code=503, detail=str(selection.error))
        return status

    return app


__all__ = ["create_app"]
