"""FastAPI application wiring together the RevCam services."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .camera import CAMERA_SOURCES, BaseCamera, CameraError, create_camera, identify_camera
from .config import ConfigManager
from .pipeline import FramePipeline
from .webrtc import WebRTCManager
from .version import APP_VERSION

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from aiortc import RTCSessionDescription

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


class CameraPayload(BaseModel):
    source: str


def create_app(config_path: Path | str = Path("data/config.json")) -> FastAPI:
    app = FastAPI(title="RevCam", version=APP_VERSION)

    config_manager = ConfigManager(Path(config_path))
    pipeline = FramePipeline(lambda: config_manager.get_orientation())
    camera: BaseCamera | None = None
    webrtc_manager: WebRTCManager | None = None
    webrtc_error: str | None = None
    active_camera_choice: str = "unknown"
    camera_errors: dict[str, str] = {}
    logger = logging.getLogger(__name__)

    def _record_camera_error(source: str, message: str | None) -> None:
        if message:
            camera_errors[source] = message
        else:
            camera_errors.pop(source, None)

    def _build_camera(selection: str, *, fallback_to_synthetic: bool) -> tuple[BaseCamera, str]:
        normalised = selection.strip().lower()
        if normalised == "auto":
            try:
                camera_instance = create_camera("picamera")
            except CameraError as exc:
                reason = str(exc)
                logger.warning("Picamera unavailable, using synthetic camera: %s", reason)
                _record_camera_error("picamera", reason)
                fallback_camera, fallback_active = _build_camera("synthetic", fallback_to_synthetic=False)
                return fallback_camera, fallback_active
            else:
                _record_camera_error("picamera", None)
                return camera_instance, identify_camera(camera_instance)

        try:
            camera_instance = create_camera(normalised)
        except CameraError as exc:
            reason = str(exc)
            _record_camera_error(normalised, reason)
            if fallback_to_synthetic and normalised != "synthetic":
                logger.warning(
                    "Failed to initialise camera '%s': %s; falling back to synthetic", selection, reason
                )
                fallback_camera, fallback_active = _build_camera("synthetic", fallback_to_synthetic=False)
                return fallback_camera, fallback_active
            raise
        else:
            _record_camera_error(normalised, None)
            return camera_instance, identify_camera(camera_instance)

    def _ensure_webrtc(current_camera: BaseCamera) -> None:
        nonlocal webrtc_manager, webrtc_error
        if webrtc_manager is None:
            try:
                webrtc_manager = WebRTCManager(camera=current_camera, pipeline=pipeline)
            except RuntimeError as exc:
                logger.warning("WebRTC disabled: %s", exc)
                webrtc_manager = None
                webrtc_error = str(exc)
            else:
                webrtc_error = None
        else:
            webrtc_manager.camera = current_camera
            webrtc_error = None

    @app.on_event("startup")
    async def startup() -> None:  # pragma: no cover - framework hook
        nonlocal camera, active_camera_choice
        selection = config_manager.get_camera()
        camera, active_camera_choice = _build_camera(selection, fallback_to_synthetic=True)
        _ensure_webrtc(camera)

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - framework hook
        if webrtc_manager:
            await webrtc_manager.shutdown()
        if camera:
            await camera.close()

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

    @app.get("/api/camera")
    async def get_camera_config() -> dict[str, object]:
        options = [{"value": value, "label": label} for value, label in CAMERA_SOURCES.items()]
        errors = {source: message for source, message in camera_errors.items() if message}
        webrtc_info = {"enabled": webrtc_manager is not None}
        if webrtc_error:
            webrtc_info["error"] = webrtc_error
        return {
            "selected": config_manager.get_camera(),
            "active": active_camera_choice,
            "options": options,
            "errors": errors,
            "version": APP_VERSION,
            "webrtc": webrtc_info,
        }

    @app.post("/api/camera")
    async def update_camera(payload: CameraPayload) -> dict[str, object]:
        nonlocal camera, active_camera_choice
        selection = payload.source.strip().lower()
        if selection not in CAMERA_SOURCES:
            raise HTTPException(status_code=400, detail="Unknown camera source")
        try:
            new_camera, active_camera_choice = _build_camera(
                selection, fallback_to_synthetic=(selection == "auto")
            )
        except CameraError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        old_camera = camera
        camera = new_camera
        config_manager.set_camera(selection)
        _ensure_webrtc(camera)
        if old_camera is not None:
            await old_camera.close()
        webrtc_info = {"enabled": webrtc_manager is not None}
        if webrtc_error:
            webrtc_info["error"] = webrtc_error
        return {
            "selected": selection,
            "active": active_camera_choice,
            "version": APP_VERSION,
            "webrtc": webrtc_info,
        }

    @app.post("/api/offer")
    async def webrtc_offer(payload: OfferPayload) -> dict[str, str]:
        if webrtc_manager is None:
            raise HTTPException(status_code=503, detail="WebRTC service unavailable")
        try:
            from aiortc import RTCSessionDescription
        except ImportError as exc:  # pragma: no cover - mirrors WebRTCManager guard
            raise HTTPException(status_code=503, detail="WebRTC support requires aiortc to be installed") from exc
        offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
        answer = await webrtc_manager.handle_offer(offer)
        return {"sdp": answer.sdp, "type": answer.type}

    return app


__all__ = ["create_app"]
