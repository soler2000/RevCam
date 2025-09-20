"""FastAPI application wiring together the RevCam services."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .camera import CAMERA_SOURCES, BaseCamera, CameraError, create_camera, identify_camera
from .config import ConfigManager
from .pipeline import FramePipeline
from .streaming import MJPEGStreamer
from .version import APP_VERSION

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_static(name: str) -> str:
    path = STATIC_DIR / name
    if not path.exists():  # pragma: no cover - sanity check
        raise FileNotFoundError(f"Static asset {name!r} missing")
    return path.read_text(encoding="utf-8")


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
    streamer: MJPEGStreamer | None = None
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

    @app.on_event("startup")
    async def startup() -> None:  # pragma: no cover - framework hook
        nonlocal camera, active_camera_choice, streamer
        selection = config_manager.get_camera()
        camera, active_camera_choice = _build_camera(selection, fallback_to_synthetic=True)
        streamer = MJPEGStreamer(camera=camera, pipeline=pipeline)

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - framework hook
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
        stream_info = {
            "enabled": streamer is not None,
            "endpoint": "/stream/mjpeg" if streamer else None,
            "content_type": streamer.media_type if streamer else None,
            "error": None,
        }
        return {
            "selected": config_manager.get_camera(),
            "active": active_camera_choice,
            "options": options,
            "errors": errors,
            "version": APP_VERSION,
            "stream": stream_info,
        }

    @app.post("/api/camera")
    async def update_camera(payload: CameraPayload) -> dict[str, object]:
        nonlocal camera, active_camera_choice, streamer
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
        if streamer is None:
            streamer = MJPEGStreamer(camera=camera, pipeline=pipeline)
        else:
            streamer.camera = camera
        if old_camera is not None:
            await old_camera.close()
        stream_info = {
            "enabled": streamer is not None,
            "endpoint": "/stream/mjpeg" if streamer else None,
            "content_type": streamer.media_type if streamer else None,
            "error": None,
        }
        return {
            "selected": selection,
            "active": active_camera_choice,
            "version": APP_VERSION,
            "stream": stream_info,
        }

    @app.get("/stream/mjpeg")
    async def stream_mjpeg():
        if streamer is None:
            raise HTTPException(status_code=503, detail="Streaming service unavailable")
        response = StreamingResponse(streamer.stream(), media_type=streamer.media_type)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    return app


__all__ = ["create_app"]
