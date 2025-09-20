"""FastAPI application wiring together the RevCam services."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .battery import BatteryMonitor
from .camera import CAMERA_SOURCES, BaseCamera, CameraError, create_camera, identify_camera
from .config import ConfigManager, Resolution, RESOLUTION_PRESETS
from .pipeline import FramePipeline
from .streaming import MJPEGStreamer
from .version import APP_VERSION
from .wifi import WiFiError, WiFiManager

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
    resolution: str | None = None


class WiFiConnectPayload(BaseModel):
    ssid: str
    password: str | None = None
    development_mode: bool = False
    rollback_seconds: float | None = None


class WiFiHotspotPayload(BaseModel):
    enabled: bool
    ssid: str | None = None
    password: str | None = None
    development_mode: bool = False
    rollback_seconds: float | None = None


def create_app(
    config_path: Path | str = Path("data/config.json"),
    *,
    wifi_manager: WiFiManager | None = None,
) -> FastAPI:
    app = FastAPI(title="RevCam", version=APP_VERSION)

    logger = logging.getLogger(__name__)

    config_manager = ConfigManager(Path(config_path))

    i2c_bus_env = os.getenv("REVCAM_I2C_BUS")
    i2c_bus_override: int | None
    if i2c_bus_env:
        try:
            i2c_bus_override = int(i2c_bus_env, 0)
        except ValueError:
            logger.warning("Invalid REVCAM_I2C_BUS value %r; ignoring", i2c_bus_env)
            i2c_bus_override = None
    else:
        i2c_bus_override = None

    battery_monitor = BatteryMonitor(capacity_mah=1000, i2c_bus=i2c_bus_override)
    wifi_manager = wifi_manager or WiFiManager()

    pipeline = FramePipeline(lambda: config_manager.get_orientation())
    camera: BaseCamera | None = None
    streamer: MJPEGStreamer | None = None
    stream_error: str | None = None
    active_camera_choice: str = "unknown"
    active_resolution: Resolution = config_manager.get_resolution()
    camera_errors: dict[str, str] = {}

    app.state.wifi_manager = wifi_manager

    def _record_camera_error(source: str, message: str | None) -> None:
        if message:
            camera_errors[source] = message
        else:
            camera_errors.pop(source, None)

    def _build_camera(
        selection: str,
        resolution: Resolution,
        *,
        fallback_to_synthetic: bool,
    ) -> tuple[BaseCamera, str]:
        normalised = selection.strip().lower()
        resolution_tuple = resolution.as_tuple()
        if normalised == "auto":
            try:
                camera_instance = create_camera("picamera", resolution=resolution_tuple)
            except CameraError as exc:
                reason = str(exc)
                logger.warning("Picamera unavailable, using synthetic camera: %s", reason)
                _record_camera_error("picamera", reason)
                fallback_camera, fallback_active = _build_camera(
                    "synthetic",
                    resolution,
                    fallback_to_synthetic=False,
                )
                return fallback_camera, fallback_active
            else:
                _record_camera_error("picamera", None)
                return camera_instance, identify_camera(camera_instance)

        try:
            camera_instance = create_camera(normalised, resolution=resolution_tuple)
        except CameraError as exc:
            reason = str(exc)
            _record_camera_error(normalised, reason)
            if fallback_to_synthetic and normalised != "synthetic":
                logger.warning(
                    "Failed to initialise camera '%s': %s; falling back to synthetic", selection, reason
                )
                fallback_camera, fallback_active = _build_camera(
                    "synthetic",
                    resolution,
                    fallback_to_synthetic=False,
                )
                return fallback_camera, fallback_active
            raise
        else:
            _record_camera_error(normalised, None)
            return camera_instance, identify_camera(camera_instance)

    @app.on_event("startup")
    async def startup() -> None:  # pragma: no cover - framework hook
        nonlocal camera, active_camera_choice, streamer, active_resolution, stream_error
        selection = config_manager.get_camera()
        resolution = config_manager.get_resolution()
        camera, active_camera_choice = _build_camera(
            selection,
            resolution,
            fallback_to_synthetic=True,
        )
        active_resolution = resolution
        try:
            streamer = MJPEGStreamer(camera=camera, pipeline=pipeline)
        except RuntimeError as exc:
            logger.error("Failed to initialise MJPEG streamer: %s", exc)
            stream_error = str(exc)
            streamer = None
        else:
            stream_error = None

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - framework hook
        nonlocal streamer
        if streamer is not None:
            await streamer.aclose()
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
        current_resolution = config_manager.get_resolution()
        resolution_options = [
            {
                "value": key,
                "label": f"{preset.width}×{preset.height}",
            }
            for key, preset in RESOLUTION_PRESETS.items()
        ]
        stream_info = {
            "enabled": streamer is not None,
            "endpoint": "/stream/mjpeg" if streamer else None,
            "content_type": streamer.media_type if streamer else None,
            "error": stream_error,
        }
        return {
            "selected": config_manager.get_camera(),
            "active": active_camera_choice,
            "options": options,
            "errors": errors,
            "version": APP_VERSION,
            "stream": stream_info,
            "resolution": {
                "selected": current_resolution.key(),
                "active": active_resolution.key(),
                "options": resolution_options,
            },
        }

    @app.get("/api/battery")
    async def get_battery_status() -> dict[str, object | None]:
        reading = battery_monitor.read()
        return reading.to_dict()

    @app.get("/api/wifi/status")
    async def get_wifi_status() -> dict[str, object | None]:
        try:
            status = await run_in_threadpool(wifi_manager.get_status)
        except WiFiError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return status.to_dict()

    @app.get("/api/wifi/networks")
    async def list_wifi_networks() -> dict[str, object]:
        try:
            networks = await run_in_threadpool(wifi_manager.scan_networks)
        except WiFiError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"networks": [network.to_dict() for network in networks]}

    @app.post("/api/wifi/connect")
    async def connect_wifi(payload: WiFiConnectPayload) -> dict[str, object | None]:
        try:
            status = await run_in_threadpool(
                wifi_manager.connect,
                payload.ssid,
                payload.password,
                development_mode=payload.development_mode,
                rollback_timeout=payload.rollback_seconds,
            )
        except WiFiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return status.to_dict()

    @app.post("/api/wifi/hotspot")
    async def configure_wifi_hotspot(payload: WiFiHotspotPayload) -> dict[str, object | None]:
        if payload.enabled:
            try:
                status = await run_in_threadpool(
                    wifi_manager.enable_hotspot,
                    payload.ssid or "",
                    payload.password,
                    development_mode=payload.development_mode,
                    rollback_timeout=payload.rollback_seconds,
                )
            except WiFiError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            try:
                status = await run_in_threadpool(wifi_manager.disable_hotspot)
            except WiFiError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return status.to_dict()

    @app.post("/api/camera")
    async def update_camera(payload: CameraPayload) -> dict[str, object]:
        nonlocal camera, active_camera_choice, streamer, active_resolution, stream_error
        selection = payload.source.strip().lower()
        if selection not in CAMERA_SOURCES:
            raise HTTPException(status_code=400, detail="Unknown camera source")
        try:
            requested_resolution = config_manager.parse_resolution(payload.resolution)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_selection = config_manager.get_camera()
        current_resolution = config_manager.get_resolution()
        if (
            camera is not None
            and selection == current_selection
            and requested_resolution == current_resolution
        ):
            stream_info = {
                "enabled": streamer is not None,
                "endpoint": "/stream/mjpeg" if streamer else None,
                "content_type": streamer.media_type if streamer else None,
                "error": stream_error,
            }
            return {
                "selected": selection,
                "active": active_camera_choice,
                "version": APP_VERSION,
                "stream": stream_info,
                "resolution": {
                    "selected": current_resolution.key(),
                    "active": active_resolution.key(),
                },
            }

        old_selection = current_selection
        old_resolution = current_resolution
        old_camera = camera
        camera = None
        if old_camera is not None:
            try:
                await old_camera.close()
            except Exception:  # pragma: no cover - defensive logging only
                logger.exception("Failed to close previous camera instance")

        try:
            new_camera, active_camera_choice = _build_camera(
                selection,
                requested_resolution,
                fallback_to_synthetic=(selection == "auto"),
            )
        except CameraError as exc:
            # Attempt to restore the previous camera configuration on failure.
            try:
                restored_camera, restored_active = _build_camera(
                    old_selection,
                    old_resolution,
                    fallback_to_synthetic=True,
                )
            except Exception:  # pragma: no cover - best-effort recovery
                camera = None
                active_camera_choice = "unknown"
            else:
                camera = restored_camera
                active_camera_choice = restored_active
                active_resolution = old_resolution
                if streamer is None and camera is not None:
                    try:
                        streamer = MJPEGStreamer(camera=camera, pipeline=pipeline)
                    except RuntimeError as streamer_exc:
                        logger.error("Failed to initialise MJPEG streamer: %s", streamer_exc)
                        stream_error = str(streamer_exc)
                        streamer = None
                    else:
                        stream_error = None
                elif streamer is not None and camera is not None:
                    streamer.camera = camera
                    stream_error = None
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        camera = new_camera
        active_resolution = requested_resolution
        config_manager.set_camera(selection)
        config_manager.set_resolution(requested_resolution)
        if streamer is None:
            try:
                streamer = MJPEGStreamer(camera=camera, pipeline=pipeline)
            except RuntimeError as exc:
                logger.error("Failed to initialise MJPEG streamer: %s", exc)
                stream_error = str(exc)
                streamer = None
            else:
                stream_error = None
        else:
            streamer.camera = camera
            stream_error = None
        stream_info = {
            "enabled": streamer is not None,
            "endpoint": "/stream/mjpeg" if streamer else None,
            "content_type": streamer.media_type if streamer else None,
            "error": stream_error,
        }
        return {
            "selected": selection,
            "active": active_camera_choice,
            "version": APP_VERSION,
            "stream": stream_info,
            "resolution": {
                "selected": requested_resolution.key(),
                "active": active_resolution.key(),
            },
        }

    @app.get("/stream/mjpeg")
    async def stream_mjpeg():
        if streamer is None:
            raise HTTPException(
                status_code=503,
                detail=stream_error or "Streaming service unavailable",
            )
        response = StreamingResponse(streamer.stream(), media_type=streamer.media_type)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    return app


__all__ = ["create_app"]
