"""FastAPI application wiring together the RevCam services."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import string
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .battery import (
    BatteryMonitor,
    BatterySupervisor,
    create_battery_overlay,
    create_wifi_overlay,
)
from .camera import CAMERA_SOURCES, BaseCamera, CameraError, create_camera, identify_camera
from .config import (
    ConfigManager,
    DistanceMounting,
    Resolution,
    RESOLUTION_PRESETS,
    StreamSettings,
)
from .diagnostics import collect_diagnostics
from .distance import DistanceCalibration, DistanceMonitor, create_distance_overlay
from .led_matrix import LedRing
from .reversing_aids import create_reversing_aids_overlay
from .sensor_fusion import Gy85KalmanFilter, SensorSample, Vector3
from .pipeline import FramePipeline
from .streaming import MJPEGStreamer, WebRTCManager, encode_frame_to_jpeg
from .version import APP_VERSION
from .wifi import WiFiCredentialStore, WiFiError, WiFiManager
from .trailer_leveling import evaluate_leveling

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_static(name: str) -> str:
    path = STATIC_DIR / name
    if not path.exists():  # pragma: no cover - sanity check
        raise FileNotFoundError(f"Static asset {name!r} missing")
    return path.read_text(encoding="utf-8")


def _project_ground_distance(
    mounting: DistanceMounting, measured_distance: float | None = None
) -> float | None:
    angle_rad = math.radians(mounting.mount_angle_deg)
    projection: float
    if measured_distance is not None and math.isfinite(measured_distance):
        projection = float(measured_distance) * math.sin(angle_rad)
    else:
        projection = mounting.mount_height_m * math.tan(angle_rad)
    return projection if math.isfinite(projection) else None


def _select_display_distance(
    measured: float | None, projected: float | None, use_projected: bool
) -> float | None:
    candidate: float | None
    if use_projected and projected is not None and math.isfinite(projected):
        candidate = float(projected)
    elif measured is not None and math.isfinite(measured):
        candidate = float(measured)
    else:
        candidate = None
    return candidate


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


class WiFiForgetPayload(BaseModel):
    identifier: str


class DistanceZonesPayload(BaseModel):
    caution: float
    warning: float
    danger: float


class DistanceCalibrationPayload(BaseModel):
    offset_m: float
    scale: float


class DistanceGeometryPayload(BaseModel):
    mount_height_m: float
    mount_angle_deg: float


class DistanceOverlayPayload(BaseModel):
    enabled: bool


class DistanceDisplayModePayload(BaseModel):
    use_projected_distance: bool


class BatteryLimitsPayload(BaseModel):
    warning_percent: float
    shutdown_percent: float


class BatteryCapacityPayload(BaseModel):
    capacity_mah: int


class StreamSettingsPayload(BaseModel):
    fps: int | None = None
    jpeg_quality: int | None = None


class WebRTCOfferPayload(BaseModel):
    sdp: str
    type: str


class WebRTCErrorReportPayload(BaseModel):
    """Client-side WebRTC failure report."""

    name: str | None = Field(default=None, max_length=128)
    message: str | None = Field(default=None, max_length=1024)
    stack: str | None = Field(default=None, max_length=4096)


class ReversingAidPointPayload(BaseModel):
    x: float
    y: float


class ReversingAidSegmentPayload(BaseModel):
    start: ReversingAidPointPayload
    end: ReversingAidPointPayload


class ReversingAidsPayload(BaseModel):
    enabled: bool | None = None
    left: list[ReversingAidSegmentPayload] | None = None
    right: list[ReversingAidSegmentPayload] | None = None


class OverlaySettingsPayload(BaseModel):
    master_enabled: bool | None = Field(default=None)
    battery_enabled: bool | None = Field(default=None)
    wifi_enabled: bool | None = Field(default=None)
    distance_enabled: bool | None = Field(default=None)
    reversing_aids_enabled: bool | None = Field(default=None)


class LedSettingsPayload(BaseModel):
    pattern: str | None = None
    error: bool | None = None
    illumination_color: str | None = None
    illumination_intensity: float | int | None = None


class VectorPayload(BaseModel):
    x: float
    y: float
    z: float


class LevelingGeometryPayload(BaseModel):
    axle_width_m: float | None = None
    hitch_to_axle_m: float | None = None
    length_m: float | None = None


class LevelingRampPayload(BaseModel):
    length_m: float | None = None
    height_m: float | None = None


class LevelingConfigPayload(BaseModel):
    geometry: LevelingGeometryPayload | None = None
    ramp: LevelingRampPayload | None = None


class LevelingSamplePayload(BaseModel):
    accelerometer: VectorPayload
    gyroscope: VectorPayload
    magnetometer: VectorPayload
    dt: float | None = Field(default=None, gt=0.0)
    mode: Literal["hitched", "unhitched"] = "hitched"


def create_app(
    config_path: Path | str = Path("data/config.json"),
    *,
    wifi_manager: WiFiManager | None = None,
) -> FastAPI:
    app = FastAPI(title="RevCam", version=APP_VERSION)

    logger = logging.getLogger(__name__)

    config_path = Path(config_path)
    config_manager = ConfigManager(config_path)

    level_filter = Gy85KalmanFilter()
    level_filter_lock = asyncio.Lock()

    CALIBRATION_OFFSET_LIMIT = 5.0
    CALIBRATION_SCALE_MIN = 0.5
    CALIBRATION_SCALE_MAX = 2.0

    def _build_distance_calibration(offset_m: float, scale: float) -> DistanceCalibration:
        try:
            calibration = DistanceCalibration(offset_m, scale)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if abs(calibration.offset_m) > CALIBRATION_OFFSET_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Calibration offset must be between "
                    f"{-CALIBRATION_OFFSET_LIMIT:g} and {CALIBRATION_OFFSET_LIMIT:g} metres"
                ),
            )
        if not (CALIBRATION_SCALE_MIN <= calibration.scale <= CALIBRATION_SCALE_MAX):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Calibration scale must be between "
                    f"{CALIBRATION_SCALE_MIN:g} and {CALIBRATION_SCALE_MAX:g}"
                ),
            )
        return calibration

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

    battery_monitor = BatteryMonitor(
        capacity_mah=config_manager.get_battery_capacity(),
        i2c_bus=i2c_bus_override,
    )
    battery_supervisor = BatterySupervisor(
        battery_monitor,
        config_manager.get_battery_limits,
        logger=logger,
    )
    distance_monitor = DistanceMonitor(
        i2c_bus=i2c_bus_override,
        calibration=config_manager.get_distance_calibration(),
    )
    if wifi_manager is None:
        credentials_path = config_path.with_name("wifi_credentials.json")
        wifi_manager = WiFiManager(credential_store=WiFiCredentialStore(credentials_path))

    pipeline = FramePipeline(
        lambda: config_manager.get_orientation(),
        overlay_enabled_provider=config_manager.get_overlay_master_enabled,
    )
    pipeline.add_overlay(
        create_wifi_overlay(
            wifi_manager.get_status,
            enabled_provider=config_manager.get_wifi_overlay_enabled,
        )
    )
    pipeline.add_overlay(
        create_battery_overlay(
            battery_monitor,
            config_manager.get_battery_limits,
            enabled_provider=config_manager.get_battery_overlay_enabled,
        )
    )
    pipeline.add_overlay(
        create_distance_overlay(
            distance_monitor,
            config_manager.get_distance_zones,
            config_manager.get_distance_overlay_enabled,
            geometry_provider=config_manager.get_distance_mounting,
            display_mode_provider=config_manager.get_distance_use_projected,
        )
    )
    pipeline.add_overlay(
        create_reversing_aids_overlay(
            config_manager.get_reversing_aids,
            enabled_provider=config_manager.get_reversing_overlay_enabled,
        )
    )
    camera: BaseCamera | None = None
    streamer: MJPEGStreamer | None = None
    webrtc_manager: WebRTCManager | None = None
    stream_error: str | None = None
    webrtc_error: str | None = None
    active_camera_choice: str = "unknown"
    active_resolution: Resolution = config_manager.get_resolution()
    camera_errors: dict[str, str] = {}

    def _distance_payload(
        reading,
        *,
        zones=None,
        calibration=None,
        overlay_enabled=None,
        mounting=None,
        use_projected=None,
    ) -> dict[str, object | None]:
        payload = reading.to_dict()
        if zones is None:
            zones = config_manager.get_distance_zones()
        if calibration is None:
            calibration = config_manager.get_distance_calibration()
        if overlay_enabled is None:
            overlay_enabled = config_manager.get_distance_overlay_enabled()
        if mounting is None:
            mounting = config_manager.get_distance_mounting()
        if use_projected is None:
            use_projected = config_manager.get_distance_use_projected()
        projection = _project_ground_distance(mounting, reading.distance_m)
        display_distance = _select_display_distance(
            reading.distance_m,
            projection,
            use_projected,
        )
        zone_reference = display_distance if display_distance is not None else reading.distance_m
        payload.update(
            zone=zones.classify(zone_reference) if zones is not None else None,
            zones=zones.to_dict() if zones is not None else {},
            calibration=calibration.to_dict() if calibration is not None else {},
            overlay_enabled=bool(overlay_enabled),
            geometry=mounting.to_dict() if mounting is not None else {},
            projected_distance_m=projection,
            use_projected_distance=bool(use_projected),
            display_distance_m=display_distance,
        )
        return payload

    led_ring = LedRing(logger=logging.getLogger(f"{__name__}.led_ring"))

    app.state.wifi_manager = wifi_manager
    app.state.distance_monitor = distance_monitor
    app.state.battery_supervisor = battery_supervisor
    app.state.led_ring = led_ring

    async def _set_ready_pattern() -> None:
        available = streamer is not None or webrtc_manager is not None
        await led_ring.set_pattern("ready" if available else "error")

    async def _serialise_led_status() -> dict[str, object]:
        status = await led_ring.get_status()
        colour = status.illumination_color
        colour_hex = "#" + "".join(f"{component:02X}" for component in colour)
        intensity_percent = max(
            0,
            min(100, int(round(status.illumination_intensity * 100))),
        )
        return {
            "patterns": list(status.patterns),
            "pattern": status.pattern,
            "active_pattern": status.active_pattern,
            "error": status.error,
            "available": status.available,
            "message": status.message,
            "illumination": {
                "color": colour_hex,
                "intensity": intensity_percent,
            },
        }

    def _parse_hex_colour(value: str) -> tuple[int, int, int]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Illumination colour must be a hex string in RRGGBB format")
        candidate = value.strip()
        if candidate.startswith("#"):
            candidate = candidate[1:]
        if len(candidate) != 6 or any(ch not in string.hexdigits for ch in candidate):
            raise ValueError("Illumination colour must be a hex string in RRGGBB format")
        red = int(candidate[0:2], 16)
        green = int(candidate[2:4], 16)
        blue = int(candidate[4:6], 16)
        return red, green, blue

    def _parse_illumination_intensity(value: float | int) -> float:
        try:
            percent = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Illumination intensity must be a number between 0 and 100") from exc
        if not math.isfinite(percent):
            raise ValueError("Illumination intensity must be finite")
        if not 0.0 <= percent <= 100.0:
            raise ValueError("Illumination intensity must be between 0 and 100")
        return percent / 100.0

    def _record_camera_error(source: str, message: str | None) -> None:
        if message:
            camera_errors[source] = message
        else:
            camera_errors.pop(source, None)

        if not camera_errors:
            has_error = False
        else:
            has_error = True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("Unable to signal LED error state; no running loop")
        else:
            loop.create_task(led_ring.set_error(has_error))

    async def _capture_snapshot_bytes() -> bytes:
        if camera is None:
            raise HTTPException(status_code=503, detail="Camera unavailable")

        try:
            frame = await camera.get_frame()
        except CameraError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Unexpected snapshot camera failure")
            raise HTTPException(status_code=500, detail="Failed to capture snapshot") from exc

        try:
            processed = await run_in_threadpool(pipeline.process, frame)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Snapshot pipeline processing failed")
            raise HTTPException(status_code=500, detail="Failed to process snapshot") from exc

        if streamer is not None:
            quality = streamer.jpeg_quality
        else:
            quality = config_manager.get_stream_settings().jpeg_quality

        try:
            jpeg_bytes = await run_in_threadpool(
                encode_frame_to_jpeg,
                processed,
                quality=quality,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Snapshot encoding failed")
            raise HTTPException(status_code=500, detail="Failed to encode snapshot") from exc

        return jpeg_bytes

    async def _snapshot_response() -> Response:
        payload = await _capture_snapshot_bytes()
        response = Response(content=payload, media_type="image/jpeg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    def _build_stream_info() -> dict[str, object | None]:
        stream_settings = config_manager.get_stream_settings()
        if streamer is not None:
            active_details: dict[str, int | None] | None = {
                "fps": streamer.fps,
                "jpeg_quality": streamer.jpeg_quality,
            }
        elif webrtc_manager is not None:
            active_details = {"fps": webrtc_manager.fps, "jpeg_quality": None}
        else:
            active_details = None

        if webrtc_manager is None:
            overall_error = webrtc_error or (stream_error if streamer is None else None)
        else:
            overall_error = stream_error if streamer is None else None

        return {
            "enabled": (streamer is not None) or (webrtc_manager is not None),
            "endpoint": "/stream/mjpeg" if streamer else None,
            "content_type": streamer.media_type if streamer else None,
            "error": overall_error,
            "settings": stream_settings.to_dict(),
            "active": active_details,
            "webrtc": {
                "enabled": webrtc_manager is not None,
                "endpoint": "/stream/webrtc",
                "error": webrtc_error,
                "fps": webrtc_manager.fps if webrtc_manager is not None else None,
            },
            "mjpeg": {
                "enabled": streamer is not None,
                "endpoint": "/stream/mjpeg",
                "error": stream_error,
                "content_type": streamer.media_type if streamer else None,
            },
        }

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
        nonlocal camera, active_camera_choice, streamer, webrtc_manager, active_resolution
        nonlocal stream_error, webrtc_error
        await led_ring.set_pattern("boot")
        try:
            await run_in_threadpool(wifi_manager.auto_connect_known_networks)
        except WiFiError as exc:
            logger.warning("Automatic Wi-Fi selection failed: %s", exc)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Unexpected error while selecting Wi-Fi network")
        finally:
            try:
                await run_in_threadpool(wifi_manager.start_hotspot_watchdog)
            except AttributeError:
                pass
        selection = config_manager.get_camera()
        resolution = config_manager.get_resolution()
        camera, active_camera_choice = _build_camera(
            selection,
            resolution,
            fallback_to_synthetic=True,
        )
        active_resolution = resolution
        stream_settings = config_manager.get_stream_settings()
        try:
            streamer = MJPEGStreamer(
                camera=camera,
                pipeline=pipeline,
                fps=stream_settings.fps,
                jpeg_quality=stream_settings.jpeg_quality,
            )
        except RuntimeError as exc:
            logger.error("Failed to initialise MJPEG streamer: %s", exc)
            stream_error = str(exc)
            streamer = None
        else:
            stream_error = None
        try:
            webrtc_manager = WebRTCManager(
                camera=camera,
                pipeline=pipeline,
                fps=stream_settings.fps,
            )
        except RuntimeError as exc:
            logger.error("Failed to initialise WebRTC streamer: %s", exc)
            webrtc_error = str(exc)
            webrtc_manager = None
        else:
            webrtc_error = None
        await _set_ready_pattern()
        battery_supervisor.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - framework hook
        nonlocal streamer, webrtc_manager
        await led_ring.set_pattern("boot")
        if streamer is not None:
            await streamer.aclose()
            streamer = None
        if webrtc_manager is not None:
            await webrtc_manager.aclose()
            webrtc_manager = None
        if camera:
            await camera.close()
        await battery_supervisor.aclose()
        distance_monitor.close()
        battery_monitor.close()
        if hasattr(wifi_manager, "close"):
            await run_in_threadpool(wifi_manager.close)
        await led_ring.aclose()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _load_static("index.html")

    @app.get("/settings", response_class=HTMLResponse)
    async def settings() -> str:
        return _load_static("settings.html")

    @app.get("/leveling", response_class=HTMLResponse)
    async def leveling_page() -> str:
        return _load_static("leveling.html")

    @app.get("/images/{asset}")
    async def get_image(asset: str):
        safe_name = Path(asset).name
        if safe_name != asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        path = STATIC_DIR / "images" / safe_name
        if not path.exists() or path.is_dir():
            raise HTTPException(status_code=404, detail="Asset not found")
        media_type = "image/svg+xml" if path.suffix.lower() == ".svg" else None
        return FileResponse(path, media_type=media_type)

    @app.get("/models/{asset}")
    async def get_model(asset: str):
        safe_name = Path(asset).name
        if safe_name != asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        path = STATIC_DIR / "models" / safe_name
        if not path.exists() or path.is_dir():
            raise HTTPException(status_code=404, detail="Asset not found")
        media_type = "model/stl" if path.suffix.lower() == ".stl" else None
        return FileResponse(path, media_type=media_type)

    @app.get("/api/led")
    async def get_led_status() -> dict[str, object]:
        return await _serialise_led_status()

    @app.post("/api/led")
    async def update_led_status(payload: LedSettingsPayload) -> dict[str, object]:
        updated = False
        if payload.pattern is not None:
            candidate = payload.pattern.strip().lower()
            if not candidate:
                raise HTTPException(status_code=400, detail="Pattern name must be provided")
            try:
                await led_ring.set_pattern(candidate)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            updated = True
        if payload.error is not None:
            await led_ring.set_error(bool(payload.error))
            updated = True
        colour_value: tuple[int, int, int] | None = None
        intensity_value: float | None = None
        if payload.illumination_color is not None:
            try:
                colour_value = _parse_hex_colour(payload.illumination_color)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if payload.illumination_intensity is not None:
            try:
                intensity_value = _parse_illumination_intensity(payload.illumination_intensity)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if colour_value is not None or intensity_value is not None:
            await led_ring.set_illumination(color=colour_value, intensity=intensity_value)
            updated = True
        if not updated:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Specify a pattern, error flag, or illumination setting to update the LED ring"
                ),
            )
        return await _serialise_led_status()

    @app.get("/api/orientation")
    async def get_orientation() -> dict[str, int | bool]:
        orientation = config_manager.get_orientation()
        return {
            "rotation": orientation.rotation,
            "flip_horizontal": orientation.flip_horizontal,
            "flip_vertical": orientation.flip_vertical,
        }

    @app.get("/api/diagnostics")
    async def get_diagnostics() -> dict[str, object]:
        try:
            return await run_in_threadpool(collect_diagnostics)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Diagnostics collection failed")
            detail = str(exc).strip()
            if detail:
                message = f"Unable to collect diagnostics: {detail}"
            else:
                message = "Unable to collect diagnostics"
            raise HTTPException(status_code=500, detail=message) from exc

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

    @app.get("/api/leveling/config")
    async def get_leveling_config() -> dict[str, object]:
        settings = config_manager.get_leveling_settings()
        return settings.to_dict()

    @app.post("/api/leveling/config")
    async def update_leveling_config(payload: LevelingConfigPayload) -> dict[str, object]:
        update_payload: dict[str, object] = {}
        if payload.geometry is not None:
            geometry_data = {
                key: value
                for key, value in payload.geometry.model_dump().items()
                if value is not None
            }
            if geometry_data:
                update_payload["geometry"] = geometry_data
        if payload.ramp is not None:
            ramp_data = {
                key: value
                for key, value in payload.ramp.model_dump().items()
                if value is not None
            }
            if ramp_data:
                update_payload["ramp"] = ramp_data
        if not update_payload:
            raise HTTPException(status_code=400, detail="No levelling values provided")
        try:
            settings = config_manager.set_leveling_settings(update_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return settings.to_dict()

    @app.post("/api/leveling/sample")
    async def process_leveling_sample(payload: LevelingSamplePayload) -> dict[str, object]:
        sample = SensorSample(
            accelerometer=Vector3(**payload.accelerometer.model_dump()),
            gyroscope=Vector3(**payload.gyroscope.model_dump()),
            magnetometer=Vector3(**payload.magnetometer.model_dump()),
        )
        async with level_filter_lock:
            orientation = level_filter.update(sample, dt=payload.dt)
        settings = config_manager.get_leveling_settings()
        evaluation = evaluate_leveling(orientation, settings)
        mode_key = "hitched" if payload.mode == "hitched" else "unhitched"
        return {
            "mode": payload.mode,
            "orientation": evaluation["orientation"],
            "analysis": evaluation[mode_key],
            "hitched": evaluation["hitched"],
            "unhitched": evaluation["unhitched"],
            "support_points": evaluation["support_points"],
            "settings": settings.to_dict(),
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
        stream_info = _build_stream_info()
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

    @app.get("/api/stream")
    async def get_stream_status() -> dict[str, object | None]:
        """Return the current streaming capabilities."""

        return _build_stream_info()

    @app.post("/api/stream/settings")
    async def update_stream_settings(payload: StreamSettingsPayload) -> dict[str, object | None]:
        nonlocal streamer, webrtc_manager
        data = payload.model_dump(exclude_none=True)
        if not data:
            raise HTTPException(status_code=400, detail="No streaming settings provided")
        try:
            settings = config_manager.set_stream_settings(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if streamer is not None:
            try:
                streamer.apply_settings(
                    fps=settings.fps,
                    jpeg_quality=settings.jpeg_quality,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to apply streaming settings")
                raise HTTPException(status_code=500, detail="Unable to apply streaming settings") from exc
        if webrtc_manager is not None:
            try:
                webrtc_manager.apply_settings(fps=settings.fps)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to apply WebRTC streaming settings")
                raise HTTPException(status_code=500, detail="Unable to apply streaming settings") from exc

        response: dict[str, object | None] = settings.to_dict()
        if streamer is not None:
            response["active"] = {
                "fps": streamer.fps,
                "jpeg_quality": streamer.jpeg_quality,
            }
        elif webrtc_manager is not None:
            response["active"] = {"fps": webrtc_manager.fps, "jpeg_quality": None}
        else:
            response["active"] = None
        return response

    @app.get("/api/battery")
    async def get_battery_status() -> dict[str, object | None]:
        reading = battery_monitor.read()
        return reading.to_dict()

    def _battery_settings_payload() -> dict[str, float | int]:
        limits = config_manager.get_battery_limits()
        capacity = config_manager.get_battery_capacity()
        payload: dict[str, float | int] = limits.to_dict()
        payload["capacity_mah"] = capacity
        return payload

    @app.get("/api/battery/limits")
    async def get_battery_limits() -> dict[str, float | int]:
        return _battery_settings_payload()

    @app.post("/api/battery/limits")
    async def update_battery_limits(payload: BatteryLimitsPayload) -> dict[str, float | int]:
        try:
            config_manager.set_battery_limits(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _battery_settings_payload()

    @app.post("/api/battery/capacity")
    async def update_battery_capacity(payload: BatteryCapacityPayload) -> dict[str, float | int]:
        try:
            capacity = config_manager.set_battery_capacity(payload.capacity_mah)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        battery_monitor.capacity_mah = capacity
        return _battery_settings_payload()

    @app.get("/api/distance")
    async def get_distance_status() -> dict[str, object | None]:
        reading = distance_monitor.read()
        zones = config_manager.get_distance_zones()
        calibration = config_manager.get_distance_calibration()
        overlay_enabled = config_manager.get_distance_overlay_enabled()
        mounting = config_manager.get_distance_mounting()
        use_projected = config_manager.get_distance_use_projected()
        return _distance_payload(
            reading,
            zones=zones,
            calibration=calibration,
            overlay_enabled=overlay_enabled,
            mounting=mounting,
            use_projected=use_projected,
        )

    @app.get("/api/distance/zones")
    async def get_distance_zones() -> dict[str, object]:
        zones = config_manager.get_distance_zones()
        return {"zones": zones.to_dict()}

    @app.post("/api/distance/zones")
    async def update_distance_zones(payload: DistanceZonesPayload) -> dict[str, object | None]:
        try:
            zones = config_manager.set_distance_zones(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        reading = distance_monitor.read()
        calibration = config_manager.get_distance_calibration()
        overlay_enabled = config_manager.get_distance_overlay_enabled()
        mounting = config_manager.get_distance_mounting()
        use_projected = config_manager.get_distance_use_projected()
        return _distance_payload(
            reading,
            zones=zones,
            calibration=calibration,
            overlay_enabled=overlay_enabled,
            mounting=mounting,
            use_projected=use_projected,
        )

    @app.get("/api/distance/calibration")
    async def get_distance_calibration_settings() -> dict[str, object]:
        calibration = config_manager.get_distance_calibration()
        return {"calibration": calibration.to_dict()}

    @app.post("/api/distance/calibration")
    async def update_distance_calibration(
        payload: DistanceCalibrationPayload,
    ) -> dict[str, object | None]:
        calibration = _build_distance_calibration(payload.offset_m, payload.scale)
        try:
            config_manager.set_distance_calibration(calibration)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        distance_monitor.set_calibration(calibration=calibration)
        zones = config_manager.get_distance_zones()
        reading = distance_monitor.read()
        overlay_enabled = config_manager.get_distance_overlay_enabled()
        mounting = config_manager.get_distance_mounting()
        use_projected = config_manager.get_distance_use_projected()
        return _distance_payload(
            reading,
            zones=zones,
            calibration=calibration,
            overlay_enabled=overlay_enabled,
            mounting=mounting,
            use_projected=use_projected,
        )

    @app.get("/api/distance/geometry")
    async def get_distance_geometry() -> dict[str, object | None]:
        mounting = config_manager.get_distance_mounting()
        return {
            "geometry": mounting.to_dict(),
            "projected_distance_m": _project_ground_distance(mounting),
            "use_projected_distance": config_manager.get_distance_use_projected(),
        }

    @app.post("/api/distance/geometry")
    async def update_distance_geometry(
        payload: DistanceGeometryPayload,
    ) -> dict[str, object | None]:
        try:
            mounting = config_manager.set_distance_mounting(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        zones = config_manager.get_distance_zones()
        reading = distance_monitor.read()
        calibration = config_manager.get_distance_calibration()
        overlay_enabled = config_manager.get_distance_overlay_enabled()
        use_projected = config_manager.get_distance_use_projected()
        return _distance_payload(
            reading,
            zones=zones,
            calibration=calibration,
            overlay_enabled=overlay_enabled,
            mounting=mounting,
            use_projected=use_projected,
        )

    @app.post("/api/distance/calibration/zero")
    async def zero_distance_calibration() -> dict[str, object | None]:
        reading = distance_monitor.read()
        if not (reading.available and reading.raw_distance_m is not None):
            raise HTTPException(
                status_code=503,
                detail="Distance reading unavailable for calibration",
            )
        current = distance_monitor.get_calibration()
        calibration = _build_distance_calibration(
            -current.scale * reading.raw_distance_m,
            current.scale,
        )
        try:
            config_manager.set_distance_calibration(calibration)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        distance_monitor.set_calibration(calibration=calibration)
        zones = config_manager.get_distance_zones()
        refreshed = distance_monitor.read()
        overlay_enabled = config_manager.get_distance_overlay_enabled()
        mounting = config_manager.get_distance_mounting()
        use_projected = config_manager.get_distance_use_projected()
        return _distance_payload(
            refreshed,
            zones=zones,
            calibration=calibration,
            overlay_enabled=overlay_enabled,
            mounting=mounting,
            use_projected=use_projected,
        )

    @app.post("/api/distance/display")
    async def update_distance_display_mode(
        payload: DistanceDisplayModePayload,
    ) -> dict[str, object | None]:
        try:
            use_projected = config_manager.set_distance_use_projected(
                payload.use_projected_distance
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        zones = config_manager.get_distance_zones()
        calibration = config_manager.get_distance_calibration()
        overlay_enabled = config_manager.get_distance_overlay_enabled()
        mounting = config_manager.get_distance_mounting()
        reading = distance_monitor.read()
        return _distance_payload(
            reading,
            zones=zones,
            calibration=calibration,
            overlay_enabled=overlay_enabled,
            mounting=mounting,
            use_projected=use_projected,
        )

    @app.post("/api/distance/overlay")
    async def update_distance_overlay(payload: DistanceOverlayPayload) -> dict[str, bool]:
        try:
            enabled = config_manager.set_distance_overlay_enabled(payload.enabled)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"overlay_enabled": enabled}

    @app.get("/api/overlays")
    def get_overlay_settings() -> dict[str, bool]:
        return {
            "master_enabled": config_manager.get_overlay_master_enabled(),
            "battery_enabled": config_manager.get_battery_overlay_enabled(),
            "wifi_enabled": config_manager.get_wifi_overlay_enabled(),
            "distance_enabled": config_manager.get_distance_overlay_enabled(),
            "reversing_aids_enabled": config_manager.get_reversing_overlay_enabled(),
        }

    @app.post("/api/overlays")
    def update_overlay_settings(payload: OverlaySettingsPayload) -> dict[str, bool]:
        try:
            if payload.master_enabled is not None:
                config_manager.set_overlay_master_enabled(payload.master_enabled)
            if payload.battery_enabled is not None:
                config_manager.set_battery_overlay_enabled(payload.battery_enabled)
            if payload.wifi_enabled is not None:
                config_manager.set_wifi_overlay_enabled(payload.wifi_enabled)
            if payload.distance_enabled is not None:
                config_manager.set_distance_overlay_enabled(payload.distance_enabled)
            if payload.reversing_aids_enabled is not None:
                config_manager.set_reversing_overlay_enabled(payload.reversing_aids_enabled)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return get_overlay_settings()

    @app.get("/api/reversing-aids")
    def get_reversing_aids() -> dict[str, object]:
        config = config_manager.get_reversing_aids()
        return config.to_dict()

    @app.post("/api/reversing-aids")
    def update_reversing_aids(payload: ReversingAidsPayload) -> dict[str, object]:
        try:
            config = config_manager.set_reversing_aids(payload.dict(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return config.to_dict()

    @app.get("/api/wifi/status")
    async def get_wifi_status() -> dict[str, object | None]:
        try:
            status = await run_in_threadpool(wifi_manager.get_status)
        except WiFiError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return status.to_dict()

    @app.get("/api/wifi/log")
    async def get_wifi_log(limit: int = 50) -> dict[str, object]:
        try:
            entries = await run_in_threadpool(wifi_manager.get_connection_log, limit)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Unable to load Wi-Fi log: %s", exc)
            raise HTTPException(status_code=503, detail="Unable to load Wi-Fi log") from exc
        return {"entries": entries}

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
                    payload.ssid,
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

    @app.post("/api/wifi/forget")
    async def forget_wifi_network(payload: WiFiForgetPayload) -> dict[str, object | None]:
        try:
            status = await run_in_threadpool(
                wifi_manager.forget_network,
                payload.identifier,
            )
        except WiFiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return status.to_dict()

    @app.post("/api/camera")
    async def update_camera(payload: CameraPayload) -> dict[str, object]:
        nonlocal camera, active_camera_choice, streamer, webrtc_manager
        nonlocal active_resolution, stream_error, webrtc_error
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
            stream_info = _build_stream_info()
            await _set_ready_pattern()
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

        await led_ring.set_pattern("boot")
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
                stream_settings = config_manager.get_stream_settings()
                if streamer is None and camera is not None:
                    try:
                        streamer = MJPEGStreamer(
                            camera=camera,
                            pipeline=pipeline,
                            fps=stream_settings.fps,
                            jpeg_quality=stream_settings.jpeg_quality,
                        )
                    except RuntimeError as streamer_exc:
                        logger.error("Failed to initialise MJPEG streamer: %s", streamer_exc)
                        stream_error = str(streamer_exc)
                        streamer = None
                    else:
                        stream_error = None
                elif streamer is not None and camera is not None:
                    streamer.camera = camera
                    stream_error = None
                if webrtc_manager is not None:
                    await webrtc_manager.aclose()
                    webrtc_manager = None
                if camera is not None:
                    try:
                        webrtc_manager = WebRTCManager(
                            camera=camera,
                            pipeline=pipeline,
                            fps=stream_settings.fps,
                        )
                    except RuntimeError as webrtc_exc:
                        logger.error("Failed to initialise WebRTC streamer: %s", webrtc_exc)
                        webrtc_error = str(webrtc_exc)
                        webrtc_manager = None
                    else:
                        webrtc_error = None
            await _set_ready_pattern()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        camera = new_camera
        active_resolution = requested_resolution
        config_manager.set_camera(selection)
        config_manager.set_resolution(requested_resolution)
        stream_settings = config_manager.get_stream_settings()
        if streamer is None:
            try:
                streamer = MJPEGStreamer(
                    camera=camera,
                    pipeline=pipeline,
                    fps=stream_settings.fps,
                    jpeg_quality=stream_settings.jpeg_quality,
                )
            except RuntimeError as exc:
                logger.error("Failed to initialise MJPEG streamer: %s", exc)
                stream_error = str(exc)
                streamer = None
            else:
                stream_error = None
        else:
            streamer.camera = camera
            stream_error = None
        if webrtc_manager is not None:
            await webrtc_manager.aclose()
            webrtc_manager = None
        try:
            webrtc_manager = WebRTCManager(
                camera=camera,
                pipeline=pipeline,
                fps=stream_settings.fps,
            )
        except RuntimeError as exc:
            logger.error("Failed to initialise WebRTC streamer: %s", exc)
            webrtc_error = str(exc)
            webrtc_manager = None
        else:
            webrtc_error = None
        await _set_ready_pattern()
        stream_info = _build_stream_info()
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

    @app.get("/api/camera/snapshot")
    async def get_camera_snapshot() -> Response:
        return await _snapshot_response()

    @app.get("/snapshot.jpg")
    async def legacy_snapshot() -> Response:
        return await _snapshot_response()

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

    @app.post("/stream/webrtc")
    async def stream_webrtc(payload: WebRTCOfferPayload) -> dict[str, str]:
        if webrtc_manager is None:
            detail = webrtc_error or stream_error or "WebRTC streaming unavailable"
            raise HTTPException(status_code=503, detail=detail)
        try:
            description = await webrtc_manager.create_session(payload.sdp, payload.type)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to negotiate WebRTC session")
            raise HTTPException(status_code=500, detail="Failed to establish WebRTC session") from exc
        return {"sdp": description.sdp, "type": description.type}

    @app.post("/api/log/webrtc-error")
    async def log_webrtc_error(payload: WebRTCErrorReportPayload) -> dict[str, str]:
        summary_parts = []
        if payload.name:
            summary_parts.append(payload.name)
        if payload.message:
            summary_parts.append(payload.message)
        summary = " – ".join(summary_parts) if summary_parts else "Unspecified WebRTC error"
        logger.warning("Client reported WebRTC error: %s", summary)
        if payload.stack:
            logger.debug("Client WebRTC error stack trace:\n%s", payload.stack)
        return {"status": "logged"}

    return app


__all__ = ["create_app"]
