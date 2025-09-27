"""FastAPI application wiring together the RevCam services."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import string
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .battery import BatteryMonitor, BatterySupervisor, create_battery_overlay
from .camera import CAMERA_SOURCES, BaseCamera, CameraError, create_camera, identify_camera
from .config import ConfigManager, Resolution, RESOLUTION_PRESETS, StreamSettings
from .diagnostics import collect_diagnostics
from .distance import DistanceCalibration, DistanceMonitor, create_distance_overlay
from .led_matrix import LedRing
from .reversing_aids import create_reversing_aids_overlay
from .pipeline import FramePipeline
from .streaming import MJPEGStreamer, WebRTCManager, encode_frame_to_jpeg
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


class WiFiForgetPayload(BaseModel):
    identifier: str


class DistanceZonesPayload(BaseModel):
    caution: float
    warning: float
    danger: float


class DistanceCalibrationPayload(BaseModel):
    offset_m: float
    scale: float


class DistanceOverlayPayload(BaseModel):
    enabled: bool


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


class LedSettingsPayload(BaseModel):
    pattern: str | None = None
    error: bool | None = None
    illumination_color: str | None = None
    illumination_intensity: float | int | None = None


def create_app(
    config_path: Path | str = Path("data/config.json"),
    *,
    wifi_manager: WiFiManager | None = None,
) -> FastAPI:
    app = FastAPI(title="RevCam", version=APP_VERSION)

    logger = logging.getLogger(__name__)

    config_manager = ConfigManager(Path(config_path))

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
    wifi_manager = wifi_manager or WiFiManager()

    pipeline = FramePipeline(lambda: config_manager.get_orientation())
    pipeline.add_overlay(
        create_battery_overlay(
            battery_monitor,
            config_manager.get_battery_limits,
            wifi_manager.get_status,
        )
    )
    pipeline.add_overlay(
        create_distance_overlay(
            distance_monitor,
            config_manager.get_distance_zones,
            config_manager.get_distance_overlay_enabled,
        )
    )
    pipeline.add_overlay(create_reversing_aids_overlay(config_manager.get_reversing_aids))
    camera: BaseCamera | None = None
    streamer: MJPEGStreamer | None = None
    webrtc_manager: WebRTCManager | None = None
    stream_error: str | None = None
    webrtc_error: str | None = None
    active_camera_choice: str = "unknown"
    active_resolution: Resolution = config_manager.get_resolution()
    camera_errors: dict[str, str] = {}

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
        zones = config_manager.get_distance_zones()
        reading = distance_monitor.read()
        payload = reading.to_dict()
        payload["zone"] = zones.classify(reading.distance_m)
        payload["zones"] = zones.to_dict()
        payload["calibration"] = config_manager.get_distance_calibration().to_dict()
        payload["overlay_enabled"] = config_manager.get_distance_overlay_enabled()
        return payload

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
        response = reading.to_dict()
        response["zone"] = zones.classify(reading.distance_m)
        response["zones"] = zones.to_dict()
        response["calibration"] = config_manager.get_distance_calibration().to_dict()
        response["overlay_enabled"] = config_manager.get_distance_overlay_enabled()
        return response

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
        response = reading.to_dict()
        response["zone"] = zones.classify(reading.distance_m)
        response["zones"] = zones.to_dict()
        response["calibration"] = calibration.to_dict()
        response["overlay_enabled"] = config_manager.get_distance_overlay_enabled()
        return response

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
        response = refreshed.to_dict()
        response["zone"] = zones.classify(refreshed.distance_m)
        response["zones"] = zones.to_dict()
        response["calibration"] = calibration.to_dict()
        response["overlay_enabled"] = config_manager.get_distance_overlay_enabled()
        return response

    @app.post("/api/distance/overlay")
    async def update_distance_overlay(payload: DistanceOverlayPayload) -> dict[str, bool]:
        try:
            enabled = config_manager.set_distance_overlay_enabled(payload.enabled)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"overlay_enabled": enabled}

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

    return app


__all__ = ["create_app"]
