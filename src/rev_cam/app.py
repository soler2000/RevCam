"""FastAPI application wiring together the RevCam services."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import string
from enum import Enum
from pathlib import Path
from typing import Literal

from datetime import datetime, timedelta, timezone

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
    SURVEILLANCE_STANDARD_PRESETS,
)
from .diagnostics import collect_diagnostics
from .distance import DistanceCalibration, DistanceMonitor, create_distance_overlay
from .led_matrix import LedRing
from .reversing_aids import create_reversing_aids_overlay
from .sensor_fusion import Gy85KalmanFilter, SensorSample, Vector3
from .pipeline import FramePipeline
from .recording import (
    RecordingManager,
    load_recording_metadata,
    load_recording_payload,
    purge_recordings,
    remove_recording_files,
)
from .streaming import MJPEGStreamer, WebRTCManager, encode_frame_to_jpeg
from .system_log import SystemLog
from .version import APP_VERSION
from .wifi import WiFiCredentialStore, WiFiError, WiFiManager
from .trailer_leveling import evaluate_leveling

STATIC_DIR = Path(__file__).resolve().parent / "static"
RECORDINGS_DIR = Path(os.environ.get("REVCAM_RECORDINGS_DIR", STATIC_DIR / "recordings"))


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


class StreamMode(str, Enum):
    REVCAM = "revcam"
    SURVEILLANCE = "surveillance"


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


class SurveillanceModePayload(BaseModel):
    mode: Literal["revcam", "surveillance"]


class BatteryLimitsPayload(BaseModel):
    warning_percent: float
    shutdown_percent: float


class BatteryCapacityPayload(BaseModel):
    capacity_mah: int


class StreamSettingsPayload(BaseModel):
    fps: int | None = None
    jpeg_quality: int | None = None


class SurveillanceSettingsPayload(BaseModel):
    profile: Literal["standard", "expert"] | None = None
    preset: str | None = None
    fps: int | None = None
    jpeg_quality: int | None = None
    expert_fps: int | None = None
    expert_jpeg_quality: int | None = None
    chunk_duration_seconds: int | None = None
    overlays_enabled: bool | None = None
    remember_recording_state: bool | None = None
    motion_detection_enabled: bool | None = None
    motion_sensitivity: int | None = None
    auto_purge_days: int | None = None
    storage_threshold_percent: float | None = None


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

    shared_system_log: SystemLog
    if wifi_manager is None:
        credentials_path = config_path.with_name("wifi_credentials.json")
        shared_system_log = SystemLog()
        wifi_manager = WiFiManager(
            credential_store=WiFiCredentialStore(credentials_path),
            system_log=shared_system_log,
        )
    else:
        existing_log = getattr(wifi_manager, "system_log", None)
        if isinstance(existing_log, SystemLog):
            shared_system_log = existing_log
        else:
            shared_system_log = SystemLog()
            if hasattr(wifi_manager, "_system_log"):
                wifi_manager._system_log = shared_system_log

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
        system_log=shared_system_log,
    )

    current_mode: StreamMode = StreamMode.REVCAM

    def _overlays_enabled() -> bool:
        master_enabled = config_manager.get_overlay_master_enabled()
        if not master_enabled:
            return False
        if current_mode is StreamMode.SURVEILLANCE:
            try:
                surveillance = config_manager.get_surveillance_settings()
            except Exception:  # pragma: no cover - defensive fallback
                return master_enabled
            if not surveillance.overlays_enabled:
                return False
        return True

    pipeline = FramePipeline(
        lambda: config_manager.get_orientation(),
        overlay_enabled_provider=_overlays_enabled,
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
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    camera: BaseCamera | None = None
    streamer: MJPEGStreamer | None = None
    webrtc_manager: WebRTCManager | None = None
    recording_manager: RecordingManager | None = None
    stream_error: str | None = None
    webrtc_error: str | None = None
    active_camera_choice: str = "unknown"
    active_resolution: Resolution = config_manager.get_resolution()
    camera_errors: dict[str, str] = {}
    def _persist_surveillance_state(recording: bool | None = None) -> None:
        try:
            settings = config_manager.get_surveillance_settings()
        except Exception:  # pragma: no cover - defensive guard
            return
        if not settings.remember_recording_state:
            return
        if recording is None:
            recording = recording_manager.is_recording if recording_manager else False
        try:
            config_manager.update_surveillance_state(current_mode.value, bool(recording))
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Failed to persist surveillance runtime state")

    async def _handle_recording_stopped(metadata: dict[str, object]) -> None:
        _persist_surveillance_state(False)

    async def _apply_auto_purge(settings) -> None:
        days = getattr(settings, "auto_purge_days", None)
        if days in (None, 0):
            return
        try:
            days_value = float(days)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return
        if days_value <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_value)
        try:
            removed = await asyncio.to_thread(purge_recordings, RECORDINGS_DIR, cutoff)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to auto purge surveillance recordings")
            return
        if removed:
            logger.info("Purged %d old surveillance recordings", len(removed))

    def _build_storage_status(settings=None) -> dict[str, object]:
        status: dict[str, object]
        try:
            if recording_manager is not None:
                status = recording_manager.get_storage_status()
            else:
                usage = shutil.disk_usage(RECORDINGS_DIR)
                total = int(getattr(usage, "total", 0))
                free = int(getattr(usage, "free", 0))
                used = int(getattr(usage, "used", total - free))
                free_percent = 100.0 if total <= 0 else max(0.0, min(100.0, (free / total) * 100.0))
                status = {
                    "total_bytes": total,
                    "free_bytes": free,
                    "used_bytes": used,
                    "free_percent": round(free_percent, 3),
                }
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to compute storage status")
            status = {}
        if settings is None:
            try:
                settings = config_manager.get_surveillance_settings()
            except Exception:  # pragma: no cover - defensive guard
                settings = None
        if settings is not None:
            status.setdefault("threshold_percent", float(settings.storage_threshold_percent))
        return status

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
    app.state.system_log = shared_system_log

    async def _set_ready_pattern() -> None:
        available = (
            streamer is not None
            or webrtc_manager is not None
            or (recording_manager is not None and current_mode is StreamMode.SURVEILLANCE)
        )
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

    async def _serialise_surveillance_status() -> dict[str, object]:
        recordings: list[dict[str, object]] = []
        processing_record: dict[str, object] | None = None
        processing_active = False
        if recording_manager is not None:
            try:
                recordings = await recording_manager.list_recordings()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to list surveillance recordings")
                recordings = []
            try:
                processing_record = await recording_manager.get_processing_metadata()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to read in-flight surveillance metadata")
                processing_record = None
            processing_active = recording_manager.is_processing
        else:
            try:
                recordings = await asyncio.to_thread(
                    load_recording_metadata, RECORDINGS_DIR
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to list surveillance recordings")
                recordings = []
        preview: dict[str, object] | None = None
        if current_mode is StreamMode.SURVEILLANCE and recording_manager is not None:
            preview = {
                "endpoint": "/surveillance/preview.mjpeg",
                "content_type": recording_manager.media_type,
            }
        settings_obj = config_manager.get_surveillance_settings()
        await _apply_auto_purge(settings_obj)
        surveillance_settings = settings_obj.serialise()
        storage_status = _build_storage_status(settings_obj)
        motion_state = (
            recording_manager.get_motion_status() if recording_manager is not None else None
        )
        recording_mode = (
            recording_manager.recording_mode if recording_manager is not None else "idle"
        )
        is_recording_flag = recording_manager.is_recording if recording_manager else False
        state = config_manager.get_surveillance_state().to_dict()
        return {
            "mode": current_mode.value,
            "recording": is_recording_flag,
            "recording_mode": recording_mode,
            "recordings": recordings,
            "preview": preview,
            "settings": surveillance_settings,
            "storage": storage_status,
            "motion": motion_state,
            "resume_state": state,
            "processing": processing_active,
            "processing_recording": processing_record,
        }

    async def _ensure_recording_manager() -> RecordingManager:
        nonlocal recording_manager
        if camera is None:
            raise RuntimeError("Camera unavailable for surveillance mode")
        surveillance_settings = config_manager.get_surveillance_settings()
        fps = surveillance_settings.resolved_fps
        jpeg_quality = surveillance_settings.resolved_jpeg_quality
        if recording_manager is None:
            recording_manager = RecordingManager(
                camera=camera,
                pipeline=pipeline,
                fps=fps,
                jpeg_quality=jpeg_quality,
                directory=RECORDINGS_DIR,
                chunk_duration_seconds=surveillance_settings.chunk_duration_seconds,
                storage_threshold_percent=surveillance_settings.storage_threshold_percent,
                motion_detection_enabled=surveillance_settings.motion_detection_enabled,
                motion_sensitivity=surveillance_settings.motion_sensitivity,
                motion_frame_decimation=surveillance_settings.motion_frame_decimation,
                motion_post_event_seconds=surveillance_settings.motion_post_event_seconds,
                on_stop=_handle_recording_stopped,
            )
        else:
            recording_manager.camera = camera
            await recording_manager.apply_settings(
                fps=fps,
                jpeg_quality=jpeg_quality,
                chunk_duration_seconds=surveillance_settings.chunk_duration_seconds,
                storage_threshold_percent=surveillance_settings.storage_threshold_percent,
                motion_detection_enabled=surveillance_settings.motion_detection_enabled,
                motion_sensitivity=surveillance_settings.motion_sensitivity,
                motion_frame_decimation=surveillance_settings.motion_frame_decimation,
                motion_post_event_seconds=surveillance_settings.motion_post_event_seconds,
            )
        return recording_manager

    async def _activate_surveillance_mode() -> RecordingManager:
        nonlocal streamer, webrtc_manager, recording_manager, current_mode
        nonlocal stream_error, webrtc_error
        if current_mode is StreamMode.SURVEILLANCE and recording_manager is not None:
            return recording_manager
        if streamer is not None:
            await streamer.aclose()
            streamer = None
        if webrtc_manager is not None:
            await webrtc_manager.aclose()
            webrtc_manager = None
        stream_error = None
        webrtc_error = None
        manager = await _ensure_recording_manager()
        current_mode = StreamMode.SURVEILLANCE
        _persist_surveillance_state()
        await _set_ready_pattern()
        return manager

    async def _activate_revcam_mode() -> None:
        nonlocal streamer, webrtc_manager, recording_manager, current_mode
        nonlocal stream_error, webrtc_error
        if recording_manager is not None:
            try:
                if recording_manager.is_recording:
                    await recording_manager.stop_recording()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to finalise active recording while switching modes")
            await recording_manager.aclose()
            recording_manager = None
        current_mode = StreamMode.REVCAM
        _persist_surveillance_state(False)
        await _refresh_video_services()
        await _set_ready_pattern()

    async def _refresh_video_services() -> None:
        nonlocal streamer, webrtc_manager, stream_error, webrtc_error, recording_manager
        stream_settings = config_manager.get_stream_settings()
        if current_mode is StreamMode.SURVEILLANCE:
            if streamer is not None:
                await streamer.aclose()
                streamer = None
            if webrtc_manager is not None:
                await webrtc_manager.aclose()
                webrtc_manager = None
            stream_error = None
            webrtc_error = None
            if camera is not None:
                try:
                    await _ensure_recording_manager()
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to prepare surveillance manager")
                    raise
            return

        if camera is None:
            streamer = None
            webrtc_manager = None
            stream_error = "Camera unavailable"
            webrtc_error = "Camera unavailable"
            return

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
        previous = camera_errors.get(source)
        cleaned: str | None
        if message:
            cleaned = str(message).strip()
            camera_errors[source] = cleaned
        else:
            cleaned = None
            camera_errors.pop(source, None)

        if not camera_errors:
            has_error = False
        else:
            has_error = True

        if isinstance(shared_system_log, SystemLog):
            if cleaned and cleaned != previous:
                shared_system_log.record(
                    "camera",
                    "camera_error",
                    f"Camera source {source} reported an error.",
                    metadata={"source": source, "error": cleaned},
                )
            elif cleaned is None and previous is not None:
                shared_system_log.record(
                    "camera",
                    "camera_recovered",
                    f"Camera source {source} recovered.",
                    metadata={"source": source},
                )

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
        in_surveillance = current_mode is StreamMode.SURVEILLANCE
        recording_available = recording_manager is not None
        preview_details: dict[str, object | None] | None = None
        if in_surveillance and recording_available:
            preview_details = {
                "enabled": True,
                "endpoint": "/surveillance/preview.mjpeg",
                "content_type": recording_manager.media_type,
            }

        if not in_surveillance and streamer is not None:
            active_details: dict[str, int | None] | None = {
                "fps": streamer.fps,
                "jpeg_quality": streamer.jpeg_quality,
            }
        elif not in_surveillance and webrtc_manager is not None:
            active_details = {"fps": webrtc_manager.fps, "jpeg_quality": None}
        else:
            active_details = None

        if webrtc_manager is None:
            overall_error = webrtc_error or (stream_error if streamer is None else None)
        else:
            overall_error = stream_error if streamer is None else None

        mjpeg_enabled = (not in_surveillance) and (streamer is not None)
        webrtc_enabled = (not in_surveillance) and (webrtc_manager is not None)

        return {
            "mode": current_mode.value,
            "enabled": mjpeg_enabled or webrtc_enabled,
            "endpoint": "/stream/mjpeg" if mjpeg_enabled else None,
            "content_type": streamer.media_type if mjpeg_enabled and streamer else None,
            "error": overall_error if not in_surveillance else "Surveillance mode active",
            "settings": stream_settings.to_dict(),
            "active": active_details,
            "webrtc": {
                "enabled": webrtc_enabled,
                "endpoint": "/stream/webrtc",
                "error": webrtc_error if not in_surveillance else "Surveillance mode active",
                "fps": webrtc_manager.fps if webrtc_enabled and webrtc_manager is not None else None,
            },
            "mjpeg": {
                "enabled": mjpeg_enabled,
                "endpoint": "/stream/mjpeg",
                "error": stream_error if mjpeg_enabled else ("Surveillance mode active" if in_surveillance else stream_error),
                "content_type": streamer.media_type if mjpeg_enabled and streamer else None,
            },
            "surveillance": preview_details,
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
        nonlocal stream_error, webrtc_error, recording_manager, current_mode
        if isinstance(shared_system_log, SystemLog):
            shared_system_log.record(
                "system",
                "startup",
                "RevCam application starting up.",
            )
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
        recording_manager = None
        current_mode = StreamMode.REVCAM
        battery_supervisor.start()
        if isinstance(shared_system_log, SystemLog):
            startup_metadata = {
                "camera": active_camera_choice,
                "resolution": active_resolution.key()
                if hasattr(active_resolution, "key")
                else None,
                "stream": stream_settings.to_dict(),
            }
            shared_system_log.record(
                "system",
                "startup_complete",
                "RevCam startup sequence completed.",
                metadata={k: v for k, v in startup_metadata.items() if v is not None},
            )
        try:
            surveillance_settings = config_manager.get_surveillance_settings()
            runtime_state = config_manager.get_surveillance_state()
        except Exception:  # pragma: no cover - defensive guard
            surveillance_settings = None
            runtime_state = None
        if (
            surveillance_settings is not None
            and runtime_state is not None
            and surveillance_settings.remember_recording_state
            and runtime_state.mode == "surveillance"
        ):
            try:
                manager = await _activate_surveillance_mode()
                if runtime_state.recording:
                    await manager.start_recording()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to resume surveillance recording state")

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - framework hook
        nonlocal streamer, webrtc_manager, recording_manager
        if isinstance(shared_system_log, SystemLog):
            shared_system_log.record(
                "system",
                "shutdown",
                "RevCam application shutting down.",
            )
        await led_ring.set_pattern("boot")
        if streamer is not None:
            await streamer.aclose()
            streamer = None
        if webrtc_manager is not None:
            await webrtc_manager.aclose()
            webrtc_manager = None
        if recording_manager is not None:
            await recording_manager.aclose()
            recording_manager = None
        _persist_surveillance_state(False)
        if camera:
            await camera.close()
        await battery_supervisor.aclose()
        distance_monitor.close()
        battery_monitor.close()
        if hasattr(wifi_manager, "close"):
            await run_in_threadpool(wifi_manager.close)
        await led_ring.aclose()
        if isinstance(shared_system_log, SystemLog):
            shared_system_log.record(
                "system",
                "shutdown_complete",
                "RevCam shutdown sequence completed.",
                metadata={"camera": active_camera_choice},
            )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _load_static("index.html")

    @app.get("/settings", response_class=HTMLResponse)
    async def settings() -> str:
        return _load_static("settings.html")

    @app.get("/leveling", response_class=HTMLResponse)
    async def leveling_page() -> str:
        return _load_static("leveling.html")

    @app.get("/surveillance", response_class=HTMLResponse)
    async def surveillance_page() -> str:
        return _load_static("surveillance.html")

    @app.get("/surveillance/player", response_class=HTMLResponse)
    async def surveillance_player_page() -> str:
        return _load_static("surveillance_player.html")

    @app.get("/surveillance/settings", response_class=HTMLResponse)
    async def surveillance_settings_page() -> str:
        return _load_static("surveillance_settings.html")

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
        nonlocal streamer, webrtc_manager, recording_manager
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
        if recording_manager is not None:
            try:
                surveillance_settings = config_manager.get_surveillance_settings()
                await recording_manager.apply_settings(
                    fps=surveillance_settings.resolved_fps,
                    jpeg_quality=surveillance_settings.resolved_jpeg_quality,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to apply surveillance recording settings")
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

    @app.get("/api/logs")
    async def get_system_log_entries(
        limit: int = 100, category: str | None = None
    ) -> dict[str, object]:
        try:
            entries = await run_in_threadpool(
                shared_system_log.tail, limit, category=category
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Unable to load system log: %s", exc)
            raise HTTPException(status_code=503, detail="Unable to load system log") from exc
        ordered = list(reversed(entries))
        return {"entries": [entry.to_dict() for entry in ordered]}

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
        nonlocal active_resolution, stream_error, webrtc_error, recording_manager, current_mode
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
                try:
                    await _refresh_video_services()
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to restore video services after camera error")
            await _set_ready_pattern()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        camera = new_camera
        active_resolution = requested_resolution
        config_manager.set_camera(selection)
        config_manager.set_resolution(requested_resolution)
        try:
            await _refresh_video_services()
        except Exception as exc:
            await _set_ready_pattern()
            raise HTTPException(status_code=500, detail="Unable to configure video pipeline") from exc
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

    @app.get("/surveillance/preview.mjpeg")
    async def surveillance_preview():
        if current_mode is not StreamMode.SURVEILLANCE:
            raise HTTPException(status_code=409, detail="Surveillance mode inactive")
        try:
            manager = await _ensure_recording_manager()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response = StreamingResponse(manager.stream(), media_type=manager.media_type)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/api/surveillance/status")
    async def surveillance_status() -> dict[str, object]:
        return await _serialise_surveillance_status()

    @app.get("/api/surveillance/settings")
    async def get_surveillance_settings() -> dict[str, object]:
        settings = config_manager.get_surveillance_settings()
        presets = [
            {"name": name, "fps": fps, "jpeg_quality": quality}
            for name, (fps, quality) in SURVEILLANCE_STANDARD_PRESETS.items()
        ]
        return {"settings": settings.serialise(), "presets": presets}

    @app.post("/api/surveillance/settings")
    async def update_surveillance_settings(
        payload: SurveillanceSettingsPayload,
    ) -> dict[str, object]:
        raw = payload.model_dump(exclude_none=True)
        if "fps" in raw and "expert_fps" not in raw:
            raw["expert_fps"] = raw.pop("fps")
        if "jpeg_quality" in raw and "expert_jpeg_quality" not in raw:
            raw["expert_jpeg_quality"] = raw.pop("jpeg_quality")
        try:
            settings = config_manager.set_surveillance_settings(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if recording_manager is not None:
            try:
                await recording_manager.apply_settings(
                    fps=settings.resolved_fps,
                    jpeg_quality=settings.resolved_jpeg_quality,
                    chunk_duration_seconds=settings.chunk_duration_seconds,
                    storage_threshold_percent=settings.storage_threshold_percent,
                    motion_detection_enabled=settings.motion_detection_enabled,
                    motion_sensitivity=settings.motion_sensitivity,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to apply surveillance recording settings")
                raise HTTPException(
                    status_code=500,
                    detail="Unable to apply surveillance settings",
                ) from exc
        return {"settings": settings.serialise()}

    @app.post("/api/surveillance/mode")
    async def set_surveillance_mode(payload: SurveillanceModePayload) -> dict[str, object]:
        target = StreamMode(payload.mode)
        if target is current_mode:
            return await _serialise_surveillance_status()
        if target is StreamMode.SURVEILLANCE:
            try:
                await _activate_surveillance_mode()
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        else:
            await _activate_revcam_mode()
        return await _serialise_surveillance_status()

    @app.post("/api/surveillance/recordings/start")
    async def start_surveillance_recording() -> dict[str, object]:
        if current_mode is not StreamMode.SURVEILLANCE:
            raise HTTPException(status_code=409, detail="Enable surveillance mode first")
        try:
            manager = await _ensure_recording_manager()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        settings = config_manager.get_surveillance_settings()
        storage_status = _build_storage_status(settings)
        free_percent = storage_status.get("free_percent")
        threshold = storage_status.get("threshold_percent")
        if (
            isinstance(free_percent, (int, float))
            and isinstance(threshold, (int, float))
            and threshold > 0
            and free_percent <= threshold
        ):
            raise HTTPException(status_code=507, detail="Insufficient storage available")
        try:
            details = await manager.start_recording()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _persist_surveillance_state(True)
        return {"recording": details}

    @app.post("/api/surveillance/recordings/start-motion")
    async def start_motion_surveillance_recording() -> dict[str, object]:
        if current_mode is not StreamMode.SURVEILLANCE:
            raise HTTPException(status_code=409, detail="Enable surveillance mode first")
        try:
            manager = await _ensure_recording_manager()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        settings = config_manager.get_surveillance_settings()
        storage_status = _build_storage_status(settings)
        free_percent = storage_status.get("free_percent")
        threshold = storage_status.get("threshold_percent")
        if (
            isinstance(free_percent, (int, float))
            and isinstance(threshold, (int, float))
            and threshold > 0
            and free_percent <= threshold
        ):
            raise HTTPException(status_code=507, detail="Insufficient storage available")
        try:
            details = await manager.start_recording(motion_mode=True)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _persist_surveillance_state(True)
        return {"recording": details}

    @app.post("/api/surveillance/recordings/stop")
    async def stop_surveillance_recording() -> dict[str, object]:
        if current_mode is not StreamMode.SURVEILLANCE:
            raise HTTPException(status_code=409, detail="Enable surveillance mode first")
        if recording_manager is None:
            raise HTTPException(status_code=409, detail="No recording in progress")
        try:
            details = await recording_manager.stop_recording()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _persist_surveillance_state(False)
        return {"recording": details}

    @app.get("/api/surveillance/recordings")
    async def list_surveillance_recordings() -> dict[str, object]:
        status = await _serialise_surveillance_status()
        return {"recordings": status.get("recordings", [])}

    @app.get("/api/surveillance/recordings/{name}")
    async def fetch_surveillance_recording(name: str) -> dict[str, object]:
        if recording_manager is None:
            try:
                return await asyncio.to_thread(
                    load_recording_payload, RECORDINGS_DIR, name
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Recording not found") from exc
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to load surveillance recording %s", name)
                raise HTTPException(
                    status_code=500, detail="Unable to load recording"
                ) from exc
        try:
            return await recording_manager.get_recording(name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Recording not found") from exc

    @app.delete("/api/surveillance/recordings/{name}")
    async def delete_surveillance_recording(name: str) -> dict[str, object]:
        try:
            if recording_manager is not None:
                await recording_manager.remove_recording(name)
            else:
                await asyncio.to_thread(remove_recording_files, RECORDINGS_DIR, name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Recording not found")
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to remove surveillance recording %s", name)
            raise HTTPException(status_code=500, detail="Unable to delete recording")
        return {"deleted": name}

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
