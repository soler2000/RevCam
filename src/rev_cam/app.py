"""FastAPI application wiring together the RevCam services."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .camera import BaseCamera, create_camera
from .config import ConfigManager
from .pipeline import FramePipeline
from .webrtc import WebRTCManager

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


def create_app(config_path: Path | str = Path("data/config.json")) -> FastAPI:
    app = FastAPI(title="RevCam", version="0.1.0")

    config_manager = ConfigManager(Path(config_path))
    pipeline = FramePipeline(lambda: config_manager.get_orientation())
    camera: BaseCamera | None = None
    webrtc_manager: WebRTCManager | None = None
    logger = logging.getLogger(__name__)

    @app.on_event("startup")
    async def startup() -> None:  # pragma: no cover - framework hook
        nonlocal camera, webrtc_manager
        camera = create_camera()
        try:
            webrtc_manager = WebRTCManager(camera=camera, pipeline=pipeline)
        except RuntimeError as exc:
            logger.warning("WebRTC disabled: %s", exc)
            webrtc_manager = None

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
