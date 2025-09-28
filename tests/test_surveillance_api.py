from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rev_cam.app import create_app
from rev_cam.surveillance import ModeManager, SurveillanceManager


def build_app(tmp_path: Path) -> tuple[FastAPI, SurveillanceManager, ModeManager]:
    surveillance_root = tmp_path / "surv"
    mode_state = tmp_path / "mode.json"
    lock_file = tmp_path / "camera.lock"

    surveillance_manager = SurveillanceManager(base_path=surveillance_root)
    mode_manager = ModeManager(lock_path=lock_file, state_path=mode_state)

    app = create_app(
        config_path=tmp_path / "config.json",
        surveillance_manager=surveillance_manager,
        mode_manager=mode_manager,
    )
    return app, surveillance_manager, mode_manager


def test_surveillance_endpoints(tmp_path):
    app, manager, mode_manager = build_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/mode")
        assert response.status_code == 200
        assert response.json() == {"mode": "idle"}

        response = client.post("/api/mode", json={"mode": "surveillance"})
        assert response.status_code == 200
        assert response.json()["mode"] == "surveillance"
        assert mode_manager.current_mode() == "surveillance"

        response = client.get("/api/surv/settings")
        assert response.status_code == 200
        assert "pre_roll_s" in response.json()

        response = client.put("/api/surv/settings", json={"storage_max_size_gb": 0.1})
        assert response.status_code == 200
        assert response.json()["storage_max_size_gb"] == 0.1

        response = client.post("/api/surv/test-motion")
        assert response.status_code == 200
        clip = response.json()["clip"]
        clip_id = clip["id"]

        response = client.get("/api/surv/clips")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == clip_id

        response = client.get(f"/api/surv/clips/{clip_id}")
        assert response.status_code == 200
        assert response.json()["id"] == clip_id

        response = client.get(f"/api/surv/clips/{clip_id}/stream")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("video/")

        response = client.get(f"/api/surv/clips/{clip_id}/thumb")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/")

        response = client.post("/api/surv/clips/export", json={"ids": [clip_id]})
        assert response.status_code == 200
        zip_url = response.json()["zip_url"]
        assert zip_url.endswith(".zip")

        response = client.get(zip_url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        response = client.delete("/api/surv/clips", json={"ids": [clip_id]})
        assert response.status_code == 200
        assert response.json()["removed"] == 1

        response = client.post("/api/surv/purge")
        assert response.status_code == 200
        assert response.json()["purged"] >= 0

        # Switching to reversing should work once surveillance released.
        response = client.post("/api/mode", json={"mode": "reversing"})
        assert response.status_code == 200
        assert response.json()["mode"] == "reversing"
