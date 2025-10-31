from __future__ import annotations

from typing import Iterator

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rev_cam.engineering import DB_PATH, reset_work_centres
from rev_cam.engineering_api import (
    WorkCentreCreatePayload,
    WorkCentreUpdatePayload,
    create_work_centre,
    get_engineering_dashboard,
    list_work_centres,
    update_work_centre,
)
from rev_cam.sop import reset_sops
from rev_cam.sop_api import (
    SopPayload,
    SopUpdatePayload,
    create_procedure,
    list_procedures,
    update_procedure,
)


@pytest.fixture(autouse=True)
def reset_database() -> Iterator[None]:
    if DB_PATH.exists():
        DB_PATH.unlink()
    reset_work_centres()
    reset_sops()
    yield
    if DB_PATH.exists():
        DB_PATH.unlink()


class _Response:
    def __init__(self, status_code: int, payload: dict[str, object]):
        self.status_code = status_code
        self._payload = payload

    @property
    def content(self) -> bytes:
        return repr(self._payload).encode("utf-8")

    def json(self) -> dict[str, object]:
        return self._payload


class SimpleAPIClient:
    def get(self, path: str):
        if path.startswith("/engineering/work-centres"):
            query = None
            if "?name=" in path:
                query = path.split("?name=")[1]
            data = list_work_centres(name=query)
            return _Response(200, data)
        if path == "/engineering/dashboard":
            data = get_engineering_dashboard().model_dump()
            return _Response(200, data)
        if path.startswith("/sop/procedures"):
            query = None
            if "?work_centre=" in path:
                query = int(path.split("?work_centre=")[1])
            data = list_procedures(work_centre=query)
            return _Response(200, data)
        raise ValueError(f"Unsupported GET path: {path}")

    def post(self, path: str, json: dict[str, object]):
        try:
            if path == "/engineering/work-centres/":
                payload = WorkCentreCreatePayload(**json)
                result = create_work_centre(payload)
                return _Response(201, result.model_dump())
            if path == "/sop/procedures/":
                payload = SopPayload(**json)
                result = create_procedure(payload)
                return _Response(201, result.model_dump())
        except (HTTPException, ValidationError) as exc:
            detail = getattr(exc, "detail", None) or str(exc)
            status = getattr(exc, "status_code", 400)
            return _Response(status, {"detail": detail})
        raise ValueError(f"Unsupported POST path: {path}")

    def patch(self, path: str, json: dict[str, object]):
        try:
            if path.startswith("/engineering/work-centres/"):
                work_centre_id = int(path.rstrip("/").split("/")[-1])
                payload = WorkCentreUpdatePayload(**json)
                result = update_work_centre(work_centre_id, payload)
                return _Response(200, result.model_dump())
            if path.startswith("/sop/procedures/"):
                sop_id = int(path.rstrip("/").split("/")[-1])
                payload = SopUpdatePayload(**json)
                result = update_procedure(sop_id, payload)
                return _Response(200, result.model_dump())
        except (HTTPException, ValidationError) as exc:
            detail = getattr(exc, "detail", None) or str(exc)
            status = getattr(exc, "status_code", 400)
            return _Response(status, {"detail": detail})
        raise ValueError(f"Unsupported PATCH path: {path}")


@pytest.fixture()
def client() -> Iterator[SimpleAPIClient]:
    yield SimpleAPIClient()
