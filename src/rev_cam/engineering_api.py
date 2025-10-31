from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .engineering import WORK_CENTRES, WorkCentre

router = APIRouter(prefix="/engineering", tags=["engineering"])


class WorkCentreCreatePayload(BaseModel):
    name: str = Field(..., min_length=1)
    code: str | None = Field(default=None)
    description: str | None = None
    notes: str | None = None


class WorkCentreUpdatePayload(BaseModel):
    name: str | None = None
    code: str | None = None
    description: str | None = None
    notes: str | None = None


class WorkCentreResponse(BaseModel):
    id: int
    name: str
    code: str | None
    description: str
    notes: str
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, model: WorkCentre) -> "WorkCentreResponse":
        return cls(**model.serialise())


@router.get("/work-centres", response_model=dict)
def list_work_centres(name: str | None = None) -> dict[str, object]:
    centres = [WorkCentreResponse.from_model(item).model_dump() for item in WORK_CENTRES.list(name_contains=name)]
    return {"count": len(centres), "results": centres}


@router.post("/work-centres", response_model=WorkCentreResponse, status_code=201)
def create_work_centre(payload: WorkCentreCreatePayload) -> WorkCentreResponse:
    try:
        work_centre = WORK_CENTRES.create(
            name=payload.name,
            code=payload.code,
            description=payload.description or "",
            notes=payload.notes or "",
        )
    except ValueError as exc:  # pragma: no cover - validated in tests
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkCentreResponse.from_model(work_centre)


@router.get("/work-centres/{work_centre_id}", response_model=WorkCentreResponse)
def get_work_centre(work_centre_id: int) -> WorkCentreResponse:
    work_centre = WORK_CENTRES.get(work_centre_id)
    if work_centre is None:
        raise HTTPException(status_code=404, detail="Work centre not found.")
    return WorkCentreResponse.from_model(work_centre)


@router.patch("/work-centres/{work_centre_id}", response_model=WorkCentreResponse)
def update_work_centre(work_centre_id: int, payload: WorkCentreUpdatePayload) -> WorkCentreResponse:
    try:
        work_centre = WORK_CENTRES.update(
            work_centre_id,
            **{key: value for key, value in payload.model_dump(exclude_unset=True).items()},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Work centre not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WorkCentreResponse.from_model(work_centre)
