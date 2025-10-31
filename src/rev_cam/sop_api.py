from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .engineering import WORK_CENTRES
from .sop import SOPS, StandardOperatingProcedure

router = APIRouter(prefix="/sop", tags=["sop"])


class SopPayload(BaseModel):
    title: str = Field(..., min_length=1)
    reference: str = Field(..., min_length=1)
    work_centre: int = Field(..., ge=1)
    instructions: str = Field(..., min_length=1)
    revision: int = Field(default=1, ge=1)
    is_active: bool = True


class SopUpdatePayload(BaseModel):
    title: str | None = None
    reference: str | None = None
    work_centre: int | None = Field(default=None, ge=1)
    instructions: str | None = None
    revision: int | None = Field(default=None, ge=1)
    is_active: bool | None = None


class SopResponse(BaseModel):
    id: int
    title: str
    reference: str
    work_centre: int
    work_centre_name: str | None
    instructions: str
    revision: int
    is_active: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, model: StandardOperatingProcedure) -> "SopResponse":
        payload = model.serialise()
        payload["work_centre_name"] = payload.get("work_centre_name")
        return cls(**payload)


@router.get("/procedures", response_model=dict)
def list_procedures(work_centre: int | None = None) -> dict[str, object]:
    procedures = [
        SopResponse.from_model(item).model_dump()
        for item in SOPS.list(work_centre_id=work_centre)
    ]
    return {"count": len(procedures), "results": procedures}


@router.post("/procedures", response_model=SopResponse, status_code=201)
def create_procedure(payload: SopPayload) -> SopResponse:
    if WORK_CENTRES.get(payload.work_centre) is None:
        raise HTTPException(status_code=400, detail="Selected work centre does not exist.")
    try:
        sop = SOPS.create(
            title=payload.title,
            reference=payload.reference,
            work_centre_id=payload.work_centre,
            instructions=payload.instructions,
            revision=payload.revision,
            is_active=payload.is_active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SopResponse.from_model(sop)


@router.patch("/procedures/{sop_id}", response_model=SopResponse)
def update_procedure(sop_id: int, payload: SopUpdatePayload) -> SopResponse:
    data = payload.model_dump(exclude_unset=True)
    if "work_centre" in data and WORK_CENTRES.get(int(data["work_centre"])) is None:
        raise HTTPException(status_code=400, detail="Selected work centre does not exist.")
    try:
        sop = SOPS.update(
            sop_id,
            **{
                "work_centre_id" if key == "work_centre" else key: value
                for key, value in data.items()
            },
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="SOP not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SopResponse.from_model(sop)
