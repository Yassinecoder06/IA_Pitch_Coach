"""Rehearsals: create before recording, then list, read the report, replay audio, re-analyse, Q&A."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.deps import get_auth, get_profile_id
from backend.auth.owner import AuthContext
from backend.rehearsals import privacy
from backend.rehearsals.compare import compare
from backend.rehearsals.goals import GOAL_METRICS, OPS, goal_service
from backend.rehearsals.service import rehearsal_service
from backend.storage.session_repository import session_repository

router = APIRouter(prefix="/api/rehearsals", tags=["rehearsals"])


class RehearsalCreate(BaseModel):
    title: str = Field(default="Rehearsal", min_length=1, max_length=200)
    deck_id: Optional[str] = None
    target_seconds: Optional[int] = Field(default=None, ge=10, le=3600)


@router.post("")
async def create_rehearsal(payload: RehearsalCreate, profile_id: str = Depends(get_profile_id)):
    try:
        return await rehearsal_service.create(profile_id, payload.title.strip(), payload.deck_id, payload.target_seconds)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("")
async def list_rehearsals(deck_id: Optional[str] = None, profile_id: str = Depends(get_profile_id)):
    return {"rehearsals": await rehearsal_service.list(profile_id, deck_id)}


class GoalCreate(BaseModel):
    metric: str
    op: str = "gte"
    target: float
    target_max: Optional[float] = None


class PrivacyPayload(BaseModel):
    keep_recordings: bool = True
    recording_retention_days: Optional[int] = Field(default=None, ge=1, le=3650)


# Static paths are declared before "/{rehearsal_id}" so they are not captured by it.
@router.get("/compare")
async def compare_rehearsals(a: str, b: str, profile_id: str = Depends(get_profile_id)):
    ra = await rehearsal_service.get(a, profile_id)
    rb = await rehearsal_service.get(b, profile_id)
    if not ra or not rb:
        raise HTTPException(status_code=404, detail="Rehearsal not found")
    if ra["status"] != "done" or rb["status"] != "done":
        raise HTTPException(status_code=409, detail="Both rehearsals need a finished report")
    return compare(ra, rb)


@router.get("/goals")
async def list_goals(profile_id: str = Depends(get_profile_id)):
    return {"goals": await goal_service.list(profile_id), "metrics": list(GOAL_METRICS), "ops": list(OPS)}


@router.post("/goals")
async def create_goal(payload: GoalCreate, profile_id: str = Depends(get_profile_id)):
    try:
        return await goal_service.create(profile_id, payload.metric, payload.op, payload.target, payload.target_max)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.delete("/goals/{goal_id}")
async def delete_goal(goal_id: str, profile_id: str = Depends(get_profile_id)):
    if not await goal_service.delete(profile_id, goal_id):
        raise HTTPException(status_code=404, detail="Goal not found")
    return {"ok": True}


@router.get("/privacy")
async def get_privacy(ctx: AuthContext = Depends(get_auth)):
    return await privacy.get_settings()


@router.put("/privacy")
async def save_privacy(payload: PrivacyPayload, ctx: AuthContext = Depends(get_auth)):
    return await privacy.save_settings(payload.keep_recordings, payload.recording_retention_days)


@router.get("/{rehearsal_id}")
async def get_rehearsal(rehearsal_id: str, profile_id: str = Depends(get_profile_id)):
    r = await rehearsal_service.get(rehearsal_id, profile_id)
    if not r:
        raise HTTPException(status_code=404, detail="Rehearsal not found")
    return r


@router.delete("/{rehearsal_id}")
async def delete_rehearsal(rehearsal_id: str, profile_id: str = Depends(get_profile_id)):
    if not await rehearsal_service.delete(rehearsal_id, profile_id):
        raise HTTPException(status_code=404, detail="Rehearsal not found")
    return {"ok": True}


@router.get("/{rehearsal_id}/audio")
async def rehearsal_audio(rehearsal_id: str, profile_id: str = Depends(get_profile_id)):
    path = await rehearsal_service.audio_path(rehearsal_id, profile_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="No recording kept for this rehearsal")
    media = "audio/ogg" if path.suffix == ".ogg" else "audio/wav"
    return FileResponse(path, media_type=media)


@router.delete("/{rehearsal_id}/audio")
async def delete_rehearsal_audio(rehearsal_id: str, profile_id: str = Depends(get_profile_id)):
    """Deletes only the recording; the report stays."""
    if not await privacy.drop_recording(rehearsal_id, profile_id):
        raise HTTPException(status_code=404, detail="No recording kept for this rehearsal")
    return {"ok": True}


@router.post("/{rehearsal_id}/reanalyse")
async def reanalyse(rehearsal_id: str, profile_id: str = Depends(get_profile_id)):
    r = await rehearsal_service.reanalyse(rehearsal_id, profile_id)
    if not r:
        raise HTTPException(status_code=409, detail="Rehearsal not found or still processing")
    return r


@router.post("/{rehearsal_id}/qa")
async def start_qa(rehearsal_id: str, profile_id: str = Depends(get_profile_id)):
    """Opens an investor Q&A coaching session that targets this rehearsal's weak points."""
    ctx = await rehearsal_service.qa_context(rehearsal_id, profile_id)
    if not ctx:
        raise HTTPException(status_code=409, detail="The rehearsal report is not ready yet")
    session = await session_repository.create_session(
        profile_id, title=ctx["title"], mode="investor_qa", context_md=ctx["context_md"]
    )
    return {"session_id": session["id"]}
