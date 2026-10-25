"""
Sessions and historical messages router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_profile_id
from backend.coaching.modes import normalize_mode
from backend.storage.session_repository import session_repository

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class CreateSessionPayload(BaseModel):
    title: str = "Startup Pitch Practice"
    mode: str = "pitch_analysis"


class UpdateModePayload(BaseModel):
    mode: str


class RenameSessionPayload(BaseModel):
    title: str


@router.get("")
async def list_sessions(limit: int = 50, profile_id: str = Depends(get_profile_id)):
    sessions = await session_repository.list_sessions(profile_id, limit=limit)
    return {"sessions": sessions}


@router.post("")
async def create_session(payload: CreateSessionPayload, profile_id: str = Depends(get_profile_id)):
    session = await session_repository.create_session(profile_id, title=payload.title, mode=normalize_mode(payload.mode))
    return session


@router.get("/progress")
async def get_overall_progress(profile_id: str = Depends(get_profile_id)):
    """Returns chronological score progression across all sessions."""
    progress = await session_repository.get_progress(profile_id)
    return {"progress": progress}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    message_limit: int = 100,
    profile_id: str = Depends(get_profile_id)
):
    session = await session_repository.get_session(session_id, profile_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await session_repository.get_recent_messages(session_id, profile_id, limit=message_limit)
    return {
        "session": session,
        "messages": messages
    }


@router.patch("/{session_id}/mode")
async def update_session_mode(
    session_id: str,
    payload: UpdateModePayload,
    profile_id: str = Depends(get_profile_id)
):
    if not await session_repository.update_mode(session_id, profile_id, normalize_mode(payload.mode)):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.patch("/{session_id}/rename")
async def rename_session(
    session_id: str,
    payload: RenameSessionPayload,
    profile_id: str = Depends(get_profile_id)
):
    if not await session_repository.rename_session(session_id, profile_id, payload.title.strip() or "Untitled Session"):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.delete("/{session_id}")
async def delete_session(session_id: str, profile_id: str = Depends(get_profile_id)):
    deleted = await session_repository.delete_session(session_id, profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.delete("/{session_id}/turns/last")
async def delete_last_turn(session_id: str, profile_id: str = Depends(get_profile_id)):
    """Deletes the latest coach and user turn to allow editing and regenerating."""
    deleted_count = await session_repository.delete_last_turn(session_id, profile_id)
    return {"deleted": deleted_count}
