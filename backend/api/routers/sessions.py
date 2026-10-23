"""
Sessions and historical messages router.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_current_user_id
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
async def list_sessions(limit: int = 50, user_id: str = Depends(get_current_user_id)):
    sessions = session_repository.list_sessions(user_id=user_id, limit=limit)
    return {"sessions": sessions}


@router.post("")
async def create_session(payload: CreateSessionPayload, user_id: str = Depends(get_current_user_id)):
    session = session_repository.create_session(title=payload.title, mode=payload.mode, user_id=user_id)
    return session


@router.get("/progress")
async def get_overall_progress(user_id: str = Depends(get_current_user_id)):
    """Returns chronological score progression across all sessions."""
    progress = session_repository.get_progress(user_id=user_id)
    return {"progress": progress}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    message_limit: int = 100,
    user_id: str = Depends(get_current_user_id)
):
    session = session_repository.get_session(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_repository.get_recent_messages(session_id, user_id, limit=message_limit)
    return {
        "session": session,
        "messages": messages
    }


@router.patch("/{session_id}/mode")
async def update_session_mode(
    session_id: str,
    payload: UpdateModePayload,
    user_id: str = Depends(get_current_user_id)
):
    session = session_repository.get_session(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session_repository.update_mode(session_id, user_id, payload.mode)
    return {"ok": True}


@router.patch("/{session_id}/rename")
async def rename_session(
    session_id: str,
    payload: RenameSessionPayload,
    user_id: str = Depends(get_current_user_id)
):
    session = session_repository.get_session(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session_repository.rename_session(session_id, user_id, payload.title.strip() or "Untitled Session")
    return {"ok": True}


@router.delete("/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    deleted = session_repository.delete_session(session_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.delete("/{session_id}/turns/last")
async def delete_last_turn(session_id: str, user_id: str = Depends(get_current_user_id)):
    """Deletes the latest coach and user turn to allow editing and regenerating."""
    deleted_count = session_repository.delete_last_turn(session_id, user_id)
    return {"deleted": deleted_count}
