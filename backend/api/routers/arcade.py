"""Training arcade: record runs, personal bests, streak and the daily drill queue."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import get_profile_id
from backend.arcade.service import arcade_service

router = APIRouter(prefix="/api/arcade", tags=["arcade"])


class RunIn(BaseModel):
    game: str
    difficulty: str = "medium"
    score: float = Field(ge=0, le=100000)
    stats: Optional[Dict[str, Any]] = None


@router.get("")
async def summary(profile_id: str = Depends(get_profile_id)):
    return await arcade_service.summary(profile_id)


@router.post("/runs")
async def record_run(payload: RunIn, profile_id: str = Depends(get_profile_id)):
    try:
        return await arcade_service.record(profile_id, payload.game, payload.difficulty, payload.score, payload.stats)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
