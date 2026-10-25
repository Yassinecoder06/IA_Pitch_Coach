"""
Local speaker profiles: list, create, edit, delete and switch.
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from backend.api.deps import get_auth
from backend.api.routers.auth import issue_session
from backend.auth.owner import AuthContext
from backend.auth.ratelimit import Backoff
from backend.storage.profile_repository import profile_repository

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

pin_backoff = Backoff(free_attempts=5)
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
PIN_RE = re.compile(r"^\d{4}$")


def _check_pin(value: Optional[str]) -> Optional[str]:
    if value not in (None, "") and not PIN_RE.match(value):
        raise ValueError("PIN must be 4 digits.")
    return value


class ProfilePayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    color: str = "#a855f7"
    pin: Optional[str] = None

    @field_validator("color")
    @classmethod
    def _color(cls, v: str) -> str:
        if not COLOR_RE.match(v):
            raise ValueError("Color must look like #a855f7.")
        return v

    @field_validator("pin")
    @classmethod
    def _pin(cls, v: Optional[str]) -> Optional[str]:
        return _check_pin(v)


class ProfileUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    color: Optional[str] = None
    pin: Optional[str] = None  # "" removes the PIN

    @field_validator("color")
    @classmethod
    def _color(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not COLOR_RE.match(v):
            raise ValueError("Color must look like #a855f7.")
        return v

    @field_validator("pin")
    @classmethod
    def _pin(cls, v: Optional[str]) -> Optional[str]:
        return _check_pin(v)


class SelectPayload(BaseModel):
    pin: Optional[str] = None


@router.get("")
async def list_profiles(ctx: AuthContext = Depends(get_auth)):
    return {"profiles": await profile_repository.list_profiles(), "active_id": ctx.profile_id}


@router.post("")
async def create_profile(payload: ProfilePayload, ctx: AuthContext = Depends(get_auth)):
    return await profile_repository.create(payload.name.strip(), payload.color, payload.pin or None)


@router.patch("/{profile_id}")
async def update_profile(profile_id: str, payload: ProfileUpdate, ctx: AuthContext = Depends(get_auth)):
    updated = await profile_repository.update(
        profile_id,
        name=payload.name.strip() if payload.name else None,
        color=payload.color,
        pin=payload.pin,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Profile not found")
    return updated


@router.delete("/{profile_id}")
async def delete_profile(profile_id: str, response: Response, ctx: AuthContext = Depends(get_auth)):
    """Deletes a profile and all its data. The last profile cannot be deleted."""
    if await profile_repository.count() <= 1:
        raise HTTPException(status_code=400, detail="Keep at least one profile.")
    if not await profile_repository.delete(profile_id):
        raise HTTPException(status_code=404, detail="Profile not found")
    if ctx.profile_id == profile_id:
        await issue_session(response, profile_id=None)
    return {"ok": True}


@router.post("/{profile_id}/select")
async def select_profile(profile_id: str, payload: SelectPayload, response: Response, ctx: AuthContext = Depends(get_auth)):
    """Switches the active profile (checks the PIN if the profile has one)."""
    retry = pin_backoff.retry_after(profile_id)
    if retry:
        raise HTTPException(status_code=429, detail=f"Too many wrong PINs. Try again in {retry}s.")
    profile = await profile_repository.get(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if not await profile_repository.check_pin(profile_id, payload.pin):
        pin_backoff.fail(profile_id)
        raise HTTPException(status_code=403, detail="Wrong PIN.")
    pin_backoff.succeed(profile_id)
    await issue_session(response, profile_id=profile_id)
    return profile
