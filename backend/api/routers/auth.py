"""
Owner authentication router adapted from Muse.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.api.deps import extract_token, get_auth
from backend.auth.owner import SESSION_COOKIE, SESSION_TTL_SECONDS, AuthContext, owner_service
from backend.auth.ratelimit import Backoff
from backend.config.settings import settings
from backend.storage.profile_repository import profile_repository

router = APIRouter(prefix="/api/auth", tags=["auth"])

login_backoff = Backoff(free_attempts=5)


class SetupRequest(BaseModel):
    passphrase: str = Field(min_length=8)
    setupCode: str


class LoginRequest(BaseModel):
    passphrase: str


class ChangePassphraseRequest(BaseModel):
    current: str
    new: str = Field(min_length=8)


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="strict",
        secure=settings.secure_cookies,
        path="/"
    )


async def issue_session(response: Response, profile_id: Optional[str] = None) -> None:
    """Signs a new session cookie; auto-selects the profile when there is only one without a PIN."""
    if profile_id is None:
        profile_id = await profile_repository.sole_unlocked_profile()
    if profile_id:
        await profile_repository.touch(profile_id)
    set_session_cookie(response, await owner_service.create_session_token(profile_id))


@router.get("/owner")
async def owner_status(request: Request):
    """Whether the coach is configured, unlocked, and which profile is active."""
    configured = await owner_service.is_configured()
    ctx = await owner_service.verify_session_token(extract_token(request)) if configured else None
    profile = await profile_repository.get(ctx.profile_id) if ctx and ctx.profile_id else None
    return {
        "configured": configured,
        "authenticated": ctx is not None,
        "profile": profile,
    }


@router.post("/owner/setup")
async def setup_owner(payload: SetupRequest, response: Response):
    """First-run setup with setup code and passphrase."""
    try:
        await owner_service.setup(payload.passphrase, payload.setupCode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    await issue_session(response)
    return {"authenticated": True}


@router.post("/owner/login")
async def login_owner(payload: LoginRequest, response: Response):
    """Unlock application with owner passphrase."""
    retry = login_backoff.retry_after()
    if retry:
        raise HTTPException(status_code=429, detail=f"Too many failed attempts. Try again in {retry}s.")

    if not await owner_service.verify_passphrase(payload.passphrase):
        login_backoff.fail()
        raise HTTPException(status_code=401, detail="Passphrase is not correct.")

    login_backoff.succeed()
    await issue_session(response)
    return {"authenticated": True}


@router.post("/owner/passphrase")
async def change_passphrase(payload: ChangePassphraseRequest, response: Response, ctx: AuthContext = Depends(get_auth)):
    """Change the passphrase. Every other browser is signed out; this one gets a fresh cookie."""
    retry = login_backoff.retry_after()
    if retry:
        raise HTTPException(status_code=429, detail=f"Too many failed attempts. Try again in {retry}s.")
    if not await owner_service.change_passphrase(payload.current, payload.new):
        login_backoff.fail()
        raise HTTPException(status_code=400, detail="Current passphrase is not correct.")
    login_backoff.succeed()
    set_session_cookie(response, await owner_service.create_session_token(ctx.profile_id))
    return {"ok": True}


@router.post("/owner/logout")
async def logout_owner(response: Response):
    """Lock application and clear session cookie."""
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"authenticated": False}
