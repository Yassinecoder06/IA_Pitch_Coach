"""
Owner authentication router adapted from Muse.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.auth.owner import owner_service, SESSION_COOKIE, SESSION_TTL_SECONDS
from backend.api.deps import extract_token

# Failed-login backoff: after FREE_ATTEMPTS failures, each further failure doubles the lockout.
FREE_ATTEMPTS = 5
MAX_LOCKOUT_SECONDS = 15 * 60
_failed_attempts = 0
_locked_until = 0.0

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SetupRequest(BaseModel):
    passphrase: str = Field(min_length=6)
    setupCode: str


class LoginRequest(BaseModel):
    passphrase: str


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


@router.get("/owner")
async def owner_status(request: Request):
    """Returns whether the coach is configured and if the current client is authenticated."""
    configured = owner_service.is_configured()
    token = extract_token(request)
    authenticated = bool(configured and owner_service.verify_session_token(token))
    return {
        "configured": configured,
        "authenticated": authenticated
    }


@router.post("/owner/setup")
async def setup_owner(payload: SetupRequest, response: Response):
    """First-run setup with setup code and passphrase."""
    try:
        owner_service.setup(payload.passphrase, payload.setupCode)
        set_session_cookie(response, owner_service.create_session_token())
        return {"authenticated": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/owner/login")
async def login_owner(payload: LoginRequest, response: Response):
    """Unlock application with owner passphrase."""
    global _failed_attempts, _locked_until

    remaining = _locked_until - time.monotonic()
    if remaining > 0:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {int(remaining) + 1}s."
        )

    if not owner_service.verify_passphrase(payload.passphrase):
        _failed_attempts += 1
        if _failed_attempts >= FREE_ATTEMPTS:
            lockout = min(MAX_LOCKOUT_SECONDS, 2 ** (_failed_attempts - FREE_ATTEMPTS + 1))
            _locked_until = time.monotonic() + lockout
        raise HTTPException(status_code=401, detail="Passphrase is not correct.")

    _failed_attempts = 0
    _locked_until = 0.0
    set_session_cookie(response, owner_service.create_session_token())
    return {"authenticated": True}


@router.post("/owner/logout")
async def logout_owner(response: Response):
    """Lock application and clear session cookie."""
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"authenticated": False}
