"""
Owner authentication router adapted from Muse.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel, Field

from backend.auth.owner import owner_service, SESSION_COOKIE, SESSION_TTL_SECONDS
from backend.api.deps import extract_token

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
        samesite="lax",
        secure=False,  # Local development / HTTP allowed
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
        token = owner_service.create_session_token()
        set_session_cookie(response, token)
        return {"authenticated": True, "token": token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/owner/login")
async def login_owner(payload: LoginRequest, response: Response):
    """Unlock application with owner passphrase."""
    if not owner_service.verify_passphrase(payload.passphrase):
        raise HTTPException(status_code=401, detail="Passphrase is not correct.")
    token = owner_service.create_session_token()
    set_session_cookie(response, token)
    return {"authenticated": True, "token": token}


@router.post("/owner/logout")
async def logout_owner(response: Response):
    """Lock application and clear session cookie."""
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"authenticated": False}
