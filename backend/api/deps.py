"""
FastAPI dependencies for authentication and profile scoping.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request

from backend.auth.owner import SESSION_COOKIE, AuthContext, owner_service


def extract_token(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        return token
    # Bearer tokens are accepted for scripted API clients.
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return None


async def get_auth(request: Request) -> AuthContext:
    """
    Requires the install to be unlocked by the owner.
    401 codes: 'unconfigured' (no passphrase yet) or 'auth_required' (locked / bad token).
    """
    if not await owner_service.is_configured():
        raise HTTPException(
            status_code=401,
            detail={"error": "Owner passphrase not configured yet.", "code": "unconfigured"}
        )
    ctx = await owner_service.verify_session_token(extract_token(request))
    if ctx is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "Unlock Pitch Coach to continue.", "code": "auth_required"}
        )
    return ctx


async def get_profile_id(ctx: AuthContext = Depends(get_auth)) -> str:
    """Requires an active profile. 401 code 'profile_required' means: show the profile picker."""
    from backend.storage.profile_repository import profile_repository

    if not ctx.profile_id or not await profile_repository.get(ctx.profile_id):
        raise HTTPException(
            status_code=401,
            detail={"error": "Choose a profile to continue.", "code": "profile_required"}
        )
    return ctx.profile_id
