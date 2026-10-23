"""
FastAPI dependencies for authentication and authorization.
"""

from __future__ import annotations

from typing import Optional
from fastapi import Request, HTTPException, Depends

from backend.auth.owner import owner_service, SESSION_COOKIE
from backend.storage.database import DEFAULT_LOCAL_USER_ID


def extract_token(request: Request) -> Optional[str]:
    # Check cookie first
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        return token
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return None


def get_current_user_id(request: Request) -> str:
    """
    Dependency that enforces owner authentication.
    If the app is unconfigured, raises 401 with code 'unconfigured'.
    If the token is missing or invalid, raises 401 with code 'auth_required'.
    """
    if not owner_service.is_configured():
        raise HTTPException(
            status_code=401,
            detail={"error": "Owner passphrase not configured yet.", "code": "unconfigured"}
        )

    token = extract_token(request)
    user_id = owner_service.verify_session_token(token)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={"error": "Unlock Pitch Coach to continue.", "code": "auth_required"}
        )
    return user_id


def get_optional_user_id(request: Request) -> Optional[str]:
    token = extract_token(request)
    if not token:
        return None
    return owner_service.verify_session_token(token)
