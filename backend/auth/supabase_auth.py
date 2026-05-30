"""Supabase authentication helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Header, HTTPException

# Load environment files with local override priority.
root_dir = Path(__file__).resolve().parents[2]
env_local_path = root_dir / ".env.local"
env_path = root_dir / ".env"

if env_local_path.exists():
    load_dotenv(env_local_path, override=True)
    if env_path.exists():
        load_dotenv(env_path, override=False)
elif env_path.exists():
    load_dotenv(env_path, override=True)

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover
    Client = None  # type: ignore
    create_client = None  # type: ignore


class SupabaseAuthClient:
    """Create a Supabase client for auth lookups."""

    _client: Optional[Client] = None

    @classmethod
    def get_client(cls) -> Optional[Client]:
        if cls._client is not None:
            return cls._client

        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_ANON_KEY", "").strip()

        if not url or not key or create_client is None:
            return None

        cls._client = create_client(url, key)
        return cls._client


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _normalize_user_payload(user: Any) -> Optional[Dict[str, Any]]:
    if user is None:
        return None
    if isinstance(user, dict):
        return user
    if hasattr(user, "model_dump"):
        return user.model_dump()
    if hasattr(user, "dict"):
        return user.dict()
    if hasattr(user, "__dict__"):
        return user.__dict__
    return {"id": getattr(user, "id", None)}


def get_user_from_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    client = SupabaseAuthClient.get_client()
    if client is None:
        return None
    try:
        response = client.auth.get_user(token)
    except Exception:
        return None

    user = getattr(response, "user", None)
    if user is None and isinstance(response, dict):
        user = response.get("user")
    return _normalize_user_payload(user)


def get_user_id_from_token(token: Optional[str]) -> Optional[str]:
    user = get_user_from_token(token)
    if not user:
        return None
    return user.get("id") or user.get("user_id")


def require_user_id(authorization: Optional[str] = Header(None)) -> str:
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    user_id = get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id
