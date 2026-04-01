"""Supabase client factory for session persistence."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Ensure direct module usage picks up local environment file priority.
root_dir = Path(__file__).resolve().parents[2]
env_local_path = root_dir / ".env.local"
env_path = root_dir / ".env"

if env_local_path.exists():
    load_dotenv(env_local_path, override=True)
if env_path.exists():
    load_dotenv(env_path, override=False)

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover
    Client = None  # type: ignore
    create_client = None  # type: ignore


class SupabaseClientFactory:
    """Creates and caches a Supabase client when configured."""

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


def get_supabase_client() -> Optional[Client]:
    """Get a configured Supabase client or None when unavailable."""
    return SupabaseClientFactory.get_client()
