"""
AI Provider repository with encrypted key storage adapted from Muse.
Providers and model routing are install-wide: they hold the owner's API keys
and are shared by every local profile.
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import select

from backend.config.settings import LEGACY_DEFAULT_SECRET, settings
from backend.db import database
from backend.db.models import AiProvider, AiSettings, iso
from backend.db.types import utcnow

# Tasks that can be routed to their own preferred model.
AI_TASKS = ("chat", "judge", "deck_gen", "qa")
DEFAULT_TTS_VOICE = "en_US-lessac-medium"


def make_key_hint(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    key = key.strip()
    if len(key) <= 8:
        return "..."
    return f"...{key[-4:]}"


def _derive_key(secret: str) -> bytes:
    return hashlib.sha256(f"provider_vault:{secret}".encode("utf-8")).digest()


def _get_encryption_key() -> bytes:
    return _derive_key(settings.secret_key)


def encrypt_secret(secret: Optional[str]) -> Optional[str]:
    if not secret:
        return None
    aes = AESGCM(_get_encryption_key())
    nonce = os.urandom(12)
    ciphertext = aes.encrypt(nonce, secret.encode("utf-8"), None)
    return base64.urlsafe_b64encode(nonce + ciphertext).decode("utf-8")


def _decrypt_with(key: bytes, encrypted: str) -> Optional[str]:
    try:
        raw = base64.urlsafe_b64decode(encrypted.encode("utf-8"))
        nonce, ciphertext = raw[:12], raw[12:]
        return AESGCM(key).decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception:
        return None


def decrypt_secret(encrypted: Optional[str]) -> Optional[str]:
    if not encrypted:
        return None
    return _decrypt_with(_get_encryption_key(), encrypted)


async def migrate_legacy_encryption() -> int:
    """Re-encrypt provider keys that were stored under the old hardcoded default secret."""
    legacy_key = _derive_key(LEGACY_DEFAULT_SECRET)
    if legacy_key == _get_encryption_key():
        return 0
    migrated = 0
    async with database.session() as s:
        rows = (await s.execute(select(AiProvider).where(AiProvider.api_key_encrypted.is_not(None)))).scalars()
        for row in rows:
            if decrypt_secret(row.api_key_encrypted) is not None:
                continue
            plain = _decrypt_with(legacy_key, row.api_key_encrypted)
            if plain is None:
                continue
            row.api_key_encrypted = encrypt_secret(plain)
            migrated += 1
    return migrated


def _provider_dict(p: AiProvider, include_key: bool = False) -> Dict[str, Any]:
    item = {
        "id": p.id,
        "kind": p.kind,
        "name": p.name,
        "base_url": p.base_url,
        "key_hint": p.key_hint,
        "has_key": bool(p.key_hint),
        "models": p.models_json or [],
        "enabled": bool(p.enabled),
        "last_fetched_at": iso(p.last_fetched_at),
        "last_error": p.last_error,
        "created_at": iso(p.created_at),
        "updated_at": iso(p.updated_at),
    }
    if include_key:
        item["api_key"] = decrypt_secret(p.api_key_encrypted)
    return item


class ProviderRepository:
    """Manages configured AI providers, keys, and model preferences."""

    async def list_providers(self) -> List[Dict[str, Any]]:
        async with database.session() as s:
            rows = (await s.execute(select(AiProvider).order_by(AiProvider.created_at))).scalars().all()
            return [_provider_dict(p) for p in rows]

    async def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        async with database.session() as s:
            p = await s.get(AiProvider, provider_id)
            return _provider_dict(p, include_key=True) if p else None

    async def create_provider(
        self,
        kind: str,
        name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with database.session() as s:
            p = AiProvider(
                kind=kind,
                name=name,
                base_url=base_url,
                api_key_encrypted=encrypt_secret(api_key),
                key_hint=make_key_hint(api_key),
                models_json=[],
                enabled=True,
            )
            s.add(p)
            await s.flush()
            return _provider_dict(p, include_key=True)

    async def update_provider(
        self,
        provider_id: str,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        async with database.session() as s:
            p = await s.get(AiProvider, provider_id)
            if not p:
                return None
            if name is not None:
                p.name = name
            if base_url is not None:
                p.base_url = base_url or None
            if enabled is not None:
                p.enabled = enabled
            if api_key is not None:  # "" clears the key
                p.api_key_encrypted = encrypt_secret(api_key)
                p.key_hint = make_key_hint(api_key)
            await s.flush()
            return _provider_dict(p, include_key=True)

    async def save_models(self, provider_id: str, models: List[Dict[str, Any]], error: Optional[str] = None) -> None:
        async with database.session() as s:
            p = await s.get(AiProvider, provider_id)
            if p:
                p.models_json = models
                p.last_error = error
                p.last_fetched_at = utcnow()

    async def delete_provider(self, provider_id: str) -> bool:
        async with database.session() as s:
            p = await s.get(AiProvider, provider_id)
            if not p:
                return False
            await s.delete(p)
            prefs = await s.get(AiSettings, 1)
            if prefs:
                prefix = f"{provider_id}:"
                prefs.chat_models_json = [r for r in prefs.chat_models_json or [] if not r.startswith(prefix)]
                prefs.fallback_models_json = [r for r in prefs.fallback_models_json or [] if not r.startswith(prefix)]
                prefs.task_models_json = {
                    t: r for t, r in (prefs.task_models_json or {}).items() if not r.startswith(prefix)
                }
            return True

    async def get_preferences(self) -> Dict[str, Any]:
        async with database.session() as s:
            row = await s.get(AiSettings, 1)
            return {
                "chat_models": list(row.chat_models_json or []) if row else [],
                "fallback_models": list(row.fallback_models_json or []) if row else [],
                "task_models": dict(row.task_models_json or {}) if row else {},
                "tts_voice": (row.tts_voice if row else None) or DEFAULT_TTS_VOICE,
            }

    async def save_preferences(
        self,
        chat_models: List[str],
        fallback_models: List[str],
        task_models: Optional[Dict[str, str]] = None,
        tts_voice: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with database.session() as s:
            row = await s.get(AiSettings, 1)
            if not row:
                row = AiSettings(id=1)
                s.add(row)
            row.chat_models_json = chat_models
            row.fallback_models_json = fallback_models
            if task_models is not None:
                row.task_models_json = {t: r for t, r in task_models.items() if t in AI_TASKS and r}
            elif row.task_models_json is None:
                row.task_models_json = {}
            if tts_voice is not None:
                row.tts_voice = tts_voice
        return await self.get_preferences()


provider_repository = ProviderRepository()
