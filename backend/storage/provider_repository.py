"""
AI Provider repository with encrypted key storage adapted from Muse.
Provides multi-AI provider management and fallback configuration.
"""

from __future__ import annotations

import os
import json
import uuid
import base64
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from backend.config.settings import settings
from backend.storage.database import db, DEFAULT_LOCAL_USER_ID


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_key_hint(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    key = key.strip()
    if len(key) <= 8:
        return "..."
    return f"...{key[-4:]}"


def _get_encryption_key() -> bytes:
    return hashlib.sha256(f"provider_vault:{settings.secret_key}".encode("utf-8")).digest()


def encrypt_secret(secret: Optional[str]) -> Optional[str]:
    if not secret:
        return None
    aes = AESGCM(_get_encryption_key())
    nonce = os.urandom(12)
    ciphertext = aes.encrypt(nonce, secret.encode("utf-8"), None)
    return base64.urlsafe_b64encode(nonce + ciphertext).decode("utf-8")


def decrypt_secret(encrypted: Optional[str]) -> Optional[str]:
    if not encrypted:
        return None
    try:
        raw = base64.urlsafe_b64decode(encrypted.encode("utf-8"))
        nonce, ciphertext = raw[:12], raw[12:]
        aes = AESGCM(_get_encryption_key())
        return aes.decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception:
        return None


class ProviderRepository:
    """Manages configured AI providers, keys, and model preferences."""

    def list_providers(self, user_id: str = DEFAULT_LOCAL_USER_ID) -> List[Dict[str, Any]]:
        rows = db.fetch_all(
            """SELECT id, user_id, kind, name, base_url, key_hint, models_json, enabled, last_fetched_at, last_error, created_at, updated_at
               FROM ai_providers WHERE user_id = ? ORDER BY created_at ASC""",
            (user_id,)
        )
        providers = []
        for r in rows:
            item = dict(r)
            try:
                item["models"] = json.loads(item["models_json"]) if item.get("models_json") else []
            except Exception:
                item["models"] = []
            item["has_key"] = bool(item.get("key_hint"))
            providers.append(item)
        return providers

    def get_provider(self, provider_id: str, user_id: str = DEFAULT_LOCAL_USER_ID) -> Optional[Dict[str, Any]]:
        row = db.fetch_one(
            """SELECT id, user_id, kind, name, base_url, api_key_encrypted, key_hint, models_json, enabled, last_fetched_at, last_error, created_at, updated_at
               FROM ai_providers WHERE id = ? AND user_id = ?""",
            (provider_id, user_id)
        )
        if not row:
            return None
        item = dict(row)
        try:
            item["models"] = json.loads(item["models_json"]) if item.get("models_json") else []
        except Exception:
            item["models"] = []
        item["api_key"] = decrypt_secret(item.get("api_key_encrypted"))
        item["has_key"] = bool(item.get("key_hint"))
        return item

    def create_provider(
        self,
        kind: str,
        name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: str = DEFAULT_LOCAL_USER_ID
    ) -> Dict[str, Any]:
        provider_id = str(uuid.uuid4())
        encrypted_key = encrypt_secret(api_key) if api_key else None
        hint = make_key_hint(api_key)
        created_at = now_iso()

        db.execute(
            """INSERT INTO ai_providers (id, user_id, kind, name, base_url, api_key_encrypted, key_hint, models_json, enabled, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, '[]', 1, ?, ?)""",
            (provider_id, user_id, kind, name, base_url, encrypted_key, hint, created_at, created_at)
        )
        return self.get_provider(provider_id, user_id) or {}

    def update_provider(
        self,
        provider_id: str,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
        user_id: str = DEFAULT_LOCAL_USER_ID
    ) -> Optional[Dict[str, Any]]:
        existing = self.get_provider(provider_id, user_id)
        if not existing:
            return None

        new_name = name if name is not None else existing["name"]
        new_base_url = base_url if base_url is not None else existing["base_url"]
        new_enabled = int(enabled) if enabled is not None else existing["enabled"]

        if api_key is not None:
            encrypted_key = encrypt_secret(api_key) if api_key else None
            hint = make_key_hint(api_key)
            db.execute(
                """UPDATE ai_providers
                   SET name = ?, base_url = ?, api_key_encrypted = ?, key_hint = ?, enabled = ?, updated_at = ?
                   WHERE id = ? AND user_id = ?""",
                (new_name, new_base_url, encrypted_key, hint, new_enabled, now_iso(), provider_id, user_id)
            )
        else:
            db.execute(
                """UPDATE ai_providers
                   SET name = ?, base_url = ?, enabled = ?, updated_at = ?
                   WHERE id = ? AND user_id = ?""",
                (new_name, new_base_url, new_enabled, now_iso(), provider_id, user_id)
            )

        return self.get_provider(provider_id, user_id)

    def save_models(
        self,
        provider_id: str,
        models: List[Dict[str, Any]],
        error: Optional[str] = None,
        user_id: str = DEFAULT_LOCAL_USER_ID
    ) -> None:
        db.execute(
            """UPDATE ai_providers
               SET models_json = ?, last_error = ?, last_fetched_at = ?, updated_at = ?
               WHERE id = ? AND user_id = ?""",
            (json.dumps(models), error, now_iso(), now_iso(), provider_id, user_id)
        )

    def delete_provider(self, provider_id: str, user_id: str = DEFAULT_LOCAL_USER_ID) -> bool:
        db.execute("DELETE FROM ai_providers WHERE id = ? AND user_id = ?", (provider_id, user_id))
        return True

    def get_preferences(self, user_id: str = DEFAULT_LOCAL_USER_ID) -> Dict[str, Any]:
        row = db.fetch_one("SELECT * FROM ai_preferences WHERE user_id = ?", (user_id,))
        if not row:
            return {
                "chat_models": [],
                "fallback_models": [],
                "tts_ref": "builtin:piper",
                "tts_voice": "en_US-lessac-medium",
                "stt_ref": "builtin:whisper"
            }
        item = dict(row)
        try:
            chat_models = json.loads(item["chat_models_json"]) if item.get("chat_models_json") else []
        except Exception:
            chat_models = []
        try:
            fallback_models = json.loads(item["fallback_models_json"]) if item.get("fallback_models_json") else []
        except Exception:
            fallback_models = []

        return {
            "chat_models": chat_models,
            "fallback_models": fallback_models,
            "tts_ref": item.get("tts_ref") or "builtin:piper",
            "tts_voice": item.get("tts_voice") or "en_US-lessac-medium",
            "stt_ref": item.get("stt_ref") or "builtin:whisper"
        }

    def save_preferences(
        self,
        chat_models: List[str],
        fallback_models: List[str],
        tts_ref: Optional[str] = None,
        tts_voice: Optional[str] = None,
        stt_ref: Optional[str] = None,
        user_id: str = DEFAULT_LOCAL_USER_ID
    ) -> Dict[str, Any]:
        existing = db.fetch_one("SELECT user_id FROM ai_preferences WHERE user_id = ?", (user_id,))
        now = now_iso()
        chat_json = json.dumps(chat_models)
        fallback_json = json.dumps(fallback_models)

        if existing:
            db.execute(
                """UPDATE ai_preferences
                   SET chat_models_json = ?, fallback_models_json = ?, tts_ref = ?, tts_voice = ?, stt_ref = ?, updated_at = ?
                   WHERE user_id = ?""",
                (chat_json, fallback_json, tts_ref, tts_voice, stt_ref, now, user_id)
            )
        else:
            db.execute(
                """INSERT INTO ai_preferences (user_id, chat_models_json, fallback_models_json, tts_ref, tts_voice, stt_ref, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, chat_json, fallback_json, tts_ref, tts_voice, stt_ref, now)
            )
        return self.get_preferences(user_id)


provider_repository = ProviderRepository()
