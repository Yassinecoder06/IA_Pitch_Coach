"""
Owner authentication system adapted from Muse.
Provides local-first single-owner protection with scrypt passphrase hashing,
setup code for first-run claiming, and HMAC-signed session tokens.
"""

from __future__ import annotations

import os
import hmac
import base64
import hashlib
import secrets
import string
import time
from typing import Optional, Dict, Any, Tuple

from backend.config.settings import settings
from backend.storage.database import db, DEFAULT_LOCAL_USER_ID

SESSION_COOKIE = "pitch_coach_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


class OwnerService:
    def __init__(self):
        self._setup_code: Optional[str] = None
        self._cached_salt: Optional[str] = None

    def _generate_setup_code(self) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        part1 = "".join(secrets.choice(alphabet) for _ in range(4))
        part2 = "".join(secrets.choice(alphabet) for _ in range(4))
        return f"{part1}-{part2}"

    def get_or_create_setup_code(self) -> str:
        if not self._setup_code:
            self._setup_code = self._generate_setup_code()
        return self._setup_code

    def is_configured(self) -> bool:
        """Check if an owner passphrase has been configured."""
        row = db.fetch_one("SELECT id FROM app_owner WHERE id = 1")
        return bool(row)

    def _hash_passphrase(self, passphrase: str, salt: bytes) -> str:
        # Scrypt key derivation
        derived = hashlib.scrypt(
            passphrase.encode("utf-8"),
            salt=salt,
            n=32768,
            r=8,
            p=1,
            maxmem=64 * 1024 * 1024,
            dklen=32
        )
        return f"scrypt$32768$8$1${base64.urlsafe_b64encode(salt).decode('utf-8')}${base64.urlsafe_b64encode(derived).decode('utf-8')}"

    def setup(self, passphrase: str, setup_code: str) -> bool:
        """Initialize owner passphrase on first run using the setup code."""
        if self.is_configured():
            raise ValueError("Pitch Coach is already configured.")

        expected_code = self.get_or_create_setup_code().replace("-", "").upper()
        provided_code = setup_code.replace("-", "").strip().upper()

        if not hmac.compare_digest(expected_code, provided_code):
            raise ValueError("Invalid setup code.")

        salt = secrets.token_bytes(16)
        session_salt = secrets.token_urlsafe(32)
        hashed = self._hash_passphrase(passphrase, salt)

        now = datetime_now_iso()
        db.execute(
            """INSERT INTO app_owner (id, passphrase_hash, session_salt, created_at, updated_at)
               VALUES (1, ?, ?, ?, ?)""",
            (hashed, session_salt, now, now)
        )
        self._cached_salt = session_salt
        self._setup_code = None
        return True

    def verify_passphrase(self, passphrase: str) -> bool:
        """Verify owner passphrase."""
        row = db.fetch_one("SELECT passphrase_hash, session_salt FROM app_owner WHERE id = 1")
        if not row:
            return False

        stored = row["passphrase_hash"]
        parts = stored.split("$")
        if len(parts) != 6 or parts[0] != "scrypt":
            return False

        salt = base64.urlsafe_b64decode(parts[4])
        expected_key = base64.urlsafe_b64decode(parts[5])

        actual_key = hashlib.scrypt(
            passphrase.encode("utf-8"),
            salt=salt,
            n=int(parts[1]),
            r=int(parts[2]),
            p=int(parts[3]),
            maxmem=64 * 1024 * 1024,
            dklen=len(expected_key)
        )
        return hmac.compare_digest(actual_key, expected_key)

    def _get_signing_key(self) -> bytes:
        row = db.fetch_one("SELECT session_salt FROM app_owner WHERE id = 1")
        salt = row["session_salt"] if row else "unconfigured-salt"
        return hashlib.sha256(f"{settings.secret_key}:{salt}".encode("utf-8")).digest()

    def create_session_token(self, user_id: str = DEFAULT_LOCAL_USER_ID) -> str:
        """Create a signed HMAC session token."""
        exp = int(time.time()) + SESSION_TTL_SECONDS
        payload = f"{user_id}:{exp}"
        payload_b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")
        sig = hmac.new(self._get_signing_key(), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload_b64}.{sig}"

    def verify_session_token(self, token: Optional[str]) -> Optional[str]:
        """Verify token and return user_id, or None if invalid."""
        if not token or "." not in token:
            return None

        parts = token.split(".")
        if len(parts) != 2:
            return None

        payload_b64, sig = parts[0], parts[1]
        expected_sig = hmac.new(self._get_signing_key(), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(sig, expected_sig):
            return None

        try:
            payload = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
            user_id, exp_str = payload.split(":")
            if int(exp_str) < int(time.time()):
                return None
            return user_id
        except Exception:
            return None


def datetime_now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


owner_service = OwnerService()
