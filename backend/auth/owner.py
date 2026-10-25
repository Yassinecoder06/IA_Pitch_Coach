"""
Owner authentication adapted from Muse.

One owner unlocks the install with a passphrase (claimed on first run with a
one-time setup code). The signed session token then optionally carries the
local profile that is currently active.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from backend.config.settings import settings
from backend.db import database
from backend.db.models import Owner
from backend.db.types import utcnow

SESSION_COOKIE = "pitch_coach_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days
NO_PROFILE = "-"


@dataclass(frozen=True)
class AuthContext:
    """A verified session: the instance is unlocked, and maybe a profile is active."""

    profile_id: Optional[str]


def hash_secret(secret: str, n: int = 32768) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(secret.encode("utf-8"), salt=salt, n=n, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=32)
    b64 = base64.urlsafe_b64encode
    return f"scrypt${n}$8$1${b64(salt).decode()}${b64(derived).decode()}"


def verify_secret(secret: str, stored: Optional[str]) -> bool:
    if not stored:
        return False
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    salt = base64.urlsafe_b64decode(parts[4])
    expected = base64.urlsafe_b64decode(parts[5])
    actual = hashlib.scrypt(
        secret.encode("utf-8"),
        salt=salt,
        n=int(parts[1]),
        r=int(parts[2]),
        p=int(parts[3]),
        maxmem=64 * 1024 * 1024,
        dklen=len(expected),
    )
    return hmac.compare_digest(actual, expected)


class OwnerService:
    def __init__(self):
        self._setup_code: Optional[str] = None
        self._session_salt: Optional[str] = None  # cached; rotated on passphrase change

    def _generate_setup_code(self) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        part1 = "".join(secrets.choice(alphabet) for _ in range(4))
        part2 = "".join(secrets.choice(alphabet) for _ in range(4))
        return f"{part1}-{part2}"

    def get_or_create_setup_code(self) -> str:
        if not self._setup_code:
            self._setup_code = self._generate_setup_code()
        return self._setup_code

    def reset_cache(self) -> None:
        self._setup_code = None
        self._session_salt = None

    async def is_configured(self) -> bool:
        async with database.session() as s:
            return await s.get(Owner, 1) is not None

    async def setup(self, passphrase: str, setup_code: str) -> bool:
        """Initialize owner passphrase on first run using the setup code."""
        if await self.is_configured():
            raise ValueError("Pitch Coach is already configured.")

        expected_code = self.get_or_create_setup_code().replace("-", "").upper()
        provided_code = setup_code.replace("-", "").strip().upper()
        if not hmac.compare_digest(expected_code, provided_code):
            raise ValueError("Invalid setup code.")

        session_salt = secrets.token_urlsafe(32)
        async with database.session() as s:
            s.add(Owner(id=1, passphrase_hash=hash_secret(passphrase), session_salt=session_salt))
        self._session_salt = session_salt
        self._setup_code = None
        return True

    async def verify_passphrase(self, passphrase: str) -> bool:
        async with database.session() as s:
            owner = await s.get(Owner, 1)
            return bool(owner) and verify_secret(passphrase, owner.passphrase_hash)

    async def change_passphrase(self, current: str, new: str) -> bool:
        """Replaces the passphrase and rotates the session salt, signing out every other browser."""
        async with database.session() as s:
            owner = await s.get(Owner, 1)
            if not owner or not verify_secret(current, owner.passphrase_hash):
                return False
            owner.passphrase_hash = hash_secret(new)
            owner.session_salt = secrets.token_urlsafe(32)
            owner.updated_at = utcnow()
            self._session_salt = owner.session_salt
        return True

    async def _get_signing_key(self) -> Optional[bytes]:
        if self._session_salt is None:
            async with database.session() as s:
                owner = await s.get(Owner, 1)
                if not owner:
                    return None
                self._session_salt = owner.session_salt
        return hashlib.sha256(f"{settings.secret_key}:{self._session_salt}".encode("utf-8")).digest()

    async def create_session_token(self, profile_id: Optional[str] = None) -> str:
        key = await self._get_signing_key()
        if key is None:
            raise ValueError("Pitch Coach is not configured.")
        exp = int(time.time()) + SESSION_TTL_SECONDS
        payload_b64 = base64.urlsafe_b64encode(f"{profile_id or NO_PROFILE}:{exp}".encode("utf-8")).decode("utf-8")
        sig = hmac.new(key, payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload_b64}.{sig}"

    async def verify_session_token(self, token: Optional[str]) -> Optional[AuthContext]:
        """Returns the session's AuthContext, or None if the token is missing, forged or expired."""
        if not token or token.count(".") != 1:
            return None
        key = await self._get_signing_key()
        if key is None:
            return None

        payload_b64, sig = token.split(".")
        expected_sig = hmac.new(key, payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        try:
            profile_id, exp_str = base64.urlsafe_b64decode(payload_b64).decode("utf-8").rsplit(":", 1)
            if int(exp_str) < int(time.time()):
                return None
        except Exception:
            return None
        return AuthContext(profile_id=None if profile_id == NO_PROFILE else profile_id)


owner_service = OwnerService()
