"""Local speaker profiles. Deleting a profile cascades to its sessions, decks and rehearsals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import func, select

from backend.auth.owner import hash_secret, verify_secret
from backend.db import database
from backend.db.models import Profile
from backend.db.types import utcnow

# PINs are short, so a cheaper scrypt cost keeps profile switching snappy; attempts are rate-limited.
PIN_SCRYPT_N = 2 ** 14


class ProfileRepository:
    async def list_profiles(self) -> List[Dict[str, Any]]:
        async with database.session() as s:
            rows = (await s.execute(select(Profile).order_by(Profile.created_at))).scalars()
            return [p.to_dict() for p in rows]

    async def count(self) -> int:
        async with database.session() as s:
            return (await s.execute(select(func.count()).select_from(Profile))).scalar_one()

    async def get(self, profile_id: str) -> Optional[Dict[str, Any]]:
        async with database.session() as s:
            p = await s.get(Profile, profile_id)
            return p.to_dict() if p else None

    async def create(self, name: str, color: str, pin: Optional[str] = None) -> Dict[str, Any]:
        async with database.session() as s:
            p = Profile(name=name, color=color, pin_hash=hash_secret(pin, PIN_SCRYPT_N) if pin else None)
            s.add(p)
            await s.flush()
            return p.to_dict()

    async def update(
        self, profile_id: str, name: Optional[str] = None, color: Optional[str] = None, pin: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """`pin=""` removes the PIN; `pin=None` leaves it unchanged."""
        async with database.session() as s:
            p = await s.get(Profile, profile_id)
            if not p:
                return None
            if name is not None:
                p.name = name
            if color is not None:
                p.color = color
            if pin is not None:
                p.pin_hash = hash_secret(pin, PIN_SCRYPT_N) if pin else None
            await s.flush()
            return p.to_dict()

    async def delete(self, profile_id: str) -> bool:
        async with database.session() as s:
            p = await s.get(Profile, profile_id)
            if not p:
                return False
            await s.delete(p)
            return True

    async def check_pin(self, profile_id: str, pin: Optional[str]) -> bool:
        async with database.session() as s:
            p = await s.get(Profile, profile_id)
            if not p:
                return False
            return p.pin_hash is None or verify_secret(pin or "", p.pin_hash)

    async def touch(self, profile_id: str) -> None:
        async with database.session() as s:
            p = await s.get(Profile, profile_id)
            if p:
                p.last_used_at = utcnow()

    async def sole_unlocked_profile(self) -> Optional[str]:
        """The profile to auto-select after unlocking: only when exactly one exists and it has no PIN."""
        async with database.session() as s:
            rows = (await s.execute(select(Profile).limit(2))).scalars().all()
            if len(rows) == 1 and rows[0].pin_hash is None:
                return rows[0].id
            return None


profile_repository = ProfileRepository()
