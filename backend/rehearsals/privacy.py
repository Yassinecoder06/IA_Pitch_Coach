"""Recording retention: keep recordings or not, and delete them after N days."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, Optional

from sqlalchemy import select

from backend.db import database
from backend.db.models import AppSettings, Rehearsal
from backend.db.types import utcnow

logger = logging.getLogger(__name__)


async def get_settings() -> Dict[str, Any]:
    async with database.session() as s:
        row = await s.get(AppSettings, 1)
        return {
            "keep_recordings": True if row is None else bool(row.keep_recordings),
            "recording_retention_days": None if row is None else row.recording_retention_days,
        }


async def save_settings(keep_recordings: bool, retention_days: Optional[int]) -> Dict[str, Any]:
    async with database.session() as s:
        row = await s.get(AppSettings, 1)
        if not row:
            row = AppSettings(id=1)
            s.add(row)
        row.keep_recordings = keep_recordings
        row.recording_retention_days = retention_days
    await sweep_recordings()
    return await get_settings()


async def drop_recording(rehearsal_id: str, profile_id: Optional[str] = None) -> bool:
    """Deletes a rehearsal's audio file; the report and transcript stay."""
    from backend.rehearsals.service import recordings_dir

    async with database.session() as s:
        r = await s.get(Rehearsal, rehearsal_id)
        if not r or (profile_id and r.profile_id != profile_id) or not r.audio_path:
            return False
        (recordings_dir() / r.audio_path).unlink(missing_ok=True)
        r.audio_path = None
        return True


async def sweep_recordings() -> int:
    """Applies the retention setting to every stored recording. Returns how many were deleted."""
    settings = await get_settings()
    days = settings["recording_retention_days"]
    if settings["keep_recordings"] and not days:
        return 0
    async with database.session() as s:
        q = select(Rehearsal.id).where(Rehearsal.audio_path.is_not(None), Rehearsal.status.in_(("done", "failed")))
        if settings["keep_recordings"]:
            q = q.where(Rehearsal.created_at < utcnow() - timedelta(days=days))
        ids = (await s.execute(q)).scalars().all()
    for rid in ids:
        await drop_recording(rid)
    if ids:
        logger.info(f"[Privacy] Deleted {len(ids)} recording(s) per retention settings.")
    return len(ids)
