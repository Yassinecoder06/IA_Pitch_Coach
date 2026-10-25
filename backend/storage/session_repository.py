"""
Repository for coaching chat sessions and their messages.
Every query is scoped to a profile; messages are reached through their session.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select

from backend.db import database
from backend.db.models import CoachSession, Message
from backend.db.types import utcnow

RUBRIC_DIMENSIONS = ("clarity", "structure", "persuasion", "delivery", "confidence", "relevance")


def _dimension_score(scores: Dict[str, Any], key: str) -> float:
    value = scores.get(key)
    if isinstance(value, dict):
        return value.get("score", 0) or 0
    return value or 0


class SessionRepository:
    """Manages pitch coaching sessions and historical messages."""

    async def create_session(
        self, profile_id: str, title: str = "Startup Pitch Practice", mode: str = "pitch_analysis", context_md: str = ""
    ) -> Dict[str, Any]:
        async with database.session() as s:
            row = CoachSession(profile_id=profile_id, title=title, current_mode=mode, context_md=context_md)
            s.add(row)
            await s.flush()
            return row.to_dict()

    async def list_sessions(self, profile_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        async with database.session() as s:
            rows = (await s.execute(
                select(CoachSession)
                .where(CoachSession.profile_id == profile_id)
                .order_by(CoachSession.updated_at.desc())
                .limit(limit)
            )).scalars()
            return [r.to_dict() for r in rows]

    async def _owned(self, s, session_id: str, profile_id: str) -> Optional[CoachSession]:
        row = await s.get(CoachSession, session_id)
        return row if row and row.profile_id == profile_id else None

    async def get_session(self, session_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
        async with database.session() as s:
            row = await self._owned(s, session_id, profile_id)
            return row.to_dict() if row else None

    async def update_mode(self, session_id: str, profile_id: str, mode: str) -> bool:
        async with database.session() as s:
            row = await self._owned(s, session_id, profile_id)
            if not row:
                return False
            row.current_mode = mode
            row.updated_at = utcnow()
            return True

    async def rename_session(self, session_id: str, profile_id: str, title: str) -> bool:
        async with database.session() as s:
            row = await self._owned(s, session_id, profile_id)
            if not row:
                return False
            row.title = title
            row.updated_at = utcnow()
            return True

    async def delete_session(self, session_id: str, profile_id: str) -> bool:
        async with database.session() as s:
            row = await self._owned(s, session_id, profile_id)
            if not row:
                return False
            await s.delete(row)
            return True

    async def append_message(
        self,
        session_id: str,
        profile_id: str,
        role: str,
        content: str,
        transcript: Optional[str] = None,
        scores: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        audio_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        async with database.session() as s:
            session = await self._owned(s, session_id, profile_id)
            if not session:
                return None
            msg = Message(
                session_id=session_id,
                role=role,
                content=content,
                transcript=transcript,
                scores_json=scores or None,
                metrics_json=metrics or None,
                audio_url=audio_url,
            )
            s.add(msg)
            session.updated_at = msg.created_at = utcnow()
            await s.flush()
            return msg.to_dict()

    async def get_recent_messages(self, session_id: str, profile_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """The latest `limit` messages of a session, oldest first."""
        async with database.session() as s:
            if not await self._owned(s, session_id, profile_id):
                return []
            rows = (await s.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )).scalars().all()
            return [m.to_dict() for m in reversed(rows)]

    async def delete_last_turn(self, session_id: str, profile_id: str) -> int:
        """Deletes the latest assistant message and the user message before it."""
        async with database.session() as s:
            if not await self._owned(s, session_id, profile_id):
                return 0
            ids = (await s.execute(
                select(Message.id)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
                .limit(2)
            )).scalars().all()
            if ids:
                await s.execute(delete(Message).where(Message.id.in_(ids)))
            return len(ids)

    async def get_progress(self, profile_id: str) -> List[Dict[str, Any]]:
        """Chronological scorecard history for progress charting."""
        async with database.session() as s:
            rows = (await s.execute(
                select(Message, CoachSession.title)
                .join(CoachSession, Message.session_id == CoachSession.id)
                .where(
                    CoachSession.profile_id == profile_id,
                    Message.role == "assistant",
                    Message.scores_json.is_not(None),
                )
                .order_by(Message.created_at)
                .limit(100)
            )).all()
        progress = []
        for msg, title in rows:
            scores = msg.scores_json or {}
            metrics = msg.metrics_json or {}
            item = {
                "session_id": msg.session_id,
                "session_title": title,
                "date": msg.to_dict()["created_at"],
                "overall": scores.get("overall", 0),
                "wpm": metrics.get("words_per_minute", 0),
                "fillers": metrics.get("filler_count", 0),
            }
            item.update({k: _dimension_score(scores, k) for k in RUBRIC_DIMENSIONS})
            progress.append(item)
        return progress


session_repository = SessionRepository()
