"""
Repository for sessions, messages, and speech metrics.
Stores full turn history, rubric scores, and speech analysis data.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from backend.storage.database import db, DEFAULT_LOCAL_USER_ID


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionRepository:
    """Manages pitch coaching sessions and historical messages."""

    def create_session(
        self,
        title: str = "Startup Pitch Practice",
        mode: str = "pitch_analysis",
        user_id: str = DEFAULT_LOCAL_USER_ID
    ) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        created_at = now_iso()
        db.execute(
            """INSERT INTO sessions (id, user_id, title, current_mode, context_md, created_at, updated_at)
               VALUES (?, ?, ?, ?, '', ?, ?)""",
            (session_id, user_id, title, mode, created_at, created_at)
        )
        return self.get_session(session_id, user_id) or {}

    def list_sessions(self, user_id: str = DEFAULT_LOCAL_USER_ID, limit: int = 50) -> List[Dict[str, Any]]:
        rows = db.fetch_all(
            """SELECT id, user_id, title, current_mode, context_md, created_at, updated_at
               FROM sessions WHERE user_id = ?
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, limit)
        )
        return rows

    def get_session(self, session_id: str, user_id: str = DEFAULT_LOCAL_USER_ID) -> Optional[Dict[str, Any]]:
        return db.fetch_one(
            """SELECT id, user_id, title, current_mode, context_md, created_at, updated_at
               FROM sessions WHERE id = ? AND user_id = ?""",
            (session_id, user_id)
        )

    def update_mode(self, session_id: str, user_id: str, mode: str) -> None:
        db.execute(
            """UPDATE sessions SET current_mode = ?, updated_at = ?
               WHERE id = ? AND user_id = ?""",
            (mode, now_iso(), session_id, user_id)
        )

    def rename_session(self, session_id: str, user_id: str, title: str) -> None:
        db.execute(
            """UPDATE sessions SET title = ?, updated_at = ?
               WHERE id = ? AND user_id = ?""",
            (title, now_iso(), session_id, user_id)
        )

    def delete_session(self, session_id: str, user_id: str = DEFAULT_LOCAL_USER_ID) -> bool:
        """Deletes session and cascades to messages and metrics."""
        session = self.get_session(session_id, user_id)
        if not session:
            return False
        db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM speech_metrics WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
        return True

    def append_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        transcript: Optional[str] = None,
        scores: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        audio_url: Optional[str] = None
    ) -> Dict[str, Any]:
        msg_id = str(uuid.uuid4())
        created_at = now_iso()
        scores_str = json.dumps(scores) if scores else None
        metrics_str = json.dumps(metrics) if metrics else None

        db.execute(
            """INSERT INTO messages (id, session_id, user_id, role, content, transcript, scores_json, metrics_json, audio_url, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, session_id, user_id, role, content, transcript, scores_str, metrics_str, audio_url, created_at)
        )
        db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (created_at, session_id))

        return {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "transcript": transcript,
            "scores": scores,
            "metrics": metrics,
            "created_at": created_at
        }

    def append_speech_metrics(
        self,
        session_id: str,
        user_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        metric_id = str(uuid.uuid4())
        created_at = now_iso()
        db.execute(
            """INSERT INTO speech_metrics (id, session_id, user_id, words_per_minute, pause_frequency, pause_duration, energy_variation, rhythm_score, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                metric_id,
                session_id,
                user_id,
                float(metrics.get("words_per_minute", 0.0)),
                float(metrics.get("pause_frequency", 0.0)),
                float(metrics.get("pause_duration", 0.0)),
                float(metrics.get("energy_variation", 0.0)),
                float(metrics.get("rhythm_score", 0.0)),
                created_at
            )
        )

    def get_recent_messages(
        self,
        session_id: str,
        user_id: str = DEFAULT_LOCAL_USER_ID,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        rows = db.fetch_all(
            """SELECT id, session_id, role, content, transcript, scores_json, metrics_json, audio_url, created_at
               FROM messages WHERE session_id = ? AND user_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (session_id, user_id, limit)
        )
        result = []
        for r in rows:
            item = dict(r)
            if item.get("scores_json"):
                try:
                    item["scores"] = json.loads(item["scores_json"])
                except Exception:
                    item["scores"] = None
            else:
                item["scores"] = None

            if item.get("metrics_json"):
                try:
                    item["metrics"] = json.loads(item["metrics_json"])
                except Exception:
                    item["metrics"] = None
            else:
                item["metrics"] = None
            result.append(item)
        return result

    def delete_last_turn(self, session_id: str, user_id: str = DEFAULT_LOCAL_USER_ID) -> int:
        """Deletes the latest assistant message and preceding user message."""
        rows = db.fetch_all(
            """SELECT id, role FROM messages WHERE session_id = ? AND user_id = ?
               ORDER BY created_at DESC LIMIT 2""",
            (session_id, user_id)
        )
        if not rows:
            return 0

        deleted = 0
        for r in rows:
            db.execute("DELETE FROM messages WHERE id = ?", (r["id"],))
            deleted += 1
        return deleted

    def get_progress(self, user_id: str = DEFAULT_LOCAL_USER_ID) -> List[Dict[str, Any]]:
        """Returns chronological scores history for progress charting."""
        rows = db.fetch_all(
            """SELECT m.session_id, s.title, m.created_at, m.scores_json, m.metrics_json
               FROM messages m
               JOIN sessions s ON m.session_id = s.id
               WHERE m.user_id = ? AND m.role = 'assistant' AND m.scores_json IS NOT NULL
               ORDER BY m.created_at ASC LIMIT 100""",
            (user_id,)
        )
        progress = []
        for r in rows:
            try:
                scores = json.loads(r["scores_json"]) if r["scores_json"] else {}
                metrics = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
                progress.append({
                    "session_id": r["session_id"],
                    "session_title": r["title"],
                    "date": r["created_at"],
                    "overall": scores.get("overall", 0),
                    "clarity": scores.get("clarity", {}).get("score", 0) if isinstance(scores.get("clarity"), dict) else scores.get("clarity", 0),
                    "structure": scores.get("structure", {}).get("score", 0) if isinstance(scores.get("structure"), dict) else 0,
                    "persuasion": scores.get("persuasion", {}).get("score", 0) if isinstance(scores.get("persuasion"), dict) else 0,
                    "delivery": scores.get("delivery", {}).get("score", 0) if isinstance(scores.get("delivery"), dict) else 0,
                    "confidence": scores.get("confidence", {}).get("score", 0) if isinstance(scores.get("confidence"), dict) else scores.get("confidence", 0),
                    "relevance": scores.get("relevance", {}).get("score", 0) if isinstance(scores.get("relevance"), dict) else scores.get("topic_relevance", 0),
                    "wpm": metrics.get("words_per_minute", 0),
                    "fillers": metrics.get("filler_count", 0),
                })
            except Exception:
                continue
        return progress


session_repository = SessionRepository()
