"""Session persistence layer backed by Supabase."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from urllib import request, error
from typing import Any, Dict, List, Optional

from backend.storage.supabase_client import get_supabase_client


@dataclass
class SessionContextWindow:
    session_id: str
    mode: str
    context_md: str
    recent_messages: List[Dict[str, str]]


class SessionManager:
    """CRUD and context-window operations for coaching sessions."""

    def __init__(self):
        self.client = get_supabase_client()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def create_session(self, title: str, mode: str, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        payload = {
            "title": title or "Untitled Session",
            "current_mode": mode,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "context_md": "",
        }
        result = self.client.table("sessions").insert(payload).execute()
        rows = result.data or []
        return rows[0] if rows else None

    def list_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        result = (
            self.client
            .table("sessions")
            .select("id,title,created_at,updated_at,current_mode")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []

    def get_session(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        result = (
            self.client
            .table("sessions")
            .select("id,title,created_at,updated_at,current_mode,context_md")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None

    def get_recent_messages(self, session_id: str, user_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        result = (
            self.client
            .table("messages")
            .select("id,session_id,role,content,transcript,created_at,audio_url")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = result.data or []
        return list(reversed(rows))

    def append_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        transcript: Optional[str] = None,
        audio_url: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "transcript": transcript,
            "audio_url": audio_url,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.client.table("messages").insert(payload).execute()
        self._touch_session(session_id, user_id)

    def append_speech_metrics(self, session_id: str, user_id: str, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "words_per_minute": metrics.get("words_per_minute", 0.0),
            "pause_frequency": metrics.get("pause_frequency", 0.0),
            "pause_duration": metrics.get("pause_duration", 0.0),
            "energy_variation": metrics.get("energy_variation", 0.0),
            "rhythm_score": metrics.get("rhythm_score", 0.0),
            "created_at": datetime.utcnow().isoformat(),
        }
        self.client.table("speech_metrics").insert(payload).execute()
        self._touch_session(session_id, user_id)

    def update_mode(self, session_id: str, user_id: str, mode: str) -> None:
        if not self.enabled:
            return
        self.client.table("sessions").update({"current_mode": mode, "updated_at": datetime.utcnow().isoformat()}).eq("id", session_id).eq("user_id", user_id).execute()

    def update_context_markdown(self, session_id: str, user_id: str, context_md: str) -> None:
        if not self.enabled:
            return

        self.client.table("sessions").update({"context_md": context_md, "updated_at": datetime.utcnow().isoformat()}).eq("id", session_id).eq("user_id", user_id).execute()

    def delete_session(self, session_id: str, user_id: str, auth_token: Optional[str] = None) -> bool:
        ok, _ = self.delete_session_with_error(session_id, user_id, auth_token)
        return ok

    def delete_session_with_error(self, session_id: str, user_id: str, auth_token: Optional[str] = None) -> tuple[bool, str]:
        if not self.enabled:
            return False, "Supabase is not configured"

        direct_deleted, direct_error = self._delete_session_direct(session_id, user_id)
        if direct_deleted:
            return True, ""

        rpc_result = self._call_rpc_via_rest("delete_session_owned", {"p_session_id": session_id}, auth_token)
        if isinstance(rpc_result, dict) and rpc_result.get("__error__"):
            return False, direct_error or str(rpc_result.get("__error__"))
        if rpc_result is None:
            return False, direct_error or "RPC unavailable or unauthorized"
        return True, ""

    def delete_last_turn(self, session_id: str, user_id: str, auth_token: Optional[str] = None) -> int:
        if not self.enabled:
            return 0

        rpc_result = self._call_rpc_via_rest("delete_last_turn_owned", {"p_session_id": session_id}, auth_token)
        if isinstance(rpc_result, int):
            return rpc_result
        if isinstance(rpc_result, bool) and rpc_result:
            return 1
        return 0

    def build_context_window(self, session_id: str, user_id: str, last_n: int = 8) -> Optional[SessionContextWindow]:
        session = self.get_session(session_id, user_id)
        if not session:
            return None

        recent = self.get_recent_messages(session_id, user_id, limit=max(1, last_n))
        llm_messages = [{"role": m["role"], "content": m.get("content", "")} for m in recent if m.get("content")]

        return SessionContextWindow(
            session_id=session_id,
            mode=session.get("current_mode", "pitch_analysis"),
            context_md=session.get("context_md", "") or "",
            recent_messages=llm_messages,
        )

    def generate_session_summary_markdown(self, session_id: str, user_id: str) -> str:
        """Generate a compact markdown memory from recent conversation and latest metrics."""
        session = self.get_session(session_id, user_id)
        if not session:
            return ""

        messages = self.get_recent_messages(session_id, user_id, limit=20)
        metrics_result = (
            self.client
            .table("speech_metrics")
            .select("words_per_minute,pause_frequency,pause_duration,energy_variation,rhythm_score,created_at")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ) if self.enabled else None

        latest_metrics = (metrics_result.data or [None])[0] if metrics_result else None

        user_topics: List[str] = []
        highlights: List[str] = []
        for msg in messages[-10:]:
            if msg.get("role") == "user":
                text = (msg.get("content") or "").strip()
                if text:
                    user_topics.append(text[:120])
            elif msg.get("role") == "assistant":
                text = (msg.get("content") or "").strip()
                if text:
                    highlights.append(text[:160])

        key_topics = user_topics[-3:]
        coaching_points = highlights[-3:]

        md = [
            f"# Session: {session.get('title', 'Pitch Practice')}",
            "",
            "## Summary",
            f"Mode: {session.get('current_mode', 'pitch_analysis')}.",
            "Session memory compressed from recent turns and speech metrics.",
            "",
            "## Key Topics",
        ]

        if key_topics:
            md.extend([f"- {t}" for t in key_topics])
        else:
            md.append("- No key topics captured yet")

        md.extend(["", "## Coaching Feedback"])
        if coaching_points:
            md.extend([f"- {c}" for c in coaching_points])
        else:
            md.append("- No coaching feedback captured yet")

        md.extend(["", "## Speech Metrics"])
        if latest_metrics:
            md.extend([
                f"- Words per minute: {latest_metrics.get('words_per_minute', 0):.2f}",
                f"- Pause frequency: {latest_metrics.get('pause_frequency', 0):.2f}",
                f"- Pause duration: {latest_metrics.get('pause_duration', 0):.2f}",
                f"- Energy variation: {latest_metrics.get('energy_variation', 0):.4f}",
                f"- Rhythm score: {latest_metrics.get('rhythm_score', 0):.2f}",
            ])
        else:
            md.append("- No metrics captured yet")

        md.extend(["", "## Conversation Highlights"])
        if highlights:
            md.extend([f"- {h}" for h in highlights[-5:]])
        else:
            md.append("- No highlights captured yet")

        return "\n".join(md)

    def _touch_session(self, session_id: str, user_id: str) -> None:
        if not self.enabled:
            return
        self.client.table("sessions").update({"updated_at": datetime.utcnow().isoformat()}).eq("id", session_id).eq("user_id", user_id).execute()

    def _delete_session_direct(self, session_id: str, user_id: str) -> tuple[bool, str]:
        """Delete after backend auth has already verified the owner token."""
        session = self.get_session(session_id, user_id)
        if not session:
            return False, "Session not found for this user"

        try:
            self.client.table("speech_metrics").delete().eq("session_id", session_id).eq("user_id", user_id).execute()
            self.client.table("messages").delete().eq("session_id", session_id).eq("user_id", user_id).execute()
            self.client.table("sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
        except Exception as exc:
            return False, str(exc)

        if self.get_session(session_id, user_id):
            return False, "Session delete did not remove the row"
        return True, ""

    def _call_rpc_via_rest(self, procedure: str, payload: Dict[str, Any], auth_token: Optional[str]) -> Optional[Any]:
        env = os.getenv("SUPABASE_ENV", "cloud").lower()
        if env == "local":
            supabase_url = os.getenv("SUPABASE_LOCAL_URL", "").strip().rstrip("/")
            supabase_key = os.getenv("SUPABASE_LOCAL_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_LOCAL_ANON_KEY", "").strip()
        else:
            supabase_url = (os.getenv("SUPABASE_CLOUD_URL", "").strip() or os.getenv("SUPABASE_URL", "").strip()).rstrip("/")
            supabase_key = (
                os.getenv("SUPABASE_CLOUD_SERVICE_ROLE_KEY", "").strip() or
                os.getenv("SUPABASE_CLOUD_ANON_KEY", "").strip() or
                os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or
                os.getenv("SUPABASE_ANON_KEY", "").strip()
            )
        if not supabase_url or not supabase_key or not auth_token:
            return None

        url = f"{supabase_url}/rest/v1/rpc/{procedure}"
        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
        req.add_header("apikey", supabase_key)
        req.add_header("Authorization", f"Bearer {auth_token}")
        req.add_header("Content-Type", "application/json")

        try:
            with request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8").strip()
                if not body:
                    return True
                return json.loads(body)
        except error.HTTPError as http_error:
            try:
                body = http_error.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                body = ""
            if http_error.code == 404:
                return None
            detail = body or http_error.reason or "RPC error"
            return {"__error__": f"{http_error.code}: {detail}"}
        except Exception as exc:
            return {"__error__": str(exc)}
