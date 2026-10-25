"""ORM models for AI Pitch Coach."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from backend.db.types import JSONType, UTCDateTime, utcnow


def new_id() -> str:
    return str(uuid.uuid4())


def iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


class Base(DeclarativeBase):
    pass


class Owner(Base):
    """The single owner who unlocks this install with a passphrase."""

    __tablename__ = "app_owner"

    id: Mapped[int] = mapped_column(sa.SmallInteger, primary_key=True, default=1)
    passphrase_hash: Mapped[str] = mapped_column(sa.Text)
    session_salt: Mapped[str] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class Profile(Base):
    """A local speaker profile. Each profile has its own sessions, decks and rehearsals."""

    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(sa.String(80))
    color: Mapped[str] = mapped_column(sa.String(16), default="#a855f7")
    pin_hash: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "has_pin": bool(self.pin_hash),
            "created_at": iso(self.created_at),
            "last_used_at": iso(self.last_used_at),
        }


class AiProvider(Base):
    """An AI provider configured for the whole install (keys are encrypted at rest)."""

    __tablename__ = "ai_providers"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(sa.String(40))
    name: Mapped[str] = mapped_column(sa.String(120))
    base_url: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    api_key_encrypted: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    key_hint: Mapped[Optional[str]] = mapped_column(sa.String(16), nullable=True)
    models_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSONType, default=list)
    enabled: Mapped[bool] = mapped_column(sa.Boolean, default=True)
    last_fetched_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class AiSettings(Base):
    """Install-wide model routing: preferred chat models, fallbacks and per-task overrides."""

    __tablename__ = "ai_settings"

    id: Mapped[int] = mapped_column(sa.SmallInteger, primary_key=True, default=1)
    chat_models_json: Mapped[List[str]] = mapped_column(JSONType, default=list)
    fallback_models_json: Mapped[List[str]] = mapped_column(JSONType, default=list)
    task_models_json: Mapped[Dict[str, str]] = mapped_column(JSONType, default=dict)
    tts_voice: Mapped[Optional[str]] = mapped_column(sa.String(80), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class AppSettings(Base):
    """Install-wide privacy settings."""

    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(sa.SmallInteger, primary_key=True, default=1)
    keep_recordings: Mapped[bool] = mapped_column(sa.Boolean, default=True)
    recording_retention_days: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)


class CoachSession(Base):
    """A conversational coaching session (the chat workspace)."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(sa.String(200), default="Untitled Session")
    current_mode: Mapped[str] = mapped_column(sa.String(40), default="pitch_analysis")
    context_md: Mapped[str] = mapped_column(sa.Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    messages: Mapped[List["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True, order_by="Message.created_at"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "title": self.title,
            "current_mode": self.current_mode,
            "context_md": self.context_md,
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
        }


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(sa.String(16))
    content: Mapped[str] = mapped_column(sa.Text)
    transcript: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    scores_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    metrics_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    audio_url: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    session: Mapped[CoachSession] = relationship(back_populates="messages")

    __table_args__ = (
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
        sa.Index("idx_messages_session", "session_id", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "transcript": self.transcript,
            "scores": self.scores_json,
            "metrics": self.metrics_json,
            "audio_url": self.audio_url,
            "created_at": iso(self.created_at),
        }


class Deck(Base):
    """A presentation: uploaded (PDF/PPTX), typed as an outline, or generated by AI."""

    __tablename__ = "decks"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(sa.String(200))
    source: Mapped[str] = mapped_column(sa.String(16))  # pdf | pptx | outline | ai
    brief_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    critique_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    slides: Mapped[List["Slide"]] = relationship(
        back_populates="deck", cascade="all, delete-orphan", passive_deletes=True, order_by="Slide.idx"
    )


class Slide(Base):
    __tablename__ = "slides"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    deck_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("decks.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(sa.Integer)
    kind: Mapped[Optional[str]] = mapped_column(sa.String(24), nullable=True)  # hook, problem, ... (AI decks)
    title: Mapped[str] = mapped_column(sa.Text, default="")
    bullets_json: Mapped[List[str]] = mapped_column(JSONType, default=list)
    notes: Mapped[str] = mapped_column(sa.Text, default="")
    text: Mapped[str] = mapped_column(sa.Text, default="")  # text extracted from an uploaded page
    image_path: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    time_budget_sec: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)

    deck: Mapped[Deck] = relationship(back_populates="slides")


class Rehearsal(Base):
    """One recorded attempt at a talk, with its analysis."""

    __tablename__ = "rehearsals"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    deck_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), sa.ForeignKey("decks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(sa.String(200), default="Rehearsal")
    status: Mapped[str] = mapped_column(sa.String(16), default="recording")  # recording|processing|done|failed
    progress: Mapped[Optional[str]] = mapped_column(sa.String(80), nullable=True)
    target_seconds: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    transcript: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    words_json: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONType, nullable=True)
    metrics_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    sensors_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    cues_json: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONType, nullable=True)
    judge_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    scores_json: Mapped[Optional[Dict[str, float]]] = mapped_column(JSONType, nullable=True)
    overall_score: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    audio_path: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)

    segments: Mapped[List["RehearsalSlideSegment"]] = relationship(
        back_populates="rehearsal", cascade="all, delete-orphan", passive_deletes=True,
        order_by="RehearsalSlideSegment.start_sec",
    )


class RehearsalSlideSegment(Base):
    """Time spent on one slide during a rehearsal, with per-slide metrics."""

    __tablename__ = "rehearsal_slide_segments"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    rehearsal_id: Mapped[str] = mapped_column(
        sa.String(36), sa.ForeignKey("rehearsals.id", ondelete="CASCADE"), index=True
    )
    slide_idx: Mapped[int] = mapped_column(sa.Integer)
    start_sec: Mapped[float] = mapped_column(sa.Float)
    end_sec: Mapped[float] = mapped_column(sa.Float)
    metrics_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)

    rehearsal: Mapped[Rehearsal] = relationship(back_populates="segments")


class Goal(Base):
    """A target for one metric, e.g. eye_contact >= 70."""

    __tablename__ = "goals"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    metric: Mapped[str] = mapped_column(sa.String(40))
    op: Mapped[str] = mapped_column(sa.String(8))  # gte | lte | between
    target: Mapped[float] = mapped_column(sa.Float)
    target_max: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metric": self.metric,
            "op": self.op,
            "target": self.target,
            "target_max": self.target_max,
            "created_at": iso(self.created_at),
        }


class GameRun(Base):
    """One play of a training-arcade game."""

    __tablename__ = "game_runs"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    game: Mapped[str] = mapped_column(sa.String(32))
    difficulty: Mapped[str] = mapped_column(sa.String(16), default="medium")
    score: Mapped[float] = mapped_column(sa.Float)
    stats_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "game": self.game,
            "difficulty": self.difficulty,
            "score": self.score,
            "stats": self.stats_json,
            "created_at": iso(self.created_at),
        }
