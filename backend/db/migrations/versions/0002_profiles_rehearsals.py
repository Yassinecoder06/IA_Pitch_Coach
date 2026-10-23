"""Local profiles, install-wide AI settings, decks, rehearsals, goals and game runs.

Data written before profiles existed belonged to a single hardcoded user id;
that id becomes the default profile so nothing is lost.

Revision ID: 0002_profiles_rehearsals
Revises: 0001_legacy
"""

import json
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op

from backend.db.types import JSONType

revision = "0002_profiles_rehearsals"
down_revision = "0001_legacy"
branch_labels = None
depends_on = None

DEFAULT_PROFILE_ID = "00000000-0000-0000-0000-000000000001"


def _ts():
    # Matches backend.db.types.UTCDateTime: TIMESTAMPTZ on PostgreSQL, ISO text on SQLite.
    return sa.String(40).with_variant(sa.DateTime(timezone=True), "postgresql")


def _now(bind):
    now = datetime.now(timezone.utc)
    return now if bind.dialect.name == "postgresql" else now.isoformat()


def _load_json(value, default):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    # PostgreSQL legacy tables used TIMESTAMPTZ already; SQLite stored ISO strings. Both stay as-is.
    op.create_table(
        "profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("color", sa.String(16), nullable=False, server_default="#a855f7"),
        sa.Column("pin_hash", sa.Text),
        sa.Column("created_at", _ts()),
        sa.Column("last_used_at", _ts()),
    )
    profiles = sa.table(
        "profiles", sa.column("id"), sa.column("name"), sa.column("color"), sa.column("created_at", _ts())
    )
    op.bulk_insert(profiles, [{"id": DEFAULT_PROFILE_ID, "name": "Me", "color": "#a855f7", "created_at": _now(bind)}])

    # Coach sessions now belong to a profile.
    with op.batch_alter_table("sessions") as batch:
        batch.drop_index("idx_sessions_user_updated")
        batch.alter_column("user_id", new_column_name="profile_id", existing_type=sa.Text, existing_nullable=False)
    # Rows from other (never-used) user ids are reassigned to the default profile so the FK holds.
    op.execute(
        sa.text("UPDATE sessions SET profile_id = :pid WHERE profile_id NOT IN (SELECT id FROM profiles)")
        .bindparams(pid=DEFAULT_PROFILE_ID)
    )
    with op.batch_alter_table("sessions") as batch:
        batch.create_foreign_key("fk_sessions_profile", "profiles", ["profile_id"], ["id"], ondelete="CASCADE")
        batch.create_index("ix_sessions_profile_id", ["profile_id"])

    # The legacy CHECK constraint was unnamed; recreate it with a name so batch mode keeps it.
    with op.batch_alter_table(
        "messages",
        table_args=(sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),),
    ) as batch:
        batch.drop_column("user_id")

    # Duplicated messages.metrics_json; never read.
    op.drop_table("speech_metrics")

    # Providers and model routing are install-wide (they hold the owner's API keys).
    with op.batch_alter_table("ai_providers") as batch:
        batch.drop_column("user_id")

    op.create_table(
        "ai_settings",
        sa.Column("id", sa.SmallInteger, primary_key=True),
        sa.Column("chat_models_json", JSONType, nullable=False),
        sa.Column("fallback_models_json", JSONType, nullable=False),
        sa.Column("task_models_json", JSONType, nullable=False),
        sa.Column("tts_voice", sa.String(80)),
        sa.Column("updated_at", _ts()),
    )
    legacy = bind.execute(
        sa.text("SELECT chat_models_json, fallback_models_json, tts_voice FROM ai_preferences ORDER BY updated_at DESC")
    ).first()
    settings_table = sa.table(
        "ai_settings",
        sa.column("id"),
        sa.column("chat_models_json", JSONType),
        sa.column("fallback_models_json", JSONType),
        sa.column("task_models_json", JSONType),
        sa.column("tts_voice"),
        sa.column("updated_at", _ts()),
    )
    op.bulk_insert(settings_table, [{
        "id": 1,
        "chat_models_json": _load_json(legacy[0], []) if legacy else [],
        "fallback_models_json": _load_json(legacy[1], []) if legacy else [],
        "task_models_json": {},
        "tts_voice": legacy[2] if legacy else None,
        "updated_at": _now(bind),
    }])
    op.drop_table("ai_preferences")

    op.create_table(
        "decks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("profile_id", sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("brief_json", JSONType),
        sa.Column("critique_json", JSONType),
        sa.Column("file_path", sa.Text),
        sa.Column("created_at", _ts()),
        sa.Column("updated_at", _ts()),
    )
    op.create_table(
        "slides",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("deck_id", sa.String(36), sa.ForeignKey("decks.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("idx", sa.Integer, nullable=False),
        sa.Column("kind", sa.String(24)),
        sa.Column("title", sa.Text, nullable=False, server_default=""),
        sa.Column("bullets_json", JSONType, nullable=False),
        sa.Column("notes", sa.Text, nullable=False, server_default=""),
        sa.Column("text", sa.Text, nullable=False, server_default=""),
        sa.Column("image_path", sa.Text),
        sa.Column("time_budget_sec", sa.Integer),
    )
    op.create_table(
        "rehearsals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("profile_id", sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("deck_id", sa.String(36), sa.ForeignKey("decks.id", ondelete="SET NULL"), index=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("progress", sa.String(80)),
        sa.Column("target_seconds", sa.Integer),
        sa.Column("duration_sec", sa.Float),
        sa.Column("transcript", sa.Text),
        sa.Column("words_json", JSONType),
        sa.Column("metrics_json", JSONType),
        sa.Column("sensors_json", JSONType),
        sa.Column("cues_json", JSONType),
        sa.Column("judge_json", JSONType),
        sa.Column("scores_json", JSONType),
        sa.Column("overall_score", sa.Float),
        sa.Column("audio_path", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("created_at", _ts()),
        sa.Column("completed_at", _ts()),
    )
    op.create_table(
        "rehearsal_slide_segments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("rehearsal_id", sa.String(36), sa.ForeignKey("rehearsals.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("slide_idx", sa.Integer, nullable=False),
        sa.Column("start_sec", sa.Float, nullable=False),
        sa.Column("end_sec", sa.Float, nullable=False),
        sa.Column("metrics_json", JSONType),
    )
    op.create_table(
        "goals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("profile_id", sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("metric", sa.String(40), nullable=False),
        sa.Column("op", sa.String(8), nullable=False),
        sa.Column("target", sa.Float, nullable=False),
        sa.Column("target_max", sa.Float),
        sa.Column("created_at", _ts()),
    )
    op.create_table(
        "game_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("profile_id", sa.String(36), sa.ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("game", sa.String(32), nullable=False),
        sa.Column("difficulty", sa.String(16), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("stats_json", JSONType),
        sa.Column("created_at", _ts()),
    )
    if is_pg:
        op.execute("ALTER TABLE app_owner ALTER COLUMN id SET DEFAULT 1")


def downgrade() -> None:
    raise NotImplementedError("Downgrading below 0002 is not supported; restore a backup instead.")
