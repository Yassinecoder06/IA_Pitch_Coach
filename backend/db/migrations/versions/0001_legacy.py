"""Schema used before Alembic was introduced (v3.0).

Existing databases that already have these tables are stamped at this revision
instead of running it.

Revision ID: 0001_legacy
Revises:
"""

import sqlalchemy as sa
from alembic import op

from backend.db.types import JSONType

revision = "0001_legacy"
down_revision = None
branch_labels = None
depends_on = None

TS = sa.String(40)  # the legacy code wrote ISO-8601 strings


def upgrade() -> None:
    op.create_table(
        "app_owner",
        sa.Column("id", sa.SmallInteger, primary_key=True),
        sa.Column("passphrase_hash", sa.Text, nullable=False),
        sa.Column("session_salt", sa.Text, nullable=False),
        sa.Column("created_at", TS),
        sa.Column("updated_at", TS),
    )
    op.create_table(
        "ai_providers",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("base_url", sa.Text),
        sa.Column("api_key_encrypted", sa.Text),
        sa.Column("key_hint", sa.Text),
        sa.Column("models_json", JSONType, nullable=False, server_default="[]"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("last_fetched_at", TS),
        sa.Column("last_error", sa.Text),
        sa.Column("created_at", TS),
        sa.Column("updated_at", TS),
    )
    op.create_table(
        "ai_preferences",
        sa.Column("user_id", sa.Text, primary_key=True),
        sa.Column("chat_models_json", JSONType, nullable=False, server_default="[]"),
        sa.Column("fallback_models_json", JSONType, nullable=False, server_default="[]"),
        sa.Column("tts_ref", sa.Text),
        sa.Column("tts_voice", sa.Text),
        sa.Column("stt_ref", sa.Text),
        sa.Column("updated_at", TS),
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False, server_default="Untitled Session"),
        sa.Column("current_mode", sa.Text, nullable=False, server_default="pitch_analysis"),
        sa.Column("context_md", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", TS),
        sa.Column("updated_at", TS),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("session_id", sa.Text, sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("transcript", sa.Text),
        sa.Column("scores_json", JSONType),
        sa.Column("metrics_json", JSONType),
        sa.Column("audio_url", sa.Text),
        sa.Column("created_at", TS),
        sa.CheckConstraint("role IN ('user', 'assistant')"),
    )
    op.create_table(
        "speech_metrics",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("session_id", sa.Text, sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("words_per_minute", sa.Float, nullable=False, server_default="0"),
        sa.Column("pause_frequency", sa.Float, nullable=False, server_default="0"),
        sa.Column("pause_duration", sa.Float, nullable=False, server_default="0"),
        sa.Column("energy_variation", sa.Float, nullable=False, server_default="0"),
        sa.Column("rhythm_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("created_at", TS),
    )
    op.create_index("idx_messages_session", "messages", ["session_id", "created_at"])
    op.create_index("idx_sessions_user_updated", "sessions", ["user_id", "updated_at"])


def downgrade() -> None:
    for table in ("speech_metrics", "messages", "sessions", "ai_preferences", "ai_providers", "app_owner"):
        op.drop_table(table)
