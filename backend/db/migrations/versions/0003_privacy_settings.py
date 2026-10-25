"""Install-wide privacy settings: whether recordings are kept, and for how long.

Revision ID: 0003_privacy_settings
Revises: 0002_profiles_rehearsals
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_privacy_settings"
down_revision = "0002_profiles_rehearsals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("id", sa.SmallInteger, primary_key=True),
        sa.Column("keep_recordings", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("recording_retention_days", sa.Integer),
    )
    op.bulk_insert(
        sa.table("app_settings", sa.column("id"), sa.column("keep_recordings"), sa.column("recording_retention_days")),
        [{"id": 1, "keep_recordings": True, "recording_retention_days": None}],
    )


def downgrade() -> None:
    op.drop_table("app_settings")
