"""Portable column types shared by the ORM models and migrations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

# JSON stored as TEXT on SQLite and JSONB on PostgreSQL.
JSONType = sa.JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC timestamps.

    SQLite keeps ISO-8601 strings (the format earlier versions of the app wrote),
    PostgreSQL uses TIMESTAMPTZ. Values always come back as aware datetimes.
    """

    impl = sa.String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(sa.DateTime(timezone=True))
        return dialect.type_descriptor(sa.String(40))

    def process_bind_param(self, value: Optional[datetime], dialect) -> Any:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
        return value if dialect.name == "postgresql" else value.isoformat()

    def process_result_value(self, value: Any, dialect) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace(" ", "T", 1))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value
