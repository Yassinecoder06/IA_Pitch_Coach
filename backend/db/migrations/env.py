"""Alembic environment. Runs migrations on a synchronous connection."""

import sqlalchemy as sa
from alembic import context

from backend.db.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection")
    if connectable is not None:
        _run(connectable)
        return
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"), poolclass=sa.pool.NullPool)
    with engine.connect() as connection:
        _run(connection)
    engine.dispose()


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
