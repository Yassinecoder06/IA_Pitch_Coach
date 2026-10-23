"""Async database engine, sessions and schema migrations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.config.settings import ROOT_DIR, settings

# Profile created by the 0002 migration; it owns everything stored before profiles existed.
DEFAULT_PROFILE_ID = "00000000-0000-0000-0000-000000000001"

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
LEGACY_REVISION = "0001_legacy"


def _split_url(url: str) -> tuple[str, str]:
    """Returns (async_url, sync_url) for a sqlite:/// or postgresql:// URL."""
    if url.startswith("sqlite"):
        path = url.split(":///", 1)[1] if ":///" in url else ""
        if path and path != ":memory:":
            p = Path(path)
            if not p.is_absolute():
                p = ROOT_DIR / p
            p.parent.mkdir(parents=True, exist_ok=True)
            path = str(p)
        return f"sqlite+aiosqlite:///{path}", f"sqlite:///{path}"
    if url.startswith(("postgres://", "postgresql://")):
        rest = url.split("://", 1)[1]
        return f"postgresql+asyncpg://{rest}", f"postgresql+psycopg://{rest}"
    return url, url


class Database:
    def __init__(self) -> None:
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
        self.url = settings.database_url
        self.sync_url = ""

    def configure(self, url: Optional[str] = None, null_pool: bool = False) -> None:
        """(Re)points the app at a database URL.

        `null_pool` opens a fresh connection per session; tests use it because they drive
        the app from several event loops, and pooled asyncpg connections are loop-bound.
        """
        if self._engine is not None:
            # Disposal is async, but nothing is in flight when tests reconfigure.
            self._engine.sync_engine.dispose()
        self.url = url or settings.database_url
        async_url, self.sync_url = _split_url(self.url)
        if async_url.startswith("sqlite"):
            # SQLite connections are cheap; no pool means no connection is ever shared across event loops.
            self._engine = create_async_engine(async_url, poolclass=NullPool)
            event.listen(self._engine.sync_engine, "connect", _sqlite_pragmas)
        elif null_pool:
            self._engine = create_async_engine(async_url, poolclass=NullPool)
        else:
            self._engine = create_async_engine(async_url, pool_pre_ping=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self.configure()
        assert self._engine is not None
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A unit of work: commits on success, rolls back on error."""
        if self._sessionmaker is None:
            self.configure()
        assert self._sessionmaker is not None
        async with self._sessionmaker() as s:
            try:
                yield s
                await s.commit()
            except BaseException:
                await s.rollback()
                raise

    def upgrade_sync(self) -> None:
        """Applies Alembic migrations. Databases created by older versions are stamped first."""
        from alembic import command
        from alembic.config import Config

        if not self.sync_url:
            self.configure(self.url)

        cfg = Config()
        cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
        cfg.set_main_option("sqlalchemy.url", self.sync_url.replace("%", "%%"))

        sync_engine = sa.create_engine(self.sync_url)
        try:
            tables = set(sa.inspect(sync_engine).get_table_names())
        finally:
            sync_engine.dispose()

        if "alembic_version" not in tables and "app_owner" in tables:
            command.stamp(cfg, LEGACY_REVISION)
        command.upgrade(cfg, "head")

    async def upgrade(self) -> None:
        await asyncio.to_thread(self.upgrade_sync)


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA journal_mode = WAL")
    cur.close()


database = Database()
