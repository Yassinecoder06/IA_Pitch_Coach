"""
Database connection and schema management for AI Pitch Coach.
Supports SQLite (zero-config local) and PostgreSQL (Docker/Production).
"""

from __future__ import annotations

import os
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from backend.config.settings import settings, ROOT_DIR

DEFAULT_LOCAL_USER_ID = "00000000-0000-0000-0000-000000000001"


class Database:
    """Unified database manager supporting SQLite and PostgreSQL."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.database_url
        self.is_postgres = self.db_url.startswith("postgres://") or self.db_url.startswith("postgresql://")

        if not self.is_postgres:
            # Normalize SQLite path
            sqlite_path = self.db_url.replace("sqlite:///", "")
            self.sqlite_file = Path(sqlite_path)
            self.sqlite_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.sqlite_file = None

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.sqlite_file))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        if not self.is_postgres:
            with self._get_sqlite_connection() as conn:
                conn.executescript("""
                CREATE TABLE IF NOT EXISTS app_owner (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    passphrase_hash TEXT NOT NULL,
                    session_salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ai_providers (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    base_url TEXT,
                    api_key_encrypted TEXT,
                    key_hint TEXT,
                    models_json TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_fetched_at TIMESTAMP,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS ai_preferences (
                    user_id TEXT PRIMARY KEY,
                    chat_models_json TEXT NOT NULL DEFAULT '[]',
                    fallback_models_json TEXT NOT NULL DEFAULT '[]',
                    tts_ref TEXT,
                    tts_voice TEXT,
                    stt_ref TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT 'Untitled Session',
                    current_mode TEXT NOT NULL DEFAULT 'pitch_analysis',
                    context_md TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    transcript TEXT,
                    scores_json TEXT,
                    metrics_json TEXT,
                    audio_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS speech_metrics (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    words_per_minute REAL NOT NULL DEFAULT 0.0,
                    pause_frequency REAL NOT NULL DEFAULT 0.0,
                    pause_duration REAL NOT NULL DEFAULT 0.0,
                    energy_variation REAL NOT NULL DEFAULT 0.0,
                    rhythm_score REAL NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);
                """)
        else:
            import psycopg
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS app_owner (
                        id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                        passphrase_hash TEXT NOT NULL,
                        session_salt TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS ai_providers (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        name TEXT NOT NULL,
                        base_url TEXT,
                        api_key_encrypted TEXT,
                        key_hint TEXT,
                        models_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        last_fetched_at TIMESTAMPTZ,
                        last_error TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS ai_preferences (
                        user_id TEXT PRIMARY KEY,
                        chat_models_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        fallback_models_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        tts_ref TEXT,
                        tts_voice TEXT,
                        stt_ref TEXT,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT 'Untitled Session',
                        current_mode TEXT NOT NULL DEFAULT 'pitch_analysis',
                        context_md TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        transcript TEXT,
                        scores_json JSONB,
                        metrics_json JSONB,
                        audio_url TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS speech_metrics (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        user_id TEXT NOT NULL,
                        words_per_minute DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        pause_frequency DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        pause_duration DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        energy_variation DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        rhythm_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);
                    """)
                conn.commit()

    def execute(self, query: str, params: Tuple = ()) -> None:
        """Execute insert/update/delete statement."""
        if not self.is_postgres:
            with self._get_sqlite_connection() as conn:
                conn.execute(query, params)
                conn.commit()
        else:
            import psycopg
            # Replace ? with %s for Postgres
            pg_query = query.replace("?", "%s")
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(pg_query, params)
                conn.commit()

    def fetch_one(self, query: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch single row as dictionary."""
        if not self.is_postgres:
            with self._get_sqlite_connection() as conn:
                cur = conn.execute(query, params)
                row = cur.fetchone()
                return dict(row) if row else None
        else:
            import psycopg
            from psycopg.rows import dict_row
            pg_query = query.replace("?", "%s")
            with psycopg.connect(self.db_url, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(pg_query, params)
                    return cur.fetchone()

    def fetch_all(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all rows as dictionaries."""
        if not self.is_postgres:
            with self._get_sqlite_connection() as conn:
                cur = conn.execute(query, params)
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        else:
            import psycopg
            from psycopg.rows import dict_row
            pg_query = query.replace("?", "%s")
            with psycopg.connect(self.db_url, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(pg_query, params)
                    return cur.fetchall()


# Global database instance
db = Database()
