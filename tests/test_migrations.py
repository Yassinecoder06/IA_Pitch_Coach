"""Upgrading a database created by v3.0 (before Alembic) must keep all user data."""

import json
import sqlite3
from pathlib import Path

from sqlalchemy import select

from backend.db import DEFAULT_PROFILE_ID, database
from backend.db.models import AiProvider, AiSettings, CoachSession, Message, Owner, Profile
from tests.conftest import run

LEGACY_SCHEMA = (Path(__file__).parent / "fixtures" / "legacy_v3_schema.sql").read_text()


def _legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    uid = DEFAULT_PROFILE_ID
    conn.execute("INSERT INTO app_owner (id, passphrase_hash, session_salt) VALUES (1, 'scrypt$hash', 'salt')")
    conn.execute(
        "INSERT INTO sessions (id, user_id, title, current_mode, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        ("s1", uid, "Seed pitch", "investor_qa", "2026-06-01T10:00:00+00:00", "2026-06-01T10:05:00.123456+00:00"),
    )
    conn.execute(
        "INSERT INTO messages (id, session_id, user_id, role, content, scores_json, created_at) VALUES (?,?,?,?,?,?,?)",
        ("m1", "s1", uid, "assistant", "Nice hook.", json.dumps({"overall": 71}), "2026-06-01T10:05:00+00:00"),
    )
    conn.execute(
        "INSERT INTO speech_metrics (id, session_id, user_id, words_per_minute) VALUES ('x', 's1', ?, 150)", (uid,)
    )
    conn.execute(
        "INSERT INTO ai_providers (id, user_id, kind, name, api_key_encrypted, key_hint, models_json) "
        "VALUES ('p1', ?, 'gemini', 'Gemini', 'blob', '...1234', ?)",
        (uid, json.dumps([{"id": "gemini-2.5-flash", "label": "Flash"}])),
    )
    conn.execute(
        "INSERT INTO ai_preferences (user_id, chat_models_json, fallback_models_json, tts_voice) VALUES (?,?,?,?)",
        (uid, json.dumps(["p1:gemini-2.5-flash"]), json.dumps(["builtin:default"]), "en_US-lessac-medium"),
    )
    conn.commit()
    conn.close()


def test_upgrade_from_legacy_v3(tmp_path):
    path = tmp_path / "legacy.db"
    _legacy_db(path)
    database.configure(f"sqlite:///{path}")
    database.upgrade_sync()
    database.upgrade_sync()  # idempotent

    async def check():
        async with database.session() as s:
            assert (await s.get(Owner, 1)).passphrase_hash == "scrypt$hash"
            assert [p.id for p in (await s.execute(select(Profile))).scalars()] == [DEFAULT_PROFILE_ID]

            session = await s.get(CoachSession, "s1")
            assert session.profile_id == DEFAULT_PROFILE_ID
            assert session.current_mode == "investor_qa"
            assert session.updated_at.microsecond == 123456 and session.updated_at.tzinfo is not None

            msg = await s.get(Message, "m1")
            assert msg.scores_json == {"overall": 71}

            provider = await s.get(AiProvider, "p1")
            assert provider.models_json[0]["id"] == "gemini-2.5-flash" and provider.key_hint == "...1234"

            prefs = await s.get(AiSettings, 1)
            assert prefs.chat_models_json == ["p1:gemini-2.5-flash"]
            assert prefs.fallback_models_json == ["builtin:default"]
            assert prefs.task_models_json == {}
    run(check())

    tables = {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "speech_metrics" not in tables and "ai_preferences" not in tables
    assert {"profiles", "decks", "slides", "rehearsals", "goals", "game_runs"} <= tables


def test_messages_cascade_with_session(owner_client):
    sid = owner_client.post("/api/sessions", json={"title": "x"}).json()["id"]

    async def add_and_delete():
        async with database.session() as s:
            s.add(Message(session_id=sid, role="user", content="hi"))
        async with database.session() as s:
            await s.delete(await s.get(CoachSession, sid))
        async with database.session() as s:
            return (await s.execute(select(Message))).scalars().all()
    assert run(add_and_delete()) == []
