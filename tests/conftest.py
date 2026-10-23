import os
import sys

# Tests must never touch the real data/secret.key
os.environ.setdefault("SECRET_KEY", "test-only-secret-key")

# Ensure project root is in sys.path for test imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import asyncio  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.routers import auth as auth_router  # noqa: E402
from backend.api.routers import profiles as profiles_router
from backend.auth.owner import owner_service  # noqa: E402
from backend.db import DEFAULT_PROFILE_ID, database  # noqa: E402

OWNER_PASSPHRASE = "CorrectHorse1"


def run(coro):
    """Run a coroutine from a synchronous test."""
    return asyncio.run(coro)


# Set TEST_POSTGRES_URL=postgresql://user:pw@host:port/db to run the suite against PostgreSQL.
TEST_POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")


def _reset_postgres(url: str) -> None:
    import sqlalchemy as sa
    engine = sa.create_engine(url.replace("postgresql://", "postgresql+psycopg://", 1))
    with engine.begin() as conn:
        conn.execute(sa.text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    engine.dispose()


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Every test gets its own migrated database (SQLite by default) and clean auth state."""
    if TEST_POSTGRES_URL:
        _reset_postgres(TEST_POSTGRES_URL)
        database.configure(TEST_POSTGRES_URL, null_pool=True)
    else:
        database.configure(f"sqlite:///{tmp_path / 'test.db'}")
    database.upgrade_sync()
    owner_service.reset_cache()
    auth_router.login_backoff.reset()
    profiles_router.pin_backoff.reset()
    yield


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


@pytest.fixture
def owner_client(client):
    """A client that has claimed the install; the default profile is auto-selected."""
    code = owner_service.get_or_create_setup_code()
    res = client.post("/api/auth/owner/setup", json={"passphrase": OWNER_PASSPHRASE, "setupCode": code})
    assert res.status_code == 200, res.text
    return client


@pytest.fixture
def default_profile_id():
    return DEFAULT_PROFILE_ID
