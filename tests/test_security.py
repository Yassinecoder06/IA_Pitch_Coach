import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.main import app
from backend.api.routers import auth as auth_router
from backend.config.settings import Settings, LEGACY_DEFAULT_SECRET
from backend.storage.database import db
from backend.storage import provider_repository as vault
from backend.auth.owner import owner_service, SESSION_COOKIE


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    db.sqlite_file = tmp_path / "test_security.db"
    db.is_postgres = False
    db.init_schema()
    owner_service._setup_code = None
    auth_router._failed_attempts = 0
    auth_router._locked_until = 0.0


def _configured_client() -> TestClient:
    client = TestClient(app)
    code = owner_service.get_or_create_setup_code()
    assert client.post("/api/auth/owner/setup", json={"passphrase": "CorrectHorse1", "setupCode": code}).status_code == 200
    return client


def test_secret_key_is_generated_and_persisted(tmp_path):
    s = Settings(DATA_DIR=str(tmp_path), SECRET_KEY=LEGACY_DEFAULT_SECRET)
    key = s.secret_key
    assert key != LEGACY_DEFAULT_SECRET and len(key) >= 48
    key_file = tmp_path / "secret.key"
    assert key_file.read_text() == key
    assert (key_file.stat().st_mode & 0o777) == 0o600
    assert Settings(DATA_DIR=str(tmp_path), SECRET_KEY="").secret_key == key


def test_explicit_secret_key_wins(tmp_path):
    assert Settings(DATA_DIR=str(tmp_path), SECRET_KEY="my-own-secret").secret_key == "my-own-secret"
    assert not (tmp_path / "secret.key").exists()


def test_legacy_encrypted_keys_are_migrated():
    legacy_cipher = vault.AESGCM(vault._derive_key(LEGACY_DEFAULT_SECRET))
    nonce = b"\x00" * 12
    blob = vault.base64.urlsafe_b64encode(nonce + legacy_cipher.encrypt(nonce, b"sk-legacy", None)).decode()
    db.execute(
        "INSERT INTO ai_providers (id, user_id, kind, name, api_key_encrypted) VALUES ('p1', 'u', 'openai', 'OpenAI', ?)",
        (blob,),
    )
    assert vault.migrate_legacy_encryption() == 1
    row = db.fetch_one("SELECT api_key_encrypted FROM ai_providers WHERE id = 'p1'")
    assert vault.decrypt_secret(row["api_key_encrypted"]) == "sk-legacy"
    assert vault.migrate_legacy_encryption() == 0


def test_websocket_rejects_missing_cookie():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 4401


def test_websocket_rejects_foreign_origin():
    client = _configured_client()
    with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 4403


def test_websocket_accepts_session_cookie_and_ignores_foreign_session():
    client = _configured_client()
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "session_id": "not-my-session"})
        ack = ws.receive_json()
    assert ack["type"] == "config_ack"
    assert ack["config"]["session_id"] is None


def test_tts_requires_auth():
    client = TestClient(app)
    code = owner_service.get_or_create_setup_code()
    owner_service.setup("CorrectHorse1", code)
    assert client.post("/api/tts/read-aloud", json={"text": "hi"}).status_code == 401


def test_login_backoff_after_repeated_failures():
    client = _configured_client()
    client.cookies.clear()
    for _ in range(auth_router.FREE_ATTEMPTS):
        assert client.post("/api/auth/owner/login", json={"passphrase": "wrong"}).status_code == 401
    assert client.post("/api/auth/owner/login", json={"passphrase": "CorrectHorse1"}).status_code == 429


def test_session_cookie_is_strict_httponly():
    client = TestClient(app)
    code = owner_service.get_or_create_setup_code()
    res = client.post("/api/auth/owner/setup", json={"passphrase": "CorrectHorse1", "setupCode": code})
    cookie_header = res.headers["set-cookie"].lower()
    assert SESSION_COOKIE in cookie_header
    assert "httponly" in cookie_header and "samesite=strict" in cookie_header
