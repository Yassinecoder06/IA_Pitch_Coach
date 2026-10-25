import pytest
from starlette.websockets import WebSocketDisconnect

from backend.api.routers import auth as auth_router
from backend.auth.owner import SESSION_COOKIE, owner_service
from backend.config.settings import LEGACY_DEFAULT_SECRET, Settings
from backend.db import database
from backend.db.models import AiProvider
from backend.storage import provider_repository as vault
from tests.conftest import OWNER_PASSPHRASE, run


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

    async def insert():
        async with database.session() as s:
            s.add(AiProvider(id="p1", kind="openai", name="OpenAI", api_key_encrypted=blob, models_json=[]))
    run(insert())

    assert run(vault.migrate_legacy_encryption()) == 1
    assert run(vault.provider_repository.get_provider("p1"))["api_key"] == "sk-legacy"
    assert run(vault.migrate_legacy_encryption()) == 0


def _close_code(client, **kwargs) -> int:
    with client.websocket_connect("/ws", **kwargs) as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    return exc.value.code


def test_websocket_rejects_missing_cookie(client):
    assert _close_code(client) == 4401


def test_websocket_rejects_foreign_origin(owner_client):
    assert _close_code(owner_client, headers={"origin": "https://evil.example"}) == 4403


def test_websocket_requires_profile(owner_client):
    owner_client.cookies.set(SESSION_COOKIE, run(owner_service.create_session_token(None)))
    assert _close_code(owner_client) == 4409


def test_websocket_ignores_foreign_session(owner_client):
    with owner_client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "session_id": "not-my-session"})
        ack = ws.receive_json()
    assert ack["type"] == "config_ack"
    assert ack["config"]["session_id"] is None


def test_tts_requires_auth(client):
    run(owner_service.setup(OWNER_PASSPHRASE, owner_service.get_or_create_setup_code()))
    assert client.post("/api/tts/read-aloud", json={"text": "hi"}).status_code == 401


def test_login_backoff_after_repeated_failures(owner_client):
    owner_client.cookies.clear()
    for _ in range(auth_router.login_backoff.free_attempts):
        assert owner_client.post("/api/auth/owner/login", json={"passphrase": "wrong"}).status_code == 401
    assert owner_client.post("/api/auth/owner/login", json={"passphrase": OWNER_PASSPHRASE}).status_code == 429


def test_session_cookie_is_strict_httponly(client):
    code = owner_service.get_or_create_setup_code()
    res = client.post("/api/auth/owner/setup", json={"passphrase": OWNER_PASSPHRASE, "setupCode": code})
    cookie_header = res.headers["set-cookie"].lower()
    assert SESSION_COOKIE in cookie_header
    assert "httponly" in cookie_header and "samesite=strict" in cookie_header


def test_change_passphrase_endpoint(owner_client):
    old_cookie = owner_client.cookies.get(SESSION_COOKIE)
    res = owner_client.post("/api/auth/owner/passphrase", json={"current": OWNER_PASSPHRASE, "new": "BrandNewPass9"})
    assert res.status_code == 200
    assert owner_client.get("/api/sessions").status_code == 200  # this browser got a fresh cookie
    owner_client.cookies.set(SESSION_COOKIE, old_cookie)
    assert owner_client.get("/api/sessions").status_code == 401  # others are signed out
