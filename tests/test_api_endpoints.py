import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.storage.database import db
from backend.auth.owner import owner_service, SESSION_COOKIE


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    test_db = tmp_path / "test_api.db"
    db.sqlite_file = test_db
    db.is_postgres = False
    db.init_schema()
    owner_service._setup_code = None


def test_health_and_modes():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    modes_res = client.get("/api/modes")
    assert modes_res.status_code == 200
    modes = modes_res.json()["modes"]
    assert len(modes) == 8


def test_auth_lifecycle():
    client = TestClient(app)

    # 1. Initially unconfigured
    status = client.get("/api/auth/owner").json()
    assert not status["configured"]
    assert not status["authenticated"]

    # 2. Setup owner
    code = owner_service.get_or_create_setup_code()
    setup_res = client.post("/api/auth/owner/setup", json={
        "passphrase": "OwnerSecretPass123",
        "setupCode": code
    })
    assert setup_res.status_code == 200
    assert setup_res.json()["authenticated"] is True
    token = setup_res.json()["token"]

    # 3. Create session with authenticated client
    headers = {"Authorization": f"Bearer {token}"}
    session_res = client.post("/api/sessions", headers=headers, json={
        "title": "Seed Round Pitch",
        "mode": "pitch_analysis"
    })
    assert session_res.status_code == 200
    session_id = session_res.json()["id"]

    # 4. List sessions
    list_res = client.get("/api/sessions", headers=headers)
    assert list_res.status_code == 200
    sessions = list_res.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id

    # 5. Delete session
    del_res = client.delete(f"/api/sessions/{session_id}", headers=headers)
    assert del_res.status_code == 200

    list_after = client.get("/api/sessions", headers=headers).json()["sessions"]
    assert len(list_after) == 0
