from backend.auth.owner import SESSION_COOKIE, owner_service


def test_health_and_modes(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    modes = client.get("/api/modes").json()["modes"]
    assert len(modes) == 8


def test_auth_lifecycle(client):
    status = client.get("/api/auth/owner").json()
    assert not status["configured"]
    assert not status["authenticated"]

    code = owner_service.get_or_create_setup_code()
    setup_res = client.post("/api/auth/owner/setup", json={"passphrase": "OwnerSecretPass123", "setupCode": code})
    assert setup_res.status_code == 200
    assert setup_res.json()["authenticated"] is True
    assert "token" not in setup_res.json()  # only delivered as an HttpOnly cookie
    assert SESSION_COOKIE in client.cookies

    status = client.get("/api/auth/owner").json()
    assert status["authenticated"] and status["profile"]["name"] == "Me"  # sole profile auto-selected

    session_id = client.post("/api/sessions", json={"title": "Seed Round Pitch", "mode": "qa"}).json()["id"]
    sessions = client.get("/api/sessions").json()["sessions"]
    assert [s["id"] for s in sessions] == [session_id]
    assert sessions[0]["current_mode"] == "investor_qa"  # alias normalised

    assert client.patch(f"/api/sessions/{session_id}/rename", json={"title": "Series A"}).status_code == 200
    assert client.get(f"/api/sessions/{session_id}").json()["session"]["title"] == "Series A"

    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    assert client.get("/api/sessions").json()["sessions"] == []
    assert client.get(f"/api/sessions/{session_id}").status_code == 404

    assert client.post("/api/auth/owner/logout").status_code == 200
    assert client.get("/api/sessions").status_code == 401


def test_login_with_passphrase(owner_client):
    owner_client.cookies.clear()
    assert owner_client.post("/api/auth/owner/login", json={"passphrase": "nope"}).status_code == 401
    assert owner_client.post("/api/auth/owner/login", json={"passphrase": "CorrectHorse1"}).status_code == 200
    assert owner_client.get("/api/sessions").status_code == 200
