from backend.auth.owner import SESSION_COOKIE


def test_profiles_scope_data_and_pins(owner_client):
    c = owner_client
    me = c.get("/api/auth/owner").json()["profile"]
    my_session = c.post("/api/sessions", json={"title": "Mine"}).json()["id"]

    cofounder = c.post("/api/profiles", json={"name": "Co-founder", "color": "#10b981", "pin": "4321"}).json()
    assert cofounder["has_pin"] is True
    assert c.post("/api/profiles", json={"name": "Bad", "pin": "12"}).status_code == 422
    assert c.post("/api/profiles", json={"name": "Bad", "color": "red"}).status_code == 422

    # PIN required to switch
    assert c.post(f"/api/profiles/{cofounder['id']}/select", json={}).status_code == 403
    assert c.post(f"/api/profiles/{cofounder['id']}/select", json={"pin": "4321"}).status_code == 200
    assert c.get("/api/auth/owner").json()["profile"]["id"] == cofounder["id"]

    # Each profile sees only its own sessions
    assert c.get("/api/sessions").json()["sessions"] == []
    assert c.get(f"/api/sessions/{my_session}").status_code == 404
    assert c.delete(f"/api/sessions/{my_session}").status_code == 404

    assert c.post(f"/api/profiles/{me['id']}/select", json={}).status_code == 200
    assert [s["id"] for s in c.get("/api/sessions").json()["sessions"]] == [my_session]


def test_with_two_profiles_unlock_needs_a_choice(owner_client):
    c = owner_client
    c.post("/api/profiles", json={"name": "Second"})
    c.cookies.clear()
    assert c.post("/api/auth/owner/login", json={"passphrase": "CorrectHorse1"}).status_code == 200
    status = c.get("/api/auth/owner").json()
    assert status["authenticated"] and status["profile"] is None
    res = c.get("/api/sessions")
    assert res.status_code == 401 and res.json()["detail"]["code"] == "profile_required"
    assert c.get("/api/profiles").status_code == 200  # picker works without a profile


def test_delete_profile_cascades(owner_client):
    c = owner_client
    other = c.post("/api/profiles", json={"name": "Temp"}).json()
    c.post(f"/api/profiles/{other['id']}/select", json={})
    c.post("/api/sessions", json={"title": "Temp's"})
    assert c.delete(f"/api/profiles/{other['id']}").status_code == 200
    names = [p["name"] for p in c.get("/api/profiles").json()["profiles"]]
    assert names == ["Me"]
    # The only remaining profile has no PIN, so the session falls back to it.
    assert c.get("/api/auth/owner").json()["profile"]["name"] == "Me"
    assert c.get("/api/sessions").json()["sessions"] == []
    me_id = c.get("/api/auth/owner").json()["profile"]["id"]
    assert c.delete(f"/api/profiles/{me_id}").status_code == 400  # keep at least one


def test_pin_backoff(owner_client):
    c = owner_client
    p = c.post("/api/profiles", json={"name": "Locked", "pin": "1111"}).json()
    for _ in range(5):
        assert c.post(f"/api/profiles/{p['id']}/select", json={"pin": "0000"}).status_code == 403
    assert c.post(f"/api/profiles/{p['id']}/select", json={"pin": "1111"}).status_code == 429
    assert SESSION_COOKIE in c.cookies
