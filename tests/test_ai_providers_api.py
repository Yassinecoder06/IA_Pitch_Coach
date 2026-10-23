import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.storage.database import db
from backend.auth.owner import owner_service


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    test_db = tmp_path / "test_ai.db"
    db.sqlite_file = test_db
    db.is_postgres = False
    db.init_schema()
    code = owner_service.get_or_create_setup_code()
    owner_service.setup("TestOwnerPass123", code)


def test_ai_providers_lifecycle():
    client = TestClient(app)
    token = owner_service.create_session_token()
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Initially contains built-in catalog
    state = client.get("/api/ai/providers", headers=headers).json()
    assert "providers" in state
    assert "catalog" in state
    assert any(m["providerId"] == "builtin" for m in state["catalog"])

    # 2. Add custom Groq / OpenAI provider
    res = client.post("/api/ai/providers", headers=headers, json={
        "kind": "openai_compatible",
        "name": "Groq Cloud",
        "baseUrl": "https://api.groq.com/openai/v1",
        "apiKey": "gsk_test1234567890"
    })
    assert res.status_code == 200
    providers = res.json()["providers"]
    p = next(x for x in providers if x["name"] == "Groq Cloud")
    assert p["name"] == "Groq Cloud"
    assert p["has_key"] is True
    assert p["key_hint"] == "...7890"

    # 3. Save preferences
    pref_res = client.put("/api/ai/preferences", headers=headers, json={
        "chat_models": [f"{p['id']}:llama-3.3-70b-versatile"],
        "fallback_models": ["builtin:default"]
    })
    assert pref_res.status_code == 200
    saved_prefs = pref_res.json()
    assert len(saved_prefs["chat_models"]) == 1

    # 4. Delete provider
    del_res = client.delete(f"/api/ai/providers/{p['id']}", headers=headers)
    assert del_res.status_code == 200
    assert not any(x["id"] == p["id"] for x in del_res.json()["providers"])
