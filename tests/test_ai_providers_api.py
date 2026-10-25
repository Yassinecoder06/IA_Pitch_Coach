from backend.storage.provider_repository import provider_repository
from tests.conftest import run


def test_ai_providers_lifecycle(owner_client):
    client = owner_client

    state = client.get("/api/ai/providers").json()
    assert "providers" in state and "catalog" in state
    assert any(m["providerId"] == "builtin" for m in state["catalog"])
    assert state["tasks"] == ["chat", "judge", "deck_gen", "qa"]

    res = client.post("/api/ai/providers", json={
        "kind": "openai_compatible",
        "name": "Groq Cloud",
        "baseUrl": "http://127.0.0.1:9/v1",  # unreachable: discovery fails fast and falls back
        "apiKey": "gsk_test1234567890"
    })
    assert res.status_code == 200
    p = next(x for x in res.json()["providers"] if x["name"] == "Groq Cloud")
    assert p["has_key"] is True
    assert p["key_hint"] == "...7890"
    assert "api_key" not in p and "api_key_encrypted" not in p  # secrets never leave the server

    stored = run(provider_repository.get_provider(p["id"]))
    assert stored["api_key"] == "gsk_test1234567890"

    ref = f"{p['id']}:llama-3.3-70b-versatile"
    saved = client.put("/api/ai/preferences", json={
        "chat_models": [ref],
        "fallback_models": ["builtin:default"],
        "task_models": {"judge": ref},
    }).json()
    assert saved["chat_models"] == [ref]
    assert saved["task_models"] == {"judge": ref}
    assert client.put("/api/ai/preferences", json={"chat_models": [], "task_models": {"bogus": ref}}).status_code == 422

    del_res = client.delete(f"/api/ai/providers/{p['id']}")
    assert del_res.status_code == 200
    assert not any(x["id"] == p["id"] for x in del_res.json()["providers"])
    prefs = client.get("/api/ai/preferences").json()
    assert prefs["chat_models"] == [] and prefs["task_models"] == {}  # refs to the deleted provider are dropped


def test_ai_routes_require_unlock(client):
    from backend.auth.owner import owner_service
    run(owner_service.setup("CorrectHorse1", owner_service.get_or_create_setup_code()))
    assert client.get("/api/ai/providers").status_code == 401
